from backtest import *
from factors import *
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
pd.set_option('mode.chained_assignment', None)

mydb = mysql.connector.connect(
        host="172.31.50.91",
        user="guest",
        password="MH#123456",
        database="astocks"
        )

def last_day_of_last_quarter(current_date):
    quarters_finished = (current_date.month - 1) // 3
    return datetime.date(current_date.year, quarters_finished * 3 + 1, 1) + datetime.timedelta(days=-(1))

def query_SQL_finance(min_date, max_date, factors = [], stocks = []):

    factor_list = 'fd, codenum'
    if len(factors) != 0:
        factor_list = factor_list + ', ' + ', '.join(factors)

    if len(stocks) != 0:
        stock_list = ', '.join([f"'{stock}'" for stock in stocks])
        query = f"SELECT {factor_list} FROM finance WHERE fd BETWEEN {min_date} AND {max_date} and codenum IN ({stock_list}) ORDER BY fd ASC;"
    else:
        query = f"SELECT {factor_list} FROM finance WHERE fd BETWEEN {min_date} AND {max_date} ORDER BY fd ASC;"
    
    finance_df = pd.read_sql(query, mydb)
    finance_df['fd'] = finance_df['fd'].astype('str')
    return finance_df

def query_SQL_finance_deriv(min_date, max_date, factors = [], stocks = []):

    factor_list = 'fd, codenum'
    if len(factors) != 0:
        factor_list = factor_list + ', ' + ', '.join(factors)

    if len(stocks) != 0:
        stock_list = ', '.join([f"'{stock}'" for stock in stocks])
        query = f"SELECT {factor_list} FROM finance_deriv WHERE fd BETWEEN {min_date} AND {max_date} and codenum IN ({stock_list}) ORDER BY fd ASC;"
    else:
        query = f"SELECT {factor_list} FROM finance_deriv WHERE fd BETWEEN {min_date} AND {max_date} ORDER BY fd ASC;"
    
    finance_deriv_df = pd.read_sql(query, mydb)
    finance_deriv_df['fd'] = finance_deriv_df['fd'].astype('str')
    return finance_deriv_df

def query_SQL_company(stocks = []):
    #Gets the industry information of the stocks
    if len(stocks) != 0:
        stock_list = ', '.join([f"'{stock}'" for stock in stocks])
        query = f'SELECT codenum, SW_c1_name_CN AS industry FROM company WHERE codenum IN ({stock_list})'
    else:
        query = f'SELECT codenum, SW_c1_name_CN AS industry FROM company'
    
    return pd.read_sql(query, mydb)

def normalize(series):
    mean = series.mean()
    std = series.std()
    upper_lim = mean + 3 * std
    lower_lim = mean - 3 * std
    series.loc[series > upper_lim] = upper_lim
    series.loc[series < lower_lim] = lower_lim
    std2 = series.std()
    return (series - mean) / std2

def replace_duplicates_with_suffixes(lst):
    element_count = {}
    result = []

    for element in lst:
        if element in element_count:
            element_count[element] += 1
            result.append(f"{element}_{element_count[element]}")
        else:
            element_count[element] = 0
            result.append(element)

    return result

def calc_factors(start_date, end_date = datetime.date.today().strftime('%Y%m%d'), stocks = [], factors = {}, divide_groups = 5, group = False):

    if group and len(factors) > 1:
        raise Exception("Cannot handle grouping for more than one factor")

    #Get columns of finance table
    query = f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'astocks' AND TABLE_NAME = 'finance';"
    finance_columns = pd.read_sql(query, mydb)
    #Get columns of finance_deriv table
    query = f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'astocks' AND TABLE_NAME = 'finance_deriv';"
    finance_deriv_columns = pd.read_sql(query, mydb)
    #Get columns of market table
    query = f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'astocks' AND TABLE_NAME = 'market';"
    market_columns = pd.read_sql(query, mydb)

    fin_ind = set()
    fin_deriv_ind = set()
    market_ind = set()
    market_ind.add('chg') #Prepares for calculating everyday gain
    market_ind.add('market_cap') #Prepares for neutralization. 'market_cap' can be handled by read_SQL_market()
    for factor in factors:
        for indicator in factors[factor]['indicators']:
            if indicator not in ['td', 'fd', 'codenum', 'chg', 'market_cap', 'gain']:
                if indicator in finance_columns['COLUMN_NAME'].values:
                    fin_ind.add(indicator)
                elif indicator in finance_deriv_columns['COLUMN_NAME'].values:
                    fin_deriv_ind.add(indicator)
                elif indicator in market_columns['COLUMN_NAME'].values:
                    market_ind.add(indicator)
                else:
                    raise Exception(f'"{indicator}" does not exist in finance, finance_deriv or market tables!')

    finance_df = query_SQL_finance(start_date, end_date, factors = fin_ind, stocks = stocks)
    finance_deriv_df = query_SQL_finance_deriv(start_date, end_date, factors = fin_deriv_ind, stocks = stocks)
    market_df = query_SQL_market(start_date, end_date, indicators = market_ind, stocks = stocks)
    company_df = query_SQL_company(stocks = stocks)

    finance_merged = pd.merge(finance_df, finance_deriv_df, how = 'inner', on = ['fd', 'codenum'])
    finance_merged = pd.merge(finance_merged, company_df, how = 'inner', on = 'codenum')
    #stocks absent in any table are dropped

    # Convert 'td' column to datetime
    dates = pd.to_datetime(market_df['td'], format = '%Y%m%d')

    # Calculate the last day of last quarter using the vectorized function
    last_day_last_quarter = dates.apply(lambda date: last_day_of_last_quarter(date).strftime('%Y%m%d'))

    # Assign the results to the 'fd' column directly
    market_df['fd'] = last_day_last_quarter

    # last_day_last_quarter = []
    # for i in range(len(dates)):
    #     last_day_last_quarter.append(last_day_of_last_quarter(dates[i]).strftime('%Y%m%d'))
    # market_df['fd'] = last_day_last_quarter

    merged_df = pd.merge(market_df, finance_merged, how = 'inner', on = ['fd', 'codenum'])

    #Holds overnight
    merged_df['gain'] = (merged_df['chg']) / 100

    #Constructing factors
    factor_cols = []
    for factor in factors:
        print(f'Calculating {factor}...')
        ind_cols = [ind for ind in factors[factor]['indicators']]
        merged_ind_df = merged_df[ind_cols]
        #Address dulplicates in ind_cols
        ind_cols = replace_duplicates_with_suffixes(ind_cols)
        merged_ind_df.columns = ind_cols
        if 'td' in ind_cols or 'codenum' in ind_cols:
            if 'lag' in factors[factor]:
                raise Exception(f'Indicator list containing td or codenum does not support lagging yet!')
        else:
            merged_ind_df = pd.concat([merged_ind_df, merged_df['td']], axis = 1)
            merged_ind_df = pd.concat([merged_ind_df, merged_df['codenum']], axis = 1)

        
        if 'lag' in factors[factor]:
            if len(factors[factor]['lag']) != len(ind_cols):
                raise Exception('"lag" list length does not match factor list length!')
            else:
                for i in range(len(ind_cols)):
                    #Group by codenum first, then shift
                    merged_ind_df[ind_cols[i]] = merged_ind_df.groupby('codenum')[ind_cols[i]].shift(factors[factor]['lag'][i])
            
        if 'function' in factors[factor]:
            merged_df[factor] = factors[factor]['function'](merged_ind_df[ind_cols])
        else:
            if len(ind_cols) == 1:
                merged_df[factor] = merged_ind_df[ind_cols[0]]
            else:
                raise Exception(f'indicators {ind_cols} missing a combination function!')

        
        if len(merged_df) < 20:
            raise Exception(f'Backtest time span is too short for factor {factor}!')

        #industry neutralization
        merged_df[f'factor_{factor}'] = merged_df[[factor, 'industry']].groupby('industry').transform(normalize)

        #market-cap neutralization
        linregress_market_cap = LinearRegression()
        linregress_market_cap.fit(merged_df.dropna(subset = ['market_cap', 'factor_' + factor])['market_cap'].values.reshape(-1, 1), merged_df.dropna(subset = ['market_cap', 'factor_' + factor])['factor_' + factor])
        merged_df[f'factor_{factor}'] = merged_df.dropna(subset = ['market_cap', 'factor_' + factor])[f'factor_{factor}'] - linregress_market_cap.predict(merged_df.dropna(subset = ['market_cap', 'factor_' + factor])['market_cap'].values.reshape(-1, 1))
        factor_cols.append(f'factor_{factor}')

        merged_df = merged_df.copy()
    
    merged_df = merged_df[['td', 'codenum', 'gain'] + factor_cols]

    #Drops NA from missing factor values due to lagged factors or missing data
    print('Null value counts:')
    print(merged_df.isnull().sum())
    merged_df = merged_df.dropna().reset_index(drop = True)

    merged_df = merged_df.sort_values('td') # Ensures it is sorted by td
    merged_df['gain_next'] = merged_df.groupby('codenum')['gain'].shift(-1)

    if not group:
        merged_df.to_csv('factors.csv', index = False)
        return merged_df
    
    else: #single factor for testing
        concat_sub_df = pd.DataFrame()
        for td in merged_df['td'].unique():
            sub_df = merged_df[['td', 'codenum', f'factor_{factor}']][merged_df['td'] == td].sort_values(f'factor_{factor}', ascending = True)
            group_size = len(sub_df) // divide_groups
            sub_df['group'] = 1 #Assign all stocks to group 1
            for i in range(1, divide_groups):
                sub_df['group'].iloc[:(divide_groups - i) * group_size] = i + 1
            concat_sub_df = pd.concat([concat_sub_df, sub_df], ignore_index = True)
        grouped_merged_df = merged_df.merge(concat_sub_df.drop(f'factor_{factor}', axis = 1), on = ['td', 'codenum'])

        print(grouped_merged_df)
        return grouped_merged_df

def t_test(group_merged_df):
    test_df = group_merged_df[['group', 'gain_next']]
    test_df['date'] = pd.to_datetime(group_merged_df['td'], format = '%Y%m%d')

    test_df['year_month'] = test_df['date'].dt.strftime('%Y%m')

    divide_groups = test_df['group'].max()
    all_dates = test_df['year_month'].unique()
    t_list = []

    for date in all_dates: #Adjusts stock holdings every day
        sub_df = test_df.loc[test_df['year_month'] == date]
        highest_group = sub_df.loc[sub_df['group'] == 1]
        lowest_group = sub_df.loc[sub_df['group'] == divide_groups]
        t_result = ttest_ind(highest_group['gain_next'], lowest_group['gain_next'])
        t_list.append(t_result.statistic)

    result = pd.DataFrame()
    result['year_month'] = all_dates.astype('str')
    result['t_value'] = t_list
    result['abs>2'] = pd.Series(t_list).abs() > 2
    
    plt.figure(figsize = (10, 5))
    sns.barplot(x = 'year_month', y = 't_value', hue = 'abs>2', data = result)
    step = len(result) // 10
    plt.xticks(range(0, len(result), step), result['year_month'][::step], rotation=45)
    plt.tight_layout()
    plt.savefig('t_test.png')
    plt.show()

    significant = result['abs>2'].sum() / len(result)
    print('{:.2f}% of t values are greater than 2!'.format(significant * 100))

def grouped_backtest(group_merged_df):
    #Adjusts holdings every day
    result = pd.DataFrame()
    annual_profits = []

    for i in range(1, group_merged_df['group'].max() + 1):
        sub_df = group_merged_df.loc[group_merged_df['group'] == i]
        avg_daily_profit = sub_df.groupby('td')['gain_next'].mean()
        cumulative = [1 + avg_daily_profit.iloc[0]]

        for j in range(1, len(avg_daily_profit)):
            cumulative.append(cumulative[j - 1] * (1 + avg_daily_profit.iloc[j]))

        group_result = pd.DataFrame()
        group_result['date'] = pd.to_datetime(avg_daily_profit.index, format = '%Y%m%d')
        group_result['daily_profit'] = list(avg_daily_profit)
        group_result['cumulative_profit'] = cumulative
        group_result['group'] = i
        result = pd.concat([result, group_result], ignore_index = True)

        annual_profits.append(cumulative[-1] ** (260 / len(cumulative)) - 1)
    
    print(f'Annual profits of Groups starting from Group 1 (greatest factor value) are {annual_profits}')
    plt.figure(figsize = (10, 5))
    sns.lineplot(x = 'date', y = 'cumulative_profit', data = result, hue = 'group')
    plt.savefig('grouped_backtest.png')
    plt.show()

def calc_all_factors(start_date, stocks = []):
    calc_factors(start_date, factors = get_all_factors(), stocks = stocks)
    print('All factors calculated!')

def test_factor(start_date, end_date, factor_dict, stocks = []):
    group_merged_df = calc_factors(start_date, end_date = end_date, factors = factor_dict, stocks = stocks, group = True)
    t_test(group_merged_df)
    grouped_backtest(group_merged_df)

if __name__ == '__main__':

    start_date = 20211231
    end_date = 20230401

    if start_date >= end_date:
        raise ValueError('Date Error!')

    stocks_tested = random_stocks(500, start_date, end_date)

    test_factor(start_date, end_date, get_FR_factor(), stocks = csi300_stocks())
