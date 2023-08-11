from factor_test import *
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

def predict_gain(df):
    factor_cols = df.filter(like = 'factor_').columns
    gain_regression = LinearRegression()
    gain_regression.fit(df[factor_cols], df['gain_next'])
    df['predicted_gain'] = gain_regression.predict(df[factor_cols])
    df['residual'] = df['gain_next'] - df['predicted_gain']
    r2 = r2_score(df['gain_next'], df['predicted_gain'])
    n = len(df)
    k = len(factor_cols)
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
    df['adjusted_r2'] = adjusted_r2
    return df, gain_regression

def factor_regression(factors_df_or_filepath):

    if type(factors_df_or_filepath) == str:
        merged_df = pd.read_csv(factors_df_or_filepath)
    elif type(factors_df_or_filepath) == pd.DataFrame:
        merged_df = factors_df_or_filepath
    else:
        raise Exception('Please input a pandas DataFrame or its filepath!')
    
    gain_regress_series = []
    sub_dfs = []
    for td, group_data in merged_df.groupby('td'):
        sub_df, sub_gain_regress = predict_gain(group_data)
        sub_dfs.append(sub_df)
        gain_regress_series.append(sub_gain_regress)
    merged_df = pd.concat(sub_dfs)

    residual_df = merged_df[['td', 'codenum', 'residual']]

    return gain_regress_series, residual_df, merged_df['adjusted_r2'].mean()

def select_factors():

    factor_pool = pd.read_csv('factors.csv')
    factor_pool.dropna(subset = ['gain_next'], inplace = True) #Drops the most current date because gain_next is to be predicted
    selected_td = factor_pool['td'].unique()[-260::5] # Select a day every week from the past year
    factor_pool = factor_pool.loc[factor_pool['td'].isin(selected_td)]
    factor_pool['td'].astype('str')

    factor_group_list = get_factor_group_list()

    #Round 1
    group_num = 0
    selected_cols = []
    vif = [1]
    for group in factor_group_list:
        group_num += 1
        factors_in_group = group.keys()

        group_remaining_cols = ['factor_' + factor_name for factor_name in factors_in_group]
        group_selected_cols = []
        max_r2 = 0
        flag = True
        while flag:
            flag = False
            for col in group_remaining_cols:
                print('Selected factors:', group_selected_cols)
                print('Testing factor:', col)

                # Factors with a significant portion of missing values are penalized by a lower r2.
                r2 = factor_regression(factor_pool[['td', 'codenum', 'gain_next'] + group_selected_cols + [col]].fillna(0))[2]
                print('Adjusted r-squared: {:.4f}/{:.4f}\n'.format(r2, max_r2))

                if r2 > max_r2 + 0.005: # Reduce the number of factors
                    max_r2 = r2
                    best_col = col
                    flag = True

            if flag == True:
                group_remaining_cols.remove(best_col)
                group_selected_cols.append(best_col)
                
                if len(group_selected_cols) > 1:
                    vif = [variance_inflation_factor(factor_pool[group_selected_cols].dropna(axis = 0).values, i) for i in range(len(group_selected_cols))]
                    if max(vif) > 5:
                        flag = False
                if len(group_remaining_cols) == 0:
                    flag = False
        
        if len(group_selected_cols) > 0:
            selected_cols += group_selected_cols
            print(f'Factors from group {group_num} selected!\nAdjusted r-squared: {max_r2}\nMax variance inflation factor: {max(vif)}\n')
        else:
            print(f'None of the factors from group {group_num} is selected.')

    #Round 2
    max_r2 = factor_regression(factor_pool[['td', 'codenum', 'gain_next'] + selected_cols].fillna(0))[2]
    last_rem_r2 = max_r2
    flag = True

    while flag:
        flag = False
        vif = [variance_inflation_factor(factor_pool[selected_cols].dropna(axis = 0).values, i) for i in range(len(selected_cols))]
        print('\nMax variance inflation factor: {:.4f}\n'.format(max(vif)))
        rem_r2 = 0
        for col in selected_cols:
            print('Selected factors:', selected_cols)
            print('Testing factor:', col)
            r2 = factor_regression(factor_pool[['td', 'codenum', 'gain_next'] + selected_cols].drop(col, axis = 1).fillna(0))[2]
            print('Adjusted r-squared: {:.4f}/{:.4f}'.format(r2, rem_r2))
            
            if r2 > rem_r2:
                worst_col = col
                rem_r2 = r2
                if r2 > max_r2:
                    max_r2 = r2
                    flag = True

        if flag == True or max(vif) > 10:
            selected_cols.remove(worst_col)
            print(f'{worst_col} dropped!')
            flag = True
            last_rem_r2 = rem_r2

    print(f'Factors selected!\nAdjusted r-squared: {last_rem_r2}\nVariance inflation factor: {max(vif)}\n')
    
    # Must be included for market cap neutralization
    if 'factor_ln_market_cap' not in selected_cols:
        selected_cols.append('factor_ln_market_cap')

    # Create 'factors_selected.csv'
    factor_pool = pd.read_csv('factors.csv')
    print(selected_cols)
    factors_selected = factor_pool[['td', 'codenum', 'gain_next'] + selected_cols].dropna(subset=selected_cols)

    # # Decolinearization. Needless now because maximum VIF is less than 5.
    # appended_cols = []
    # for col in selected_cols:
    #     if len(appended_cols) > 0:
    #         factor_colinear = LinearRegression()
    #         factor_colinear.fit(factor_pool[appended_cols], factor_pool[col])
    #         res = factor_pool[col] - factor_colinear.predict(factor_pool[appended_cols])
    #         factors_selected[col] = res
    #     else:
    #         factors_selected[col] = factor_pool[col]
    #     appended_cols.append(col)

    factors_selected.to_csv('factors_selected.csv', index = False)
    print('factors_selected.csv filled!')

def factor_regression_history():

    factors_selected = pd.read_csv('factors_selected.csv').dropna(subset = ['gain_next'])
    selected_cols = factors_selected.filter(like = 'factor_').columns
    # Calculate factor return history
    gain_regress_series, residual_df = factor_regression(factors_selected)[0:2]
    factor_return_df = np.array([(gain_regress.intercept_, *gain_regress.coef_) for gain_regress in gain_regress_series])
    factor_return_df = pd.DataFrame(data = factor_return_df)
    factor_return_df.columns = ['intercept'] + list(selected_cols)
    factor_return_df['td'] = factors_selected['td'].unique()
    factor_return_df.to_csv('factor_return.csv', index = False)
    print('factor_return.csv created!')

    residual_df.to_csv('residual.csv', index = False)
    print('residual.csv created!')

def update_factor_history(start_date, stocks = []):
    selected_cols = pd.read_csv('factors_selected.csv').filter(like = 'factor_').columns
    selected_factors = [col.replace('factor_', '') for col in selected_cols]
    calc_factors(start_date, factors = {factor: get_all_factors()[factor] for factor in selected_factors}, stocks = stocks)[['td', 'codenum', 'gain_next'] + selected_cols].to_csv('selected_factors.csv')
    print('Selected factors updated!')
    factor_regression_history()

def reselect_factors(start_date, stocks = []):
    calc_all_factors(start_date, stocks = stocks)
    select_factors()
    factor_regression_history()

if __name__ == '__main__':

    start_date = 20210101
    end_date = 20230730

    if start_date >= end_date:
        raise ValueError('Date Error!')
    
    random.seed(4)
    stocks_tested = list(set(list(random_stocks(300, start_date, end_date)) + list(csi300_stocks())))
    reselect_factors(start_date, stocks = stocks_tested)