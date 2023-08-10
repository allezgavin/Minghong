import numpy as np
import pandas as pd
from backtest import *
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
pd.set_option('mode.chained_assignment', None)
td_group = {}
codenum_group = {}

def abs(x):
    return np.abs(x)

def log(x):
    return np.log(x)

def sign(x):
    return np.sign(x)

def delta(x, d):
    output = pd.Series()
    for indices in codenum_group.values():
        sub_x = x.loc[indices]
        output = pd.concat([output, sub_x - sub_x.shift(d)])
    return output.reindex(x.index)

def ts_sum(x, d):
    output = pd.Series()
    for indices in codenum_group.values():
        sub_x = x.loc[indices]
        output = pd.concat([output, sub_x.rolling(window = d, min_periods = d).sum()])
    return output.reindex(x.index)

def ts_rank(x, d):
    output = pd.Series()
    for indices in codenum_group.values():
        sub_x = x.loc[indices]
        output = pd.concat([output, sub_x.rolling(window = d, min_periods = d).apply(lambda x: (x.argsort().argsort().iloc[-1] + 1), raw=False)])
    return output.reindex(x.index)

def ts_min(x, d):
    output = pd.Series()
    for indices in codenum_group.values():
        sub_x = x.loc[indices]
        output = pd.concat([output, sub_x.rolling(window = d, min_periods = d).min()])
    return output.reindex(x.index)

def ts_max(x, d):
    output = pd.Series()
    for indices in codenum_group.values():
        sub_x = x.loc[indices]
        output = pd.concat([output, sub_x.rolling(window = d, min_periods = d).max()])
    return output.reindex(x.index)

def ts_std(x, d):
    output = pd.Series()
    for indices in codenum_group.values():
        sub_x = x.loc[indices]
        output = pd.concat([output, sub_x.rolling(window = d, min_periods = d).std()])
    return output.reindex(x.index)

def ts_argmax(x, d):
    output = pd.Series()
    for indices in codenum_group.values():
        sub_x = x.loc[indices]
        output = pd.concat([output, sub_x.rolling(window = d, min_periods = d).apply(lambda x: np.argmax(x) + 1, raw=False)])
    return output.reindex(x.index)

def ts_argmin(x, d):
    output = pd.Series()
    for indices in codenum_group.values():
        sub_x = x.loc[indices]
        output = pd.concat([output, sub_x.rolling(window = d, min_periods = d).apply(lambda x: np.argmin(x) + 1, raw=False)])
    return output.reindex(x.index)

def rank(x):
    output = pd.Series()
    for indices in td_group.values():
        sub_x = x.loc[indices]
        output = pd.concat([output, sub_x.rank()])
    return output.reindex(x.index)

def delay(x, d):
    output = pd.Series()
    for indices in codenum_group.values():
        sub_x = x.loc[indices]
        output = pd.concat([output, sub_x.shift(d)])
    return output.reindex(x.index)

def correlation(x, y, d):
    output = pd.Series()
    for indices in codenum_group.values():
        sub_x = x.loc[indices]
        sub_y = y.loc[indices]
        result = sub_x.rolling(window = d, min_periods = d).corr(sub_y)

        # Rolling correlation calculation may result in infinities rather than NaN.
        # See https://github.com/pandas-dev/pandas/issues/29264
        result.loc[(result > 1) | (result < -1)] = np.nan

        output = pd.concat([output, result])
    return output.reindex(x.index)

def covariance(x, y, d):
    output = pd.Series()
    for indices in codenum_group.values():
        sub_x = x.loc[indices]
        sub_y = y.loc[indices]
        output = pd.concat([output, sub_x.rolling(window = d, min_periods = d).cov(sub_y)])
    return output.reindex(x.index)

def scale(x, a=1):
    output = pd.Series()
    for indices in td_group.values():
        sub_x = x.loc[indices]
        output = pd.concat([output, sub_x * (a / np.sum(np.abs(x)))])
    return output.reindex(x.index)

def signedpower(x, a):
    return np.power(x, a)

def decay_linear(x, d):
    weights = np.arange(d, 0, -1)
    output = pd.Series()
    for indices in codenum_group.values():
        sub_x = x.loc[indices]
        output = pd.concat([output, np.convolve(sub_x, weights / weights.sum(), mode='valid')])
    return output.reindex(x.index)

def get_basic_factors():
    basic_factors = {}
    basic_factors['PE'] = {'indicators': ['close', 'EPS'], 'function': lambda df: df['close'] / df['EPS']}
    basic_factors['netprofitrate'] = {'indicators': ['netprofitrate']}
    return basic_factors

# # Does not apply under market cap neutralization
# def get_size_factor():
#     size_factor = {}
#     size_factor['ln_capital'] = {'indicators': ['market_cap'], 'function': lambda x: np.log(x.iloc[:, 0])}
#     return size_factor

def get_value_factors():
    value_factors = {}
    value_factors['EP'] = {'indicators': ['net_profit', 'market_cap'], 'function': lambda df: df['net_profit'] / df['market_cap']}
    value_factors['EPCut'] = {'indicators': ['deducted_profit', 'market_cap'], 'function': lambda df: df['deducted_profit'] / df['market_cap']}
    value_factors['BP'] = {'indicators': ['total_assets', 'market_cap'], 'function': lambda df: df['total_assets'] / df['market_cap']}
    value_factors['SP'] = {'indicators': ['operating_revenue', 'market_cap'], 'function': lambda df: df['operating_revenue'] / df['market_cap']}
    value_factors['OCFP'] = {'indicators': ['OCFPS']}
    value_factors['NCFP'] = {'indicators': ['net_operating_cashflow', 'net_invest_cashflow', 'net_finance_cashflow', 'market_cap'],
                            'function': lambda df : df[['net_operating_cashflow', 'net_invest_cashflow', 'net_finance_cashflow']].sum(axis = 1) / df['market_cap']}
    return value_factors

def get_growth_factors():
    growth_factors = {}
    growth_factors['sales_growth_ttm'] = {'indicators': ['operating_revenue'], 'function': lambda df: delta(df['operating_revenue'], 260)}
    growth_factors['profit_growth_ttm'] = {'indicators': ['deducted_profit'], 'function': lambda df: delta(df['deducted_profit'], 260)}
    growth_factors['operationcashflow_growth_ttm'] = {'indicators': ['net_operating_cashflow'], 'function': lambda df: delta(df['net_operating_cashflow'], 260)}
    return growth_factors

def get_financial_quality_factors():
    financial_quality_factors = {} #TTM or of that quarter?
    financial_quality_factors['ROE'] = {'indicators': ['net_profit', 'total_shareholders_equity'], 'function': lambda df: df['net_profit'] / df['total_shareholders_equity']}
    financial_quality_factors['ROA'] = {'indicators': ['net_profit', 'total_assets'], 'function': lambda df: df['net_profit'] / df['total_assets']}
    financial_quality_factors['grossprofitmargin'] = {'indicators': ['grossmargin']}
    financial_quality_factors['profitmargin'] = {'indicators': ['operating_profit', 'operating_revenue'], 'function': lambda df: df['operating_profit'] / df['operating_revenue']}
    financial_quality_factors['assetturnover'] = {'indicators': ['operating_revenue', 'total_assets'], 'function': lambda df: df['operating_revenue'] / df['total_assets']}
    financial_quality_factors['operationcashflowratio'] = {'indicators': ['net_operating_cashflow', 'net_profit'], 'function': lambda df: df['net_operating_cashflow'] / df['net_profit']}
    return financial_quality_factors

def get_leverage_factors():
    leverage_factors = {}
    leverage_factors['market_value_leverage'] = {'indicators': ['market_cap', 'total_noncurrent_liabilities'], #also preferred stocks should be included
                                                'function': lambda x: (x.iloc[:, 0] + x.iloc[:, 1]) / x.iloc[:, 0]}
    leverage_factors['financial_leverage'] = {'indicators': ['total_assets', 'total_shareholders_equity'], 'function': lambda df: df['total_assets'] / df['total_shareholders_equity']}
    leverage_factors['debtequityratio'] = {'indicators': ['total_noncurrent_liabilities', 'total_shareholders_equity'], 'function': lambda df: df['total_noncurrent_liabilities'] / df['total_shareholders_equity']}
    leverage_factors['cashratio'] = {'indicators': ['cash', 'account_receivable', 'total_current_liabilities'],
                                    'function': lambda df: (df['cash'] + df['account_receivable']) / df['total_current_liabilities']}
    leverage_factors['currentratio'] = {'indicators': ['total_current_assets', 'total_current_liabilities'], 'function': lambda df: df['total_current_assets'] / df['total_current_liabilities']}

    return leverage_factors

# def get_size_factor():
#     return size_factor

def HAlpha(df):
    months = 6
    # Alpha of every stock in the past 6 months.
    start_date = df['td'].min()
    df['td'] = df['td'].astype('str')
    bench_df = query_SQL_csi300(start_date)[['td', 'gain']]
    bench_df.columns = ['td', 'gain_benchmark']
    df = pd.merge(df, bench_df, how = 'left', on = 'td')
    output = pd.Series()
    for stock, sub_df in df.groupby('codenum'):
        sub_df['var_bench'] = sub_df['gain_benchmark'].rolling(window = 21 * months, min_periods = 21 * months).var()
        sub_df['cov'] = sub_df['gain'].rolling(window = 21 * months, min_periods = 21 * months).cov(sub_df['gain_benchmark'])
        sub_df['beta'] = sub_df['cov'] / sub_df['var_bench']
        sub_df['bench_annual_profit'] = sub_df['gain_benchmark'].rolling(window = 21 * months, min_periods = 21 * months).sum()
        sub_df['annual_profit'] = sub_df['gain'].rolling(window = 21 * months, min_periods = 21 * months).sum()
        sub_df['alpha'] = sub_df['annual_profit'] - sub_df['beta'] * sub_df['bench_annual_profit']
        output = pd.concat([output, sub_df['alpha']])
    return output.reindex(df.index)

def get_momentum_factors():
    momentum_factors = {}
    momentum_factors['HALpha'] = {'indicators': ['td', 'codenum', 'gain'], 'function': HAlpha}
    momentum_factors['relative_strength_1m'] = {'indicators': ['close'], 'function': lambda df: delta(df['close'], 1 * 21)}
    return momentum_factors

def beta_consistency(df):
    months = 6
    # beta times summed square of regression residuals.
    start_date = df['td'].min()
    df['td'] = df['td'].astype('str')
    bench_df = query_SQL_csi300(start_date)[['td', 'gain']]
    bench_df.columns = ['td', 'gain_benchmark']
    df = pd.merge(df, bench_df, how = 'left', on = 'td')
    output = pd.Series()
    for stock, sub_df in df.groupby('codenum'):
        sub_df[['weekly_gain', 'weekly_benchmark']] = sub_df[['gain', 'gain_benchmark']].rolling(window = 5, min_periods = 5).sum()
        sub_df['var_bench'] = sub_df['weekly_benchmark'].rolling(window = 21 * months, min_periods = 21 * months).var()
        sub_df['cov'] = sub_df['weekly_gain'].rolling(window = 21 * months, min_periods = 21 * months).cov(sub_df['weekly_benchmark'])
        sub_df['beta'] = sub_df['cov'] / sub_df['var_bench']
        sub_df['residual^2'] = np.nan
        for i in range(21 * months, len(sub_df)):
            sub_df['residual^2'].iloc[i] = np.square((sub_df['beta'].iloc[i] * sub_df['gain_benchmark'].iloc[i - 21 * months: i] - sub_df['gain'].iloc[i - 21 * months: i])).sum()
        sub_df['beta_consistency'] = sub_df['beta'] * sub_df['residual^2']
        output = pd.concat([output, sub_df['beta_consistency']])
    return output.reindex(df.index)

def get_volatility_factors():
    volatility_factors = {}
    volatility_factors['high_low_1m'] = {'indicators': ['high', 'low'], 'function': lambda df: ts_max(df['high'], 1 * 21) / ts_max(df['low'], 1 * 21)}
    volatility_factors['high_low_2m'] = {'indicators': ['high', 'low'], 'function': lambda df: ts_max(df['high'], 2 * 21) / ts_max(df['low'], 2 * 21)}
    volatility_factors['high_low_3m'] = {'indicators': ['high', 'low'], 'function': lambda df: ts_max(df['high'], 3 * 21) / ts_max(df['low'], 3 * 21)}
    volatility_factors['high_low_6m'] = {'indicators': ['high', 'low'], 'function': lambda df: ts_max(df['high'], 6 * 21) / ts_max(df['low'], 6 * 21)}
    volatility_factors['high_low_12m'] = {'indicators': ['high', 'low'], 'function': lambda df: ts_max(df['high'], 12 * 21) / ts_max(df['low'], 12 * 21)}

    volatility_factors['std_1m'] = {'indicators': ['high'], 'function': lambda df: ts_std(df['high'], 6 * 21)}
    volatility_factors['std_2m'] = {'indicators': ['high'], 'function': lambda df: ts_std(df['high'], 6 * 21)}
    volatility_factors['std_3m'] = {'indicators': ['high'], 'function': lambda df: ts_std(df['high'], 6 * 21)}
    volatility_factors['std_6m'] = {'indicators': ['high'], 'function': lambda df: ts_std(df['high'], 6 * 21)}
    volatility_factors['std_12m'] = {'indicators': ['high'], 'function': lambda df: ts_std(df['high'], 6 * 21)}
    volatility_factors['ln_price'] = {'indicators': ['close'], 'function': lambda df: np.log(df['close'])}
    volatility_factors['beta_consistency'] = {'indicators': ['td', 'codenum', 'gain'], 'function': beta_consistency}
    return volatility_factors

def get_turnover_factors():
    turnover_factors = {}
    turnover_factors['turnover_1m'] = {'indicators': ['vol' ,'total_share'], 'function': lambda df: ts_sum(df['vol'], 1 * 21) / df['total_share']}
    turnover_factors['turnover_2m'] = {'indicators': ['vol' ,'total_share'], 'function': lambda df: ts_sum(df['vol'], 2 * 21) / df['total_share']}
    turnover_factors['turnover_3m'] = {'indicators': ['vol' ,'total_share'], 'function': lambda df: ts_sum(df['vol'], 3 * 21) / df['total_share']}
    turnover_factors['turnover_6m'] = {'indicators': ['vol' ,'total_share'], 'function': lambda df: ts_sum(df['vol'], 6 * 21) / df['total_share']}
    turnover_factors['turnover_12m'] = {'indicators': ['vol' ,'total_share'], 'function': lambda df: ts_sum(df['vol'], 12 * 21) / df['total_share']}
    return turnover_factors

def get_modified_momentum_factors():
    modified_momentum_factors = {}
    modified_momentum_factors['weighted_strength_1m'] = {'indicators': ['gain', 'vol'], 'function': lambda df: ts_sum(df['gain'] * df['vol'], 1 * 21)}
    modified_momentum_factors['weighted_strength_2m'] = {'indicators': ['gain', 'vol'], 'function': lambda df: ts_sum(df['gain'] * df['vol'], 2 * 21)}
    modified_momentum_factors['weighted_strength_3m'] = {'indicators': ['gain', 'vol'], 'function': lambda df: ts_sum(df['gain'] * df['vol'], 3 * 21)}
    modified_momentum_factors['weighted_strength_6m'] = {'indicators': ['gain', 'vol'], 'function': lambda df: ts_sum(df['gain'] * df['vol'], 6 * 21)}
    modified_momentum_factors['weighted_strength_12m'] = {'indicators': ['gain', 'vol'], 'function': lambda df: ts_sum(df['gain'] * df['vol'], 12 * 21)}
    return modified_momentum_factors


def codenum_adjacency(df):
    output = pd.Series()
    for sub_df in df.sort_values('codenum').groupby('td'):
        sub_df['adjacent_gain'] = sub_df['gain'].rolling(window = 5).sum()
        output = pd.concat([output, sub_df['adjacent_gain']])
    return output.reindex(df.index)

def get_codenum_factor():
    codenum_factor = {}
    codenum_factor['codenum_adjacency'] = {'indicators': ['td', 'codenum', 'gain'], 'function': codenum_adjacency}
    return codenum_factor

def ranked_chg_div(df):
    n = 21
    return -1 * correlation(rank(delta(df['vol'], 1)), rank(delta(df['close'], 1)), n)

def ts_ranked_chg_div(df):
    m = 10
    k = 10
    n = 21
    return -1 * correlation(ts_rank(delta(df['vol'], 1), m), ts_rank(delta(df['close'], 1), k), n)

def get_volume_price_factors():
    volume_price_factors = {}
    volume_price_factors['FR'] = {'indicators': ['vol', 'total_share', 'gain'], 'function': lambda df: correlation(df['vol'] / df['total_share'], df['gain'], 21)}
    volume_price_factors['vp_div'] = {'indicators': ['vol', 'close'], 'function': lambda df: correlation(df['vol'], df['close'], 21)}
    volume_price_factors['ranked_vp_chg_div'] = {'indicators': ['vol', 'close'], 'function': lambda df: correlation(rank(df['vol']), rank(df['close']), 21)}
    volume_price_factors['vp_chg_div'] = {'indicators': ['vol', 'close'], 'function': lambda df: correlation(delta(df['vol'], 1), delta(df['close'], 1), 21)}
    volume_price_factors['ranked_vp_chg_div'] = {'indicators': ['vol', 'close'], 'function': ranked_chg_div}
    volume_price_factors['ts_ranked_vp_chg_div'] = {'indicators': ['vol', 'close'], 'function': ts_ranked_chg_div}
    return volume_price_factors

def WQ001(df):
    return rank(ts_argmax(signedpower(pd.Series(data =
                            np.where(
                                df['gain'] < 0,
                                ts_std(df['gain'], 20),
                                df['close']
                            ), index = df.index)
                            , 2), 5)) - 0.5


def WQ002(df):
    return -1 * correlation(rank(delta(log(df['vol']), 2)), rank(((df['close'] - df['open']) / df['open'])), 6)

def WQ003(df):
    return (-1 * correlation(rank(df['open']), rank(df['vol']), 10))

def WQ004(df):
    return -1 * ts_rank(rank(df['low']), 9)

# Does not apply without VWAP data
# def WQ005(df):
#     return (rank((open - (ts_sum(df['vwap'], 10) / 10))) * (-1 * abs(rank((df['close'] - df['vwap'])))))

def WQ006(df):
    return -1 * correlation(df['open'], df['vol'], 10)

def WQ007(df):
    df['adv20'] = df.groupby('codenum')['vol'].transform(lambda x: x.rolling(window=20, min_periods=20).mean())
    return np.where(df['adv20'] < df['vol'],
        ((-1 * ts_rank(abs(delta(df['close'], 7)), 60)) * sign(delta(df['close'], 7))),
        -1
    )

def WQ008(df):
    # Modified the delay to get better alpha
    return -1 * rank(((ts_sum(df['open'], 5) * ts_sum(df['gain'], 5)) - delay((ts_sum(df['open'], 5) * ts_sum(df['gain'], 5)), 5)))

def WQ009(df):
    return np.where(ts_min(delta(df['close'], 1), 5) > 0,
                                delta(df['close'], 1),
                                np.where(ts_max(delta(df['close'], 1), 5) < 0,
                                    delta(df['close'], 1),
                                    -1 * delta(df['close'], 1)
                                )
                            )

def WQ010(df):
    return rank(pd.Series(data =
                            np.where(ts_min(delta(df['close'], 1), 5) > 0,
                                delta(df['close'], 1),
                                np.where(ts_max(delta(df['close'], 1), 5) < 0,
                                    delta(df['close'], 1),
                                    -1 * delta(df['close'], 1)
                                )
                            ), index = df.index)
    )

def WQ011(df):
    # Does not work without vwap
    # ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))
    return None

def WQ012(df):
    # Oversimplistic. Quite useless
    return sign(delta(df['vol'], 1)) * (-1 * delta(df['close'], 1))

def get_WQ_factors():
    WQ_factors = {}
    # WQ_factors['WQ001'] = {'indicators': ['codenum', 'close', 'gain'], 'function': WQ001}
    # WQ_factors['WQ002'] = {'indicators': ['codenum', 'vol', 'close', 'open'], 'function': WQ002}
    WQ_factors['WQ003'] = {'indicators': ['codenum', 'vol', 'open'], 'function': WQ003}
    # WQ_factors['WQ004'] = {'indicators': ['codenum', 'low'], 'function': WQ004}
    WQ_factors['WQ006'] = {'indicators': ['codenum', 'open', 'vol'], 'function': WQ006}
    WQ_factors['WQ007'] = {'indicators': ['codenum', 'vol', 'close'], 'function': WQ007}
    WQ_factors['WQ008'] = {'indicators': ['codenum', 'open', 'gain'], 'function': WQ008}
    WQ_factors['WQ009'] = {'indicators': ['codenum', 'close'], 'function': WQ009}
    WQ_factors['WQ010'] = {'indicators': ['codenum', 'close'], 'function': WQ010}
    # WQ_factors['WQ012'] = {'indicators': ['vol', 'close'], 'function': WQ012}
    WQ_factors['WQ018'] = {'indicators': ['close', 'open'], 'function': lambda df: correlation(df['close'], df['open'], 21)}
    WQ_factors['WQ028'] = {'indicators': ['high', 'low', 'close'], 'function': lambda df: (df['high'] + df['low']) / 2 - df['close']}
    return WQ_factors

def get_all_factors():
    return {**get_basic_factors(), **get_value_factors(), **get_growth_factors(), **get_financial_quality_factors(), **get_leverage_factors(), **get_momentum_factors(), 
            **get_volatility_factors(), **get_turnover_factors(), **get_modified_momentum_factors(), **get_codenum_factor()}


# Factor calculations and tests below

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

    merged_df = pd.merge(market_df, finance_merged, how = 'inner', on = ['fd', 'codenum'])
    merged_df['gain'] = (merged_df['chg']) / 100

    # Get indices of td and codenum groups
    global td_group, codenum_group # Multable objects so information can be passed to factor functions
    all_td = merged_df['td'].unique()
    all_stocks = merged_df['codenum'].unique()
    for td in all_td:
        indices = merged_df[merged_df['td'] == td].index.tolist()
        td_group[td] = indices
    for codenum in all_stocks:
        indices = merged_df[merged_df['codenum'] == codenum].index.tolist()
        codenum_group[codenum] = indices

    # Constructing factors
    factor_cols = []
    for factor in factors:
        print(f'Calculating {factor}...')
        ind_cols = [ind for ind in factors[factor]['indicators']]
        merged_ind_df = merged_df[ind_cols]
        # Address dulplicates in ind_cols
        ind_cols = replace_duplicates_with_suffixes(ind_cols)
        merged_ind_df.columns = ind_cols

        if 'function' in factors[factor]:
            merged_df[factor] = factors[factor]['function'](merged_ind_df[ind_cols])
        else:
            if len(ind_cols) == 1:
                merged_df[factor] = merged_ind_df[ind_cols[0]]
            else:
                raise Exception(f'indicators {ind_cols} missing a combination function!')

        if len(merged_df) < 36:
            raise Exception(f'Backtest time span is too short for factor {factor}!')

        # # This will be controlled in the quadratic programming and the definition of some factors
        # # Industry neutralization
        # merged_df[f'factor_{factor}'] = merged_df[[factor, 'industry']].groupby('industry').transform(normalize)

        # # This will be controlled by the market-cap factor and the quadratic programming
        # # Market-cap neutralization
        # linregress_market_cap = LinearRegression()
        # linregress_market_cap.fit(merged_df.dropna(subset = ['market_cap', 'factor_' + factor])['market_cap'].values.reshape(-1, 1), merged_df.dropna(subset = ['market_cap', 'factor_' + factor])['factor_' + factor])
        # merged_df[f'factor_{factor}'] = merged_df.dropna(subset = ['market_cap', 'factor_' + factor])[f'factor_{factor}'] - linregress_market_cap.predict(merged_df.dropna(subset = ['market_cap', 'factor_' + factor])['market_cap'].values.reshape(-1, 1))
        # factor_cols.append(f'factor_{factor}')

        # # With neither neutralizations, only normalization:
        print(merged_df[factor])
        merged_df[f'factor_{factor}'] = normalize(merged_df[factor])
        print(merged_df[f'factor_{factor}'])
        factor_cols.append(f'factor_{factor}')

        merged_df = merged_df.copy()
    
    merged_df = merged_df[['td', 'codenum', 'gain'] + factor_cols]

    print('Null value counts:')
    print(merged_df.isnull().sum())

    # # This statement drops NA from missing factor values due to lagged factors or missing data
    # # Should not drop because some factors have a lot of missing values
    # merged_df = merged_df.dropna().reset_index(drop = True)

    merged_df = merged_df.sort_values('td') # Ensures it is sorted by td
    merged_df['gain_next'] = merged_df.groupby('codenum')['gain'].shift(-1)
    print(merged_df)

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

    #stocks_tested = random_stocks(500, start_date, end_date)

    test_factor(start_date, end_date, get_WQ_factors(), stocks = csi300_stocks())

    #test_factor(start_date, end_date, {'relative_strength_1m':get_momentum_factors()['relative_strength_1m']}, stocks = csi300_stocks())
