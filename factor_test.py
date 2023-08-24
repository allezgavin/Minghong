import numpy as np
import pandas as pd
from backtest import *
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
pd.set_option('mode.chained_assignment', None)
td_group = {}
codenum_group = {}

# Define operations
def abs(x):
    return np.abs(x)

def log(x):
    return np.log(x)

def sign(x):
    return np.sign(x)

def lag(x, d):
    output = pd.Series(dtype='float64')
    for indices in codenum_group.values():
        sub_x = x.loc[indices]
        output = pd.concat([output, sub_x.shift(d)])
    return output.reindex(x.index)

def delta(x, d):
    output = pd.Series(dtype='float64')
    for indices in codenum_group.values():
        sub_x = x.loc[indices]
        output = pd.concat([output, sub_x - sub_x.shift(d)])
    return output.reindex(x.index)

def delta_pct(x, d):
    output = pd.Series(dtype='float64')
    for indices in codenum_group.values():
        sub_x = x.loc[indices]
        output = pd.concat([output, (sub_x - sub_x.shift(d)) / sub_x.shift(d) - 1])
    return output.reindex(x.index)

def ts_sum(x, d):
    output = pd.Series(dtype='float64')
    for indices in codenum_group.values():
        sub_x = x.loc[indices]
        output = pd.concat([output, sub_x.rolling(window = d, min_periods = d).sum()])
    return output.reindex(x.index)

def ts_rank(x, d):
    output = pd.Series(dtype='float64')
    for indices in codenum_group.values():
        sub_x = x.loc[indices]
        output = pd.concat([output, sub_x.rolling(window = d, min_periods = d).apply(lambda x: (x.argsort().argsort().iloc[-1] + 1), raw=False)])
    return output.reindex(x.index)

def ts_min(x, d):
    output = pd.Series(dtype='float64')
    for indices in codenum_group.values():
        sub_x = x.loc[indices]
        
        window_min = sub_x.iloc[:d].min()
        sub_output = [np.nan for i in range(d - 1)] + [window_min]
        for i in range(d, len(indices)):
            old = sub_x.iloc[i - d]
            new = sub_x.iloc[i]
            if old == window_min:
                window_min = sub_x.iloc[i-d+1:i+1].min()
            elif new < window_min:
                window_min = new
            sub_output.append(window_min)
        sub_output = pd.Series(sub_output, index=sub_x.index)

        output = pd.concat([output, sub_output])
    return output.reindex(x.index)

def ts_max(x, d):
    output = pd.Series(dtype='float64')
    for indices in codenum_group.values():
        sub_x = x.loc[indices]
        
        window_max = sub_x.iloc[:d].max()
        sub_output = [np.nan for i in range(d - 1)] + [window_max]
        for i in range(d, len(indices)):
            old = sub_x.iloc[i - d]
            new = sub_x.iloc[i]
            if old == window_max:
                window_max = sub_x.iloc[i-d+1:i+1].max()
            elif new < window_max:
                window_max = new
            sub_output.append(window_max)
        sub_output = pd.Series(sub_output, index=sub_x.index)

        output = pd.concat([output, sub_output])
    return output.reindex(x.index)

def ts_std(x, d):
    output = pd.Series(dtype='float64')
    for indices in codenum_group.values():
        sub_x = x.loc[indices]
        output = pd.concat([output, sub_x.rolling(window = d, min_periods = d).std()])
    return output.reindex(x.index)

def ts_argmax(x, d):
    output = pd.Series(dtype='float64')
    for indices in codenum_group.values():
        sub_x = x.loc[indices]
        output = pd.concat([output, sub_x.rolling(window = d, min_periods = d).apply(lambda x: np.argmax(x) + 1, raw=False)])
    return output.reindex(x.index)

def ts_argmin(x, d):
    output = pd.Series(dtype='float64')
    for indices in codenum_group.values():
        sub_x = x.loc[indices]
        output = pd.concat([output, sub_x.rolling(window = d, min_periods = d).apply(lambda x: np.argmin(x) + 1, raw=False)])
    return output.reindex(x.index)

def rank(x):
    output = pd.Series(dtype='float64')
    for indices in td_group.values():
        sub_x = x.loc[indices]
        output = pd.concat([output, sub_x.rank()])
    return output.reindex(x.index)

def delay(x, d):
    output = pd.Series(dtype='float64')
    for indices in codenum_group.values():
        sub_x = x.loc[indices]
        output = pd.concat([output, sub_x.shift(d)])
    return output.reindex(x.index)

def correlation(x, y, d):
    output = pd.Series(dtype='float64')
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
    output = pd.Series(dtype='float64')
    for indices in codenum_group.values():
        sub_x = x.loc[indices]
        sub_y = y.loc[indices]
        output = pd.concat([output, sub_x.rolling(window = d, min_periods = d).cov(sub_y)])
    return output.reindex(x.index)

def scale(x, a=1):
    output = pd.Series(dtype='float64')
    for indices in td_group.values():
        sub_x = x.loc[indices]
        output = pd.concat([output, sub_x * (a / np.sum(np.abs(x)))])
    return output.reindex(x.index)

def signedpower(x, a):
    return np.power(x, a)

def decay_linear(x, d):
    weights = np.arange(d, 0, -1)
    output = pd.Series(dtype='float64')
    for indices in codenum_group.values():
        sub_x = x.loc[indices]
        output = pd.concat([output, np.convolve(sub_x, weights / weights.sum(), mode='valid')])
    return output.reindex(x.index)

# Must have no exposure. Controlled by quadratic programming
def get_size_factor():
    size_factor = {}
    size_factor['ln_market_cap'] = {'indicators': ['market_cap'], 'function': lambda x: np.log(x.iloc[:, 0])}
    return size_factor

def get_value_factors():
    value_factors = {}
    value_factors['netprofitrate'] = {'indicators': ['netprofitrate']}
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
    financial_quality_factors = {}
    financial_quality_factors['ROE'] = {'indicators': ['net_profit', 'total_shareholders_equity'], 'function': lambda df: df['net_profit'] / df['total_shareholders_equity']}
    financial_quality_factors['ROA'] = {'indicators': ['net_profit', 'total_assets'], 'function': lambda df: df['net_profit'] / df['total_assets']}
    financial_quality_factors['grossprofitmargin'] = {'indicators': ['grossmargin']}
    # financial_quality_factors['profitmargin'] = {'indicators': ['operating_profit', 'operating_revenue'], 'function': lambda df: df['operating_profit'] / df['operating_revenue']}
    financial_quality_factors['assetturnover'] = {'indicators': ['operating_revenue', 'total_assets'], 'function': lambda df: df['operating_revenue'] / df['total_assets']}
    financial_quality_factors['operationcashflowratio'] = {'indicators': ['net_operating_cashflow', 'net_profit'], 'function': lambda df: df['net_operating_cashflow'] / df['net_profit']}

    return financial_quality_factors

def get_leverage_factors():
    leverage_factors = {}
    leverage_factors['market_value_leverage'] = {'indicators': ['market_cap', 'total_noncurrent_liabilities'], #also preferred stocks should be included
                                                'function': lambda df: (df['market_cap'] + df['total_noncurrent_liabilities']) / df['market_cap']}
    leverage_factors['financial_leverage'] = {'indicators': ['total_assets', 'total_shareholders_equity'], 'function': lambda df: df['total_assets'] / df['total_shareholders_equity']}
    leverage_factors['debtequityratio'] = {'indicators': ['total_noncurrent_liabilities', 'total_shareholders_equity'], 'function': lambda df: df['total_noncurrent_liabilities'] / df['total_shareholders_equity']}
    leverage_factors['cashratio'] = {'indicators': ['cash', 'account_receivable', 'total_current_liabilities'],
                                    'function': lambda df: (df['cash'] + df['account_receivable']) / df['total_current_liabilities']}
    leverage_factors['currentratio'] = {'indicators': ['total_current_assets', 'total_current_liabilities'], 'function': lambda df: df['total_current_assets'] / df['total_current_liabilities']}

    return leverage_factors

def HAlpha(df):
    months = 12
    # Alpha of every stock in the past 12 months.
    start_date = df['td'].min()
    df['td'] = df['td'].astype('str')
    bench_df = query_SQL_indexprice()[['td', 'gain']]
    bench_df.columns = ['td', 'gain_benchmark']
    df = pd.merge(df, bench_df, how = 'left', on = 'td')
    output = pd.Series(dtype='float64')
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
    momentum_factors['relative_strength_2m'] = {'indicators': ['close'], 'function': lambda df: delta(df['close'], 2 * 21)}
    momentum_factors['relative_strength_3m'] = {'indicators': ['close'], 'function': lambda df: delta(df['close'], 3 * 21)}
    momentum_factors['relative_strength_6m'] = {'indicators': ['close'], 'function': lambda df: delta(df['close'], 6 * 21)}
    momentum_factors['relative_strength_12m'] = {'indicators': ['close'], 'function': lambda df: delta(df['close'], 12 * 21)}
    momentum_factors['weighted_strength_1m'] = {'indicators': ['gain', 'vol'], 'function': lambda df: ts_sum(df['gain'] * df['vol'], 1 * 21)}
    momentum_factors['weighted_strength_2m'] = {'indicators': ['gain', 'vol'], 'function': lambda df: ts_sum(df['gain'] * df['vol'], 2 * 21)}
    momentum_factors['weighted_strength_3m'] = {'indicators': ['gain', 'vol'], 'function': lambda df: ts_sum(df['gain'] * df['vol'], 3 * 21)}
    momentum_factors['weighted_strength_6m'] = {'indicators': ['gain', 'vol'], 'function': lambda df: ts_sum(df['gain'] * df['vol'], 6 * 21)}
    momentum_factors['weighted_strength_12m'] = {'indicators': ['gain', 'vol'], 'function': lambda df: ts_sum(df['gain'] * df['vol'], 12 * 21)}

    return momentum_factors

def beta_consistency(df):
    months = 6
    # beta times summed square of regression residuals.
    start_date = df['td'].min()
    df['td'] = df['td'].astype('str')
    bench_df = query_SQL_indexprice()[['td', 'gain']]
    bench_df.columns = ['td', 'gain_benchmark']
    df = pd.merge(df, bench_df, how = 'left', on = 'td')
    output = pd.Series(dtype='float64')
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
    volatility_factors['high_low_1m'] = {'indicators': ['high', 'low'], 'function': lambda df: ts_max(df['high'], 1 * 21) / ts_min(df['low'], 1 * 21)}
    volatility_factors['high_low_2m'] = {'indicators': ['high', 'low'], 'function': lambda df: ts_max(df['high'], 2 * 21) / ts_min(df['low'], 2 * 21)}
    volatility_factors['high_low_3m'] = {'indicators': ['high', 'low'], 'function': lambda df: ts_max(df['high'], 3 * 21) / ts_min(df['low'], 3 * 21)}
    volatility_factors['high_low_6m'] = {'indicators': ['high', 'low'], 'function': lambda df: ts_max(df['high'], 6 * 21) / ts_min(df['low'], 6 * 21)}
    volatility_factors['high_low_12m'] = {'indicators': ['high', 'low'], 'function': lambda df: ts_max(df['high'], 12 * 21) / ts_min(df['low'], 12 * 21)}
    volatility_factors['std_1m'] = {'indicators': ['high'], 'function': lambda df: ts_std(df['high'], 1 * 21)}
    volatility_factors['std_2m'] = {'indicators': ['high'], 'function': lambda df: ts_std(df['high'], 2 * 21)}
    volatility_factors['std_3m'] = {'indicators': ['high'], 'function': lambda df: ts_std(df['high'], 3 * 21)}
    volatility_factors['std_6m'] = {'indicators': ['high'], 'function': lambda df: ts_std(df['high'], 6 * 21)}
    volatility_factors['std_12m'] = {'indicators': ['high'], 'function': lambda df: ts_std(df['high'], 12 * 21)}
    volatility_factors['ln_price'] = {'indicators': ['close'], 'function': lambda df: np.log(df['close'])}
    # Slow due to loops
    # volatility_factors['beta_consistency'] = {'indicators': ['td', 'codenum', 'gain'], 'function': beta_consistency}

    return volatility_factors

def get_turnover_factors():
    turnover_factors = {}
    turnover_factors['turnover_1m'] = {'indicators': ['vol' ,'float_share'], 'function': lambda df: ts_sum(df['vol'], 1 * 21) / df['float_share']}
    turnover_factors['turnover_2m'] = {'indicators': ['vol' ,'float_share'], 'function': lambda df: ts_sum(df['vol'], 2 * 21) / df['float_share']}
    turnover_factors['turnover_3m'] = {'indicators': ['vol' ,'float_share'], 'function': lambda df: ts_sum(df['vol'], 3 * 21) / df['float_share']}
    turnover_factors['turnover_6m'] = {'indicators': ['vol' ,'float_share'], 'function': lambda df: ts_sum(df['vol'], 6 * 21) / df['float_share']}
    turnover_factors['turnover_12m'] = {'indicators': ['vol' ,'float_share'], 'function': lambda df: ts_sum(df['vol'], 12 * 21) / df['float_share']}

    return turnover_factors

def codenum_adjacency(df):
    output = pd.Series(dtype='float64')
    for td, sub_df in df.sort_values('codenum').groupby('td'):
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
    volume_price_factors['FR'] = {'indicators': ['vol', 'float_share', 'gain'], 'function': lambda df: correlation(df['vol'] / df['float_share'], df['gain'], 21)}
    volume_price_factors['WQ003'] = {'indicators': ['codenum', 'vol', 'open'], 'function': lambda df: (-1 * correlation(rank(df['open']), rank(df['vol']), 10))}
    volume_price_factors['WQ006'] = {'indicators': ['codenum', 'open', 'vol'], 'function': lambda df: -1 * correlation(df['open'], df['vol'], 10)}
    # volume_price_factors['WQ012'] = {'indicators': ['vol', 'close'], 'function': lambda df: sign(delta(df['vol'], 1)) * (-1 * delta(df['close'], 1))}
    volume_price_factors['vp_chg_div'] = {'indicators': ['vol', 'close'], 'function': lambda df: correlation(delta(df['vol'], 1), delta(df['close'], 1), 21)}
    # volume_price_factors['ranked_vp_chg_div'] = {'indicators': ['vol', 'close'], 'function': ranked_chg_div}
    # volume_price_factors['ts_ranked_vp_chg_div'] = {'indicators': ['vol', 'close'], 'function': ts_ranked_chg_div}
    return volume_price_factors

def WQ001(df):
    return rank(ts_argmax(signedpower(pd.Series(data =
                            np.where(
                                df['gain'] < 0,
                                ts_std(df['gain'], 20),
                                df['close']
                            ), index = df.index)
                            , 2), 5)) - 0.5


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

def get_WQ_factors():
    WQ_factors = {}
    # WQ_factors['WQ001'] = {'indicators': ['codenum', 'close', 'gain'], 'function': WQ001}
    # WQ_factors['WQ002'] = {'indicators': ['codenum', 'vol', 'close', 'open'], 'function': lambda df: -1 * correlation(rank(delta(log(df['vol']), 2)), rank(((df['close'] - df['open']) / df['open'])), 6)}
    # WQ_factors['WQ004'] = {'indicators': ['codenum', 'low'], 'function': lambda df: -1 * ts_rank(rank(df['low']), 9)}
    WQ_factors['WQ007'] = {'indicators': ['codenum', 'vol', 'close'], 'function': WQ007}
    WQ_factors['WQ008'] = {'indicators': ['codenum', 'open', 'gain'], 'function': WQ008}
    WQ_factors['WQ009'] = {'indicators': ['codenum', 'close'], 'function': WQ009}
    WQ_factors['WQ010'] = {'indicators': ['codenum', 'close'], 'function': WQ010}
    WQ_factors['WQ018'] = {'indicators': ['close', 'open'], 'function': lambda df: correlation(df['close'], df['open'], 21)}
    WQ_factors['WQ028'] = {'indicators': ['high', 'low', 'close'], 'function': lambda df: (df['high'] + df['low']) / 2 - df['close']}
    return WQ_factors

def sell_tend(df):
    N = 10
    d = 0.23
    df['turnover'] = df['vol'] / df['float_share']
    gain_series = []
    loss_series = []
    omega_series = []
    for n in range(1, N):
        diff = delta_pct(df['close'], n)
        gain = diff.copy()
        gain[gain < 0] = 0
        loss = diff.copy()
        loss[loss > 0] = 0
        omega = np.multiply(lag(df['turnover'], n), np.prod([1 - lag(df['turnover'], i) for i in range(n)], axis = 0))
        gain_series.append(gain)
        loss_series.append(loss)
        omega_series.append(omega)
    gain_series = pd.DataFrame(gain_series)
    loss_series = pd.DataFrame(loss_series)
    omega_series = pd.DataFrame(omega_series)
    omega_series = omega_series / omega_series.sum(axis = 0)
    x =  (omega_series * gain_series - omega_series * loss_series * d).sum(axis = 0)
    return x

def TK_weight(weekly):
    y = 0.61
    d = 0.69
    current = weekly.iloc[-1]

    if current > 0:
        P_g = (weekly > current).sum()
        P_ge = P_g + (weekly == current).sum()
        P_g = P_g / len(weekly)
        P_ge = P_ge / len(weekly)
        g = P_g ** y / (P_g ** y + (1-P_g) ** y) ** (1/y)
        ge = P_ge ** y / (P_ge ** y + (1-P_ge) ** y) ** (1/y)
        pi = ge - g

    else:
        P_s = (weekly < current).sum()
        P_se = P_s + (weekly == current).sum()
        P_s = P_s / len(weekly)
        P_se = P_se / len(weekly)
        s = P_s ** d / (P_s ** d + (1-P_s) ** d) ** (1/d)
        se = P_se ** d / (P_se ** d + (1-P_se) ** d) ** (1/d)
        pi = se - s

    return pi

def TK(df):
    a = 0.88
    l = 2.25
    N_weeks = 60
    TK = pd.Series(dtype='float64')
    for stock, sub_df in df.groupby('codenum'):
        sub_df['weekly'] = sub_df['gain'].rolling(window = 5, min_periods = 5).sum()
        sub_df['util'] = sub_df['weekly'].copy()
        mask = (sub_df['util'] < 0)
        sub_df['util'] = sub_df['util'].abs() ** a
        sub_df['util'][mask] = -l * sub_df['util'][mask]
        sub_df['weight'] = sub_df['weekly'].rolling(window = N_weeks, min_periods = N_weeks).apply(TK_weight)
        sub_df['util'].fillna(0, inplace=True)
        sub_df['weight'].fillna(0, inplace=True)
        sub_df['TK'] = sub_df['util'].rolling(window = N_weeks, min_periods = N_weeks).apply(lambda x: np.dot(x, sub_df['weight'].iloc[-N_weeks:]), raw = True)
        TK = pd.concat([TK, sub_df['TK']])
    return TK.reindex(df.index)

def get_bahavioral_factors():
    behavioral_factors = {}
    behavioral_factors['sell_tend'] = {'indicators': ['close', 'vol', 'float_share'], 'function': sell_tend}
    # Useless
    # behavioral_factors['TK'] = {'indicators': ['codenum', 'gain'], 'function': TK}
    return behavioral_factors

def get_style_factors():
    return {**get_value_factors(), **get_growth_factors(), **get_financial_quality_factors(), **get_leverage_factors(), **get_momentum_factors(), 
            **get_volatility_factors(), **get_turnover_factors()}

def get_alpha_factors():
    return {**get_codenum_factor(), **get_volume_price_factors(), **get_WQ_factors(), **get_bahavioral_factors()}

def get_factor_group_list():
    return [get_style_factors(), get_alpha_factors(), get_size_factor()]

def get_all_factors():
    factor_dict = {}
    for factor_group in get_factor_group_list():
        factor_dict = {**factor_dict, **factor_group}
    return factor_dict

# Factor calculations and tests
def last_day_of_last_quarter(current_date):
    quarters_finished = (current_date.month - 1) // 3
    return datetime.date(current_date.year, quarters_finished * 3 + 1, 1) + datetime.timedelta(days=-(1))

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

def calc_factors(factors = {}):

    if len(factors) == 0:
        to_csv = True
        factors = get_all_factors()
    else:
        to_csv = False

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
    market_ind.add('chg') # Prepares for calculating everyday gain
    market_ind.add('market_cap') # Prepares for neutralization. 'market_cap' is handled by read_SQL_market()
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

    print('Querying finance table...')
    finance_df = query_SQL_finance(factors = fin_ind)
    print('Querying finance_deriv table...')
    finance_deriv_df = query_SQL_finance_deriv(factors = fin_deriv_ind)
    print('Querying market table...')
    market_df = query_SQL_market(indicators = market_ind)
    print('Querying company table...')
    company_df = query_SQL_company()
    print('Queries finished!')

    finance_merged = pd.merge(finance_df, finance_deriv_df, how = 'inner', on = ['fd', 'disclosure', 'codenum'])

    finance_merged['disclosure'] = finance_merged['disclosure'].astype(int)
    merged_df = pd.merge_asof(market_df, finance_merged, left_on = 'td', right_on = 'disclosure', by = 'codenum', direction = 'backward')
    merged_df = pd.merge(merged_df, company_df, how = 'inner', on = 'codenum')
    merged_df['gain'] = (merged_df['chg']) / 100
    merged_df['ln_market_cap'] = np.log(merged_df['market_cap'])

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

        if factor in get_style_factors().keys():
            #industry neutralization
            merged_df[f'factor_{factor}'] = merged_df[[factor, 'industry']].groupby('industry').transform(normalize)

            #market-cap neutralization
            linregress_market_cap = LinearRegression()
            linregress_market_cap.fit(merged_df.dropna(subset = ['ln_market_cap', 'factor_' + factor])['ln_market_cap'].values.reshape(-1, 1), merged_df.dropna(subset = ['ln_market_cap', 'factor_' + factor])['factor_' + factor])
            merged_df[f'factor_{factor}'] = merged_df.dropna(subset = ['ln_market_cap', 'factor_' + factor])[f'factor_{factor}'] - linregress_market_cap.predict(merged_df.dropna(subset = ['ln_market_cap', 'factor_' + factor])['ln_market_cap'].values.reshape(-1, 1))
            factor_cols.append(f'factor_{factor}')

        else:
            # With neither neutralizations, only normalization:
            merged_df[f'factor_{factor}'] = normalize(merged_df[factor])
            factor_cols.append(f'factor_{factor}')

        merged_df = merged_df.copy()
    
    merged_df = merged_df[['td', 'codenum', 'gain'] + factor_cols]

    print('Null value counts:')
    print(merged_df.isnull().sum())

    merged_df = merged_df.sort_values('td') # Ensures it is sorted by td
    merged_df['gain_next'] = merged_df.groupby('codenum')['gain'].shift(-1)

    if to_csv:
        merged_df.to_csv('factors.csv', index = False)
        print('All factors calculated!')

    return merged_df

def IC_test(factor_key = '', period = period, df = 'factors.csv'):
    if type(df) == str:
        factor_pool = pd.read_csv(df)
    else:
        factor_pool = df
    if factor_key != '':
        factor_pool = factor_pool[['td', 'codenum', 'gain_next', 'factor_' + factor_key]]
    factor_cols = factor_pool.filter(like = 'factor_').columns.to_list()
    sub_dfs = []
    for stock, sub_df in factor_pool.groupby('codenum'):
        sub_df['gain_next_mean'] = sub_df['gain_next'].rolling(window = period).mean().shift(-period+1)
        sub_dfs.append(sub_df)
    factor_pool = pd.concat(sub_dfs)
    IC_series = []
    td_index = []
    for td, sub_df in factor_pool.groupby('td'):
        IC = sub_df[factor_cols].corrwith(sub_df['gain_next_mean'])
        IC_series.append(IC)
        td_index.append(td)
    IC_series = pd.DataFrame(data = IC_series, index = td_index)
    IC_series.index.name = 'td'
    IC_series = IC_series.reset_index()
    recent_IC_series = IC_series
    #recent_IC_series = IC_series.iloc[-260:]

    IR = recent_IC_series[factor_cols].mean() / recent_IC_series[factor_cols].std()

    if len(factor_cols) == 1:
        IC_factor = factor_cols[0]
        IC_series['year_month'] = pd.to_datetime(IC_series['td'], format = '%Y%m%d').dt.strftime('%Y%m')
        monthly_mean = IC_series.groupby('year_month')[IC_factor].mean().reset_index()
        plt.figure(figsize = (10, 5))
        plt.xticks(rotation=45)
        sns.barplot(x = 'year_month', y = IC_factor, data = monthly_mean)
        plt.savefig('IC_test.png')
        plt.show()
    return IC_series, IR

def IR_test(factor_key = '', periods = [1,2,3,5,10,20,40], df = 'factors.csv'):
    IR_df = pd.DataFrame()
    for elem in periods:
        print('Calculating data with period length', period)
        IR = IC_test(factor_key=factor_key, period=elem, df = df)[1]
        IR_df = pd.concat([IR_df, IR], axis = 1)
    return IR_df

def group_backtest(factor_key, period = 1, divide_groups = 5, df = 'factors.csv'):
    factor_name = 'factor_' + factor_key
    if type(df) == str:
        factor_pool = pd.read_csv(df)
    else:
        factor_pool = df
    factor_pool = factor_pool[['td', 'codenum', 'gain_next'] + [factor_name]]
    sub_dfs = []
    for stock, sub_df in factor_pool.groupby('codenum'):
        sub_df['gain_next_mean'] = sub_df['gain_next'].rolling(window = period).mean().shift(-period + 1)
        sub_df = sub_df.iloc[-1::-period].iloc[::-1]
        sub_dfs.append(sub_df)
    factor_pool = pd.concat(sub_dfs)
    sub_dfs = []
    for td, sub_df in factor_pool.groupby('td'):
        sub_df = sub_df.sort_values(factor_name)
        group_size = len(sub_df) // divide_groups
        sub_df['group'] = 1
        for i in range(1, divide_groups):
            sub_df['group'].iloc[:(divide_groups - i) * group_size] = i + 1
        sub_dfs.append(sub_df)
    group_merged_df = pd.concat(sub_dfs)

    result = pd.DataFrame()
    annual_profits = []
    for group, sub_df in group_merged_df.groupby('group'):
        avg_profit = sub_df.groupby('td')['gain_next_mean'].mean()
        group_result = pd.DataFrame()
        group_result['date'] = pd.to_datetime(avg_profit.index, format = '%Y%m%d')
        group_result['avg_profit'] = list(avg_profit)
        cumulative = (1 + group_result['avg_profit']).cumprod() / (1 + group_result['avg_profit'].iloc[0])
        group_result['cumulative_profit'] = cumulative
        group_result['group'] = group
        result = pd.concat([result, group_result])
        annual_profits.append(cumulative.dropna().iloc[-1] ** (260 / len(cumulative) / period) - 1)
    
    result.reset_index(drop = True, inplace= True)

    print(f'Annual profits of Groups starting from Group 1 (greatest factor value) are {annual_profits}')
    plt.figure(figsize = (10, 5))
    sns.lineplot(x = 'date', y = 'cumulative_profit', data = result, hue = 'group')
    plt.savefig('grouped_backtest.png')
    plt.show()

def test_factor(factor_key, period = period):
    df = calc_factors(factors = {factor_key: get_all_factors()[factor_key]})
    IC_test(factor_key = factor_key, period = period, df = df)
    group_backtest(factor_key = factor_key, period = period, df = df)

if __name__ == '__main__':

    test_factor('high_low_6m', period = 1)
    # IC_test()