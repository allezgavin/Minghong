import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from backtest import *

def subtract(df):
    return df.iloc[:, 0] / df.iloc[:, 1]

def divide(df):
    return df.iloc[:, 0] / df.iloc[:, 1]

def growth(df):
    return (df.iloc[:, 0] - df.iloc[:, 1]) / df.iloc[:, 1]

def HAlpha60(df):
    #df.columns is ['fd', 'codenum']
    df_short = df.drop_duplicates().reset_index(drop = True)
    df_short['end_date'] = pd.to_datetime(df_short['fd'], format = '%Y%m%d')
    df_short['start_date'] = df_short['end_date'] - np.timedelta64(60, 'M') #60 months
    alpha_list = []
    for i in range(len(df_short)):
        print(df_short['start_date'])
        portfolio = pd.DataFrame()
        portfolio['td'] = pd.date_range(start = df_short['start_date'][i], end = df_short['end_date'][i], freq = 'D').strftime('%Y%m%d')
        portfolio['codenum'] = df_short['codenum'][i]
        portfolio['weight'] = 1
        alpha = backtest(df_short['start_date'][i].strftime(format = '%Y%m%d'), portfolio, end_date = df_short['end_date'][i].strftime(format = '%Y%m%d')).alpha
        alpha_list.append(alpha)
    df_short['alpha'] = alpha_list
    df = pd.merge(df, df_short, how = 'left', on = ['fd', 'codenum'])
    return df['alpha']

def moving_average(series, window_size = 5):
    # Define a kernel for 1D convolution to calculate the moving average
    kernel = np.ones(window_size) / (window_size)

    # Calculate the padding needed for the input series
    pad_width = (window_size - 1) // 2
    padded_series = np.pad(series, (pad_width, pad_width), mode='edge')

    # Use the numpy convolution function to calculate the moving average
    return np.convolve(padded_series, kernel, mode='valid')

def weighted_gain_codenum(df):
    original_order = df[['td', 'codenum']]
    df['moving_avg_gain'] = df.sort_values('codenum').groupby('td')['gain'].transform(moving_average)
    df = pd.merge(original_order, df, on = ['td', 'codenum'])
    return df['moving_avg_gain']


def get_basic_factors():
    basic_factors = {}
    basic_factors['PE'] = {'indicators': ['close', 'EPS'], 'function': divide}
    basic_factors['netprofitrate'] = {'indicators': ['netprofitrate']}
    return basic_factors

# # Does not apply under market cap neutralization
# def get_size_factor():
#     size_factor = {}
#     size_factor['ln_capital'] = {'indicators': ['market_cap'], 'function': lambda x: np.log(x.iloc[:, 0])}
#     return size_factor

def get_value_factors():
    value_factors = {}
    value_factors['EP'] = {'indicators': ['net_profit', 'market_cap'], 'function': divide}
    value_factors['EPCut'] = {'indicators': ['deducted_profit', 'market_cap'], 'function': divide}
    value_factors['BP'] = {'indicators': ['total_assets', 'market_cap'], 'function': divide}
    value_factors['SP'] = {'indicators': ['operating_revenue', 'market_cap'], 'function': divide}
    value_factors['OCFP'] = {'indicators': ['OCFPS']}
    value_factors['NCFP'] = {'indicators': ['net_operating_cashflow', 'net_invest_cashflow', 'net_finance_cashflow', 'market_cap'],
                            'function': lambda x : x.iloc[:, 0] + x.iloc[:, 1] + x.iloc[:, 2] / x.iloc[:, 3]}
    # value_factors['DP'] = {'indicators': ['dividends', 'market_cap'], 'function': divide}
    # value_factors['FCFP'] = {'indicators': ['free_cashflow', 'market_cap'], 'function': divide}
    return value_factors

def get_growth_factors():
    growth_factors = {}
    growth_factors['sales_growth_ttm'] = {'indicators': ['operating_revenue', 'operating_revenue'], 'lag': [0, 260], 'function': growth}
    growth_factors['profit_growth_ttm'] = {'indicators': ['deducted_profit', 'deducted_profit'], 'lag': [0, 260], 'function': growth}
    growth_factors['operationcashflow_growth_ttm'] = {'indicators': ['net_operating_cashflow', 'net_operating_cashflow'], 'lag': [0, 260], 'function': growth}
    return growth_factors

def get_financial_quality_factors():
    financial_quality_factors = {} #TTM or of that quarter?
    financial_quality_factors['ROE'] = {'indicators': ['net_profit', 'total_shareholders_equity'], 'function': divide}
    financial_quality_factors['ROA'] = {'indicators': ['net_profit', 'total_assets'], 'function': divide}
    financial_quality_factors['grossprofitmargin'] = {'indicators': ['grossmargin']}
    financial_quality_factors['profitmargin'] = {'indicators': ['operating_profit', 'operating_revenue'], 'function': divide}
    financial_quality_factors['assetturnover'] = {'indicators': ['operating_revenue', 'total_assets'], 'function': divide}
    financial_quality_factors['operationcashflowratio'] = {'indicators': ['net_operating_cashflow', 'net_profit'], 'function': divide}

    return financial_quality_factors

def get_leverage_factors():
    leverage_factors = {}
    leverage_factors['market_value_leverage'] = {'indicators': ['market_cap', 'total_noncurrent_liabilities'], #also preferred stocks should be included
                                                'function': lambda x: (x.iloc[:, 0] + x.iloc[:, 1]) / x.iloc[:, 0]}
    leverage_factors['financial_leverage'] = {'indicators': ['total_assets', 'total_shareholders_equity'], 'function': divide}
    leverage_factors['debtequityratio'] = {'indicators': ['total_noncurrent_liabilities', 'total_shareholders_equity'], 'function': divide}
    leverage_factors['cashratio'] = {'indicators': ['cash', 'account_receivable', 'total_current_liabilities'],
                                    'function': lambda x: (x.iloc[:, 0] + x.iloc[:, 1]) / x.iloc[:, 2]}
    leverage_factors['currentratio'] = {'indicators': ['total_current_assets', 'total_current_liabilities'], 'function': divide}

    return leverage_factors

# def get_size_factor():
#     return size_factor

def get_momentum_factors():
    momentum_factors = {}
    # # Extremely slow!
    # momentum_factors['HALpha'] = {'indicators': ['fd', 'codenum'], 'function': HAlpha60}
    momentum_factors['relative_strength_1m'] = {'indicators': ['close', 'close'], 'lag': [0, 21], 'function': growth}
    # It is faster to lag a fixed number of rows. Fow more accuracy, use 'lag_unit'.
    momentum_factors['relative_strength_2m'] = {'indicators': ['close', 'close'], 'lag': [0, 2 * 21], 'function': growth}
    momentum_factors['relative_strength_3m'] = {'indicators': ['close', 'close'], 'lag': [0, 3 * 21], 'function': growth}
    momentum_factors['relative_strength_6m'] = {'indicators': ['close', 'close'], 'lag': [0, 6 * 21], 'function': growth}
    momentum_factors['relative_strength_12m'] = {'indicators': ['close', 'close'], 'lag': [0, 12 * 21], 'function': growth}
    return momentum_factors

def get_volatility_factors():
    volatility_factors = {}
    volatility_factors['high_low_1m'] = {'indicators': (['high' for i in range(21)] + ['low' for i in range(21)]), 'lag': list(range(21)) + list(range(21)),
                                         'function': lambda df: df.iloc[:, :21].max(axis = 1) / df.iloc[:, 21:].min(axis = 1)}
    volatility_factors['high_low_2m'] = {'indicators': (['high' for i in range(42)] + ['low' for i in range(42)]), 'lag': list(range(42)) + list(range(42)),
                                         'function': lambda df: df.iloc[:, :42].max(axis = 1) / df.iloc[:, 42:].min(axis = 1)}
    volatility_factors['high_low_3m'] = {'indicators': (['high' for i in range(63)] + ['low' for i in range(63)]), 'lag': list(range(63)) + list(range(63)),
                                         'function': lambda df: df.iloc[:, :63].max(axis = 1) / df.iloc[:, 63:].min(axis = 1)}
    volatility_factors['high_low_6m'] = {'indicators': (['high' for i in range(126)] + ['low' for i in range(126)]), 'lag': list(range(126)) + list(range(126)),
                                         'function': lambda df: df.iloc[:, :126].max(axis = 1) / df.iloc[:, 126:].min(axis = 1)}
    # This method is quite slow
    volatility_factors['high_low_12m'] = {'indicators': (['high' for i in range(260)] + ['low' for i in range(260)]), 'lag': list(range(260)) + list(range(260)),
                                         'function': lambda df: df.iloc[:, :260].max(axis = 1) / df.iloc[:, 260:].min(axis = 1)}
    volatility_factors['std_1m'] = {'indicators': ['high' for i in range(21)], 'lag': list(range(21)),
                                    'function': lambda df: df.std(axis = 1)}
    volatility_factors['std_2m'] = {'indicators': ['high' for i in range(42)], 'lag': list(range(42)),
                                    'function': lambda df: df.std(axis = 1)}
    volatility_factors['std_3m'] = {'indicators': ['high' for i in range(63)], 'lag': list(range(63)),
                                    'function': lambda df: df.std(axis = 1)}
    volatility_factors['std_6m'] = {'indicators': ['high' for i in range(126)], 'lag': list(range(126)),
                                    'function': lambda df: df.std(axis = 1)}
    volatility_factors['std_12m'] = {'indicators': ['high' for i in range(260)], 'lag': list(range(260)),
                                    'function': lambda df: df.std(axis = 1)}
    volatility_factors['ln_price'] = {'indicators': ['close'], 'function': lambda df: np.log(df.iloc[:, 0])}
    #volatility_factors['beta_consistence']
    return volatility_factors

def get_turnover_factors():
    turnover_factors = {}
    turnover_factors['turnover_1m'] = {'indicators': ['vol' for i in range(21)] + ['total_share'], 'lag': list(range(21)) + [0],
                                       'function': lambda df: df.iloc[:, :-1].sum(axis = 1) / df.iloc[:, -1]}
    turnover_factors['turnover_2m'] = {'indicators': ['vol' for i in range(42)] + ['total_share'], 'lag': list(range(42)) + [0],
                                       'function': lambda df: df.iloc[:, :-1].sum(axis = 1) / df.iloc[:, -1]}
    turnover_factors['turnover_3m'] = {'indicators': ['vol' for i in range(63)] + ['total_share'], 'lag': list(range(63)) + [0],
                                       'function': lambda df: df.iloc[:, :-1].sum(axis = 1) / df.iloc[:, -1]}
    turnover_factors['turnover_6m'] = {'indicators': ['vol' for i in range(126)] + ['total_share'], 'lag': list(range(126)) + [0],
                                       'function': lambda df: df.iloc[:, :-1].sum(axis = 1) / df.iloc[:, -1]}
    turnover_factors['turnover_12m'] = {'indicators': ['vol' for i in range(260)] + ['total_share'], 'lag': list(range(260)) + [0],
                                       'function': lambda df: df.iloc[:, :-1].sum(axis = 1) / df.iloc[:, -1]}
    return turnover_factors

def get_modified_momentum_factors():
    modified_momentum_factors = {}
    modified_momentum_factors['weighted_strength_1m'] = {'indicators': (['gain' for i in range(21)] + ['vol' for i in range(21)]), 'lag': list(range(21)) + list(range(21)),
                                                         'function': lambda df: pd.DataFrame([df.iloc[:, i] * df.iloc[:, i + 21] for i in range(21)]).mean(axis = 0)}
    modified_momentum_factors['weighted_strength_2m'] = {'indicators': (['gain' for i in range(42)] + ['vol' for i in range(42)]), 'lag': list(range(42)) + list(range(42)),
                                                         'function': lambda df: pd.DataFrame([df.iloc[:, i] * df.iloc[:, i + 42] for i in range(42)]).mean(axis = 0)}
    modified_momentum_factors['weighted_strength_3m'] = {'indicators': (['gain' for i in range(63)] + ['vol' for i in range(63)]), 'lag': list(range(63)) + list(range(63)),
                                                         'function': lambda df: pd.DataFrame([df.iloc[:, i] * df.iloc[:, i + 63] for i in range(63)]).mean(axis = 0)}
    modified_momentum_factors['weighted_strength_6m'] = {'indicators': (['gain' for i in range(126)] + ['vol' for i in range(126)]), 'lag': list(range(126)) + list(range(126)),
                                                         'function': lambda df: pd.DataFrame([df.iloc[:, i] * df.iloc[:, i + 126] for i in range(126)]).mean(axis = 0)}
    modified_momentum_factors['weighted_strength_12m'] = {'indicators': (['gain' for i in range(260)] + ['vol' for i in range(260)]), 'lag': list(range(260)) + list(range(260)),
                                                        'function': lambda df: pd.DataFrame([df.iloc[:, i] * df.iloc[:, i + 260] for i in range(260)]).mean(axis = 0)}
    return modified_momentum_factors

def get_codenum_factor():
    codenum_factor = {}
    codenum_factor['codenum_adjacency'] = {'indicators': ['td', 'codenum', 'gain'], 'function': weighted_gain_codenum}
    return codenum_factor

def corr_dropna(list1,list2):
    return pd.DataFrame([list(list1),list(list2)]).dropna(axis = 1).T.corr().iloc[0,1]

def delta(df):
    return pd.DataFrame([df.iloc[:, i + 1] - df.iloc[:, i] for i in range(df.shape[1] - 1)]).T

def chg_div(df):
    length = df.shape[1] // 2
    df1 = delta(df.iloc[:, :length])
    df2 = delta(df.iloc[:, length:])
    return [corr_dropna(df1.iloc[i, :], df2.iloc[i, :]) for i in range(len(df))]

def ranked_chg_div(df):
    if 'td' not in df.columns:
        raise TypeError('"td" should be in the DataFrame!')
    df_td = df['td']
    length = (df.shape[1] - 1) // 2
    # Split into two dataframes for taking the delta
    df1 = delta(df.drop('td', axis = 1).iloc[:, :length])
    df2 = delta(df.drop('td', axis = 1).iloc[:, length:])
    # Concatenate back to a single dataframe
    df = pd.concat([df1, df2], axis = 1)
    df['td'] = df_td
    df = df.groupby('td').transform(lambda x: x.rank()).drop('td', axis = 1)
    return [corr_dropna(df.iloc[i, :length], df.iloc[i, length:]) for i in range(len(df))]
    
def ts_ranked_chg_div(df):
    df['vol'] = df['vol'] - df['vol'].shift(1)
    df['close'] = df['close'] - df['close'].shift(1)
    sub_dfs = []
    for code, sub_df in df.groupby('codenum'):
        sub_df.drop('codenum', axis = 1, inplace = True)
        
        for i in range(9, len(sub_df)):
            sub_df.iloc[i, :] = sub_df.iloc[i-9:i+1, :].rank().iloc[9, :]

        for i in range(9):
            sub_df.iloc[i, :] = np.nan

        sub_df['corr'] = np.nan
        for i in range(20, len(sub_df)):
            sub_df['corr'].iloc[i] = corr_dropna(sub_df['vol'][i-20:i+1], sub_df['close'][i-20:i+1])
        sub_dfs.append(sub_df)
    transformed_df = pd.concat(sub_dfs)
    transformed_df = transformed_df.reindex(df.index)

    return transformed_df['corr']

def get_volume_price_factors():
    volume_price_factors = {}
    # volume_price_factors['FR'] = {'indicators': ['vol' for i in range(21)] + ['total_share' for i in range(21)] + ['gain' for i in range(21)], 'lag': list(range(21)) * 3, 'function': lambda df: [corr_dropna(df.iloc[i, :21].values / df.iloc[i, 21:42].values, df.iloc[i, 42:]) for i in range(len(df))]}
    # volume_price_factors['vp_div'] = {'indicators': ['vol'] * 21 + ['close'] * 21, 'lag': list(range(21)) * 2, 'function': lambda df: [corr_dropna(df.iloc[i, :21], df.iloc[i, 21:]) for i in range(len(df))]}
    # volume_price_factors['ranked_vp_div'] = {'indicators': ['vol'] * 21 + ['close'] * 21, 'lag': list(range(21)) * 2, 'rank': [True] * 21 * 2, 'function': lambda df: [corr_dropna(df.iloc[i, :21], df.iloc[i, 21:]) for i in range(len(df))]}
    # volume_price_factors['vp_chg_div'] = {'indicators': ['vol'] * 22 + ['close'] * 22, 'lag': list(range(22)) * 2, 'function': chg_div}
    # volume_price_factors['ranked_vp_chg_div'] = {'indicators': ['td'] + ['vol'] * 22 + ['close'] * 22, 'lag': [0] + list(range(22)) * 2, 'function': ranked_chg_div}
    volume_price_factors['ts_ranked_vp_chg_div'] = {'indicators': ['codenum', 'vol', 'close'], 'function': ts_ranked_chg_div}
    return volume_price_factors

def get_WQ_factors():
    WQ_factors = {}

    WQ_factors['WQ028'] = {'indicators': ['high', 'low', 'close'], 'function': lambda df: (df.iloc[:, 0] + df.iloc[:, 1]) / 2 - df.iloc[:, 2]}
    WQ_factors['WQ008'] = {'indicators': ['open'] * 10 + ['gain'] * 10, 'lag': list(range(10)) * 2, 'function': lambda df: df.iloc[:, 5:10].values.sum(axis = 1) * df.iloc[:, 15:].values.sum(axis = 1) - df.iloc[:, :5].values.sum(axis = 1) * df.iloc[:, 10:15].values.sum(axis = 1)}

    #这个不行
    #WQ_factors['WQ018'] = {'indicators': ['close' for i in range(21)] + ['open' for i in range(21)], 'lag': list(range(21)) * 2, 'function': lambda df: [corr_dropna(df.iloc[i, :21], df.iloc[i, 21:]) for i in range(len(df))]}
    
    return WQ_factors

def get_all_factors():
    return {**get_basic_factors(), **get_value_factors(), **get_growth_factors(), **get_financial_quality_factors(), **get_leverage_factors(), **get_momentum_factors(), 
            **get_volatility_factors(), **get_turnover_factors(), **get_modified_momentum_factors(), **get_codenum_factor()}