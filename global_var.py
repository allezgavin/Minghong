import numpy as np
import pandas as pd
import datetime
import mysql.connector
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="pandas only supports SQLAlchemy connectable")
warnings.filterwarnings("ignore", category=UserWarning, message="The default dtype for empty Series will be 'object'")

local = True

if local:
    schema_name = "Minghong"
    mydb = mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="20040203",
        database=schema_name
        )
else:
    schema_name = "astocks"
    mydb = mysql.connector.connect(
        host="172.31.50.91",
        user="guest",
        password="MH#123456",
        database=schema_name
        )


def query_SQL_market(indicators):
    if 'market_cap' in indicators:
        indicators.remove('market_cap')
        ind_str = ', '.join(indicators)
        ind_str += ', close * total_share AS market_cap'
    else:
        ind_str = ', '.join(indicators)

    stock_list = ', '.join([f"'{stock}'" for stock in stocks])
    query = f"SELECT td, codenum, {ind_str} FROM market WHERE td BETWEEN {start_date} AND {end_date} and codenum IN ({stock_list}) ORDER BY td ASC;"
    
    df = pd.read_sql(query, mydb)
    
    # 复权
    if 'close' in df.columns:
        df['close_not_recovered'] = df['close']
        for stock, sub_df in df.groupby('codenum'):
            sub_df['close'] = sub_df['close'].iloc[-1] / (np.prod(1 + sub_df['chg'] / 100) / (1 + sub_df['chg'] / 100).cumprod())

    return df

def query_SQL_finance(factors = []):
    factor_list = 'fd, codenum'
    if len(factors) != 0:
        factor_list = factor_list + ', ' + ', '.join(factors)

    stock_list = ', '.join([f"'{stock}'" for stock in stocks])
    query = f"SELECT {factor_list} FROM finance WHERE fd BETWEEN {start_date} AND {end_date} and codenum IN ({stock_list}) ORDER BY fd ASC;"

    finance_df = pd.read_sql(query, mydb)
    year = finance_df['fd'] // 10000
    month_day = finance_df['fd'] % 10000
    month_day = month_day.replace({
        331: 430,
        630: 830,
        930: 1031,
        1231: 10430
    })
    finance_df['disclosure'] = (year * 10000 + month_day).astype('str')
    finance_df['fd'] = finance_df['fd'].astype('str')

    # Delete 4th quarter data
    finance_df = finance_df[~finance_df['fd'].str.contains('1231')]

    return finance_df

def query_SQL_finance_deriv(factors = []):

    factor_list = 'fd, codenum'
    if len(factors) != 0:
        factor_list = factor_list + ', ' + ', '.join(factors)

    stock_list = ', '.join([f"'{stock}'" for stock in stocks])
    query = f"SELECT {factor_list} FROM finance_deriv WHERE fd BETWEEN {start_date} AND {end_date} and codenum IN ({stock_list}) ORDER BY fd ASC;"
    
    finance_deriv_df = pd.read_sql(query, mydb)
    year = finance_deriv_df['fd'] // 10000
    month_day = finance_deriv_df['fd'] % 10000
    month_day = month_day.replace({
        331: 430,
        630: 830,
        930: 1031,
        1231: 10430
    })
    finance_deriv_df['disclosure'] = (year * 10000 + month_day).astype('str')
    finance_deriv_df['fd'] = finance_deriv_df['fd'].astype('str')

    # Delete 4th quarter data
    finance_deriv_df = finance_deriv_df[~finance_deriv_df['fd'].str.contains('1231')]

    return finance_deriv_df

def query_SQL_company():
    #Gets the industry information of the stocks
    stock_list = ', '.join([f"'{stock}'" for stock in stocks])
    query = f'SELECT codenum, SW_c1_name_CN AS industry FROM company WHERE codenum IN ({stock_list})'
    
    return pd.read_sql(query, mydb)

period = 1

# indexcode = '000016.SH' # 上证50
# indexcode = '000300.SH' # 上证300
indexcode = '000905.SH' # 中证500
# indexcode = '000852.SH' # 中证1000
# indexcode = '932000.SH' # 中证2000
# indexcode = '399006.SZ' # 创业板指
# indexcode = '399296.SZ' # 创业板动量成长指数

def query_SQL_indexprice():
    query = f"SELECT td, close, chg / 100 AS gain FROM indexprice WHERE td BETWEEN {start_date} AND {end_date} and indexnum='{indexcode}' ORDER BY td ASC"
    df = pd.read_sql(query, mydb).dropna(subset = ['gain']).reset_index(drop = True)
    df['td'] = df['td'].astype('str')
    df['cumulative'] = df['close'] / df['close'][0]

    # 复权
    df['close'] = df['close'].iloc[-1] / (np.prod(1 + df['gain']) / (1 + df['gain']).cumprod())

    return df

def query_SQL_indexweight():
    bench_query = f"SELECT td, code, weight / 100 AS weight FROM indexweight WHERE indexnum='{indexcode}';"
    return pd.read_sql(bench_query, mydb)

def index_stocks():
    return query_SQL_indexweight()['code'].unique()

stocks = index_stocks()

start_date = 20150101
end_date = int(datetime.date.today().strftime('%Y%m%d'))

if end_date <= start_date:
    raise Exception('end_date <= start_date!')
if start_date < 20020101:
    raise Exception('start_date too early!')

qp_size = 500
max_alpha_exposure = 1
max_style_exposure = 0.5
risk_coef = 1.5

if __name__ == '__main__':
    print(query_SQL_indexprice())
    pass
