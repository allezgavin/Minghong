import numpy as np
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
import seaborn as sns
import random
import datetime
import statistics
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="pandas only supports SQLAlchemy connectable")

mydb = mysql.connector.connect(
    host="172.31.50.91",
    user="guest",
    password="MH#123456",
    database="astocks"
    )

class BacktestResult():
    def __init__(self, ap, bap, rp, b, wlr, rrr, a, dr, ir, te, j, md, sp, stn, tn, v, rs, sv):
        self.annual_profit = ap
        self.benchmark_annual_profit = bap
        self.relative_profit = rp
        self.beta = b
        self.win_loss_ratio = wlr
        self.risk_return_ratio = rrr
        self.alpha = a
        self.downside_risk = dr
        self.info_ratio = ir
        self.track_error = te
        self.jensen = j
        self.maximum_drawdown = md
        self.sharpe = sp
        self.sortino = stn
        self.treynor = tn
        self.volatility = v
        self.R_squared = rs
        self.semivar = sv
    
    def __str__(self):
        return f'Relative profit: {self.relative_profit}\nWin-loss ratio: {self.win_loss_ratio}\nRisk-return ratio: {self.risk_return_ratio}\nR-squared: {self.R_squared}'


def query_SQL_market(min_date, max_date, indicators, stocks = []):
    if 'market_cap' in indicators:
        indicators.remove('market_cap')
        ind_str = ', '.join(indicators)
        ind_str += ', close * total_share AS market_cap'
    else:
        ind_str = ', '.join(indicators)

    if len(stocks) != 0:
        stock_list = ', '.join([f"'{stock}'" for stock in stocks])
        query = f"SELECT td, codenum, {ind_str} FROM market WHERE td BETWEEN {min_date} AND {max_date} and codenum IN ({stock_list}) ORDER BY td ASC;"
    else:
        query = f"SELECT td, codenum, {ind_str} FROM market WHERE td BETWEEN {min_date} AND {max_date} ORDER BY td ASC;"
    
    df = pd.read_sql(query, mydb)

    #Many abnormal values in the 'chg' column!
    if 'chg' in df.columns:
        df = df.loc[(df['chg'] <= 10) & (df['chg'] >= -10)].reset_index(drop = True)

    return df

def query_SQL_csi300(start_date, end_date = datetime.date.today().strftime('%Y%m%d')):
 
    query = f"SELECT td, chg / 100 AS gain, open, close FROM indexprice WHERE td BETWEEN {start_date} AND {end_date} and indexnum='000300.SH' ORDER BY td ASC"
    
    df = pd.read_sql(query, mydb).dropna(subset = ['gain']).reset_index(drop = True)
    df['td'] = df['td'].astype('str')
    df['cumulative'] = df['close'] / df['close'][0]

    return df

def query_SQL_csi300_weight():
    bench_query = "SELECT td, code, weight / 100 AS weight FROM indexweight WHERE indexnum = '000300.SH';"
    return pd.read_sql(bench_query, mydb)

def backtest(portofolio_or_pathfile, overnight = True, annual_interest_rate = 0.0165):
    if type(portofolio_or_pathfile) == str:
        port = pd.read_csv(portofolio_or_pathfile)
    elif type(portofolio_or_pathfile) == pd.DataFrame:
        port = portofolio_or_pathfile
    else:
        raise Exception('portofolio file type not supported! Please use .csv filepath or pd.DataFrame')
    port['td'] = port['td'].astype('str')
    port = port.sort_values('td', ascending = True)
    start_date = list(port['td'])[0]
    end_date = list(port['td'])[-1]
    
    df = query_SQL_market(start_date, end_date, indicators = ['open', 'close', 'chg'], stocks = port['codenum'].unique())
    
    if overnight:
        #Compares the closing price with that on the previous day
        df['rise'] = df['chg'] / 100
    else:
        #Compares the closing price with the opening price on the same day
        df['rise'] = (df['close'] - df['open']) / df['open']

    df['td'] = df['td'].astype('str')
    
    df = pd.merge(df, port, how = 'left', on = ['td', 'codenum'])
    df.fillna(0, inplace = True)

    df = df[['td', 'codenum', 'rise', 'weight']]
    df['gain'] = df['rise'] * df['weight']
    gain_by_day = df.groupby('td').sum()['gain']
    gain_by_day = list(gain_by_day)
    
    cumulative = [1 + gain_by_day[0]]
    for i in range(1, len(gain_by_day)):
        cumulative.append((1 + gain_by_day[i]) * cumulative[i - 1])
    
    day_total = pd.DataFrame()
    day_total['td'] = df['td'].unique()
    day_total['date'] = pd.to_datetime(day_total['td'], format = '%Y%m%d')
    day_total.sort_values('date', ascending = True, inplace = True)
    day_total['gain'] = gain_by_day
    day_total['cumulative'] = cumulative

    bench = query_SQL_csi300(start_date, end_date = end_date)

    #how = 'inner' only keeps the rows that have matching dates in both dataframes!!!
    merged = bench.merge(day_total, how = 'inner', on = 'td')
    merged.columns = ['td', 'gain_benchmark', 'open_benchmark', 'close_benchmark', 'cumulative_benchmark', 'date', 'gain_trader', 'cumulative_trader']
    merged.set_index('td', inplace = True)

    if len(merged) < len(bench):
        print('Warning: missing dates in market data!')
    if len(merged) < len(day_total):
        print('Warning: missing dates in CSI300 data!')

    merged['gain_trader'].to_csv('daily_profit.csv')
    merged['gain_benchmark'].to_csv('daily_benchmark.csv')
    merged['cumulative_trader'].to_csv('cumulative_profit.csv')
    merged['cumulative_benchmark'].to_csv('cumulative_benchmark.csv')

    weekly_profit = merged.groupby(pd.Grouper(key = 'date', freq = 'W-MON'))['gain_trader'].sum().reset_index(drop = True)
    weekly_profit.to_csv('weekly_profit.csv')
    
    # opens = merged.groupby(pd.Grouper(key='date', freq='W-MON'))['open_benchmark'].first()
    # closes = merged.groupby(pd.Grouper(key='date', freq='W-MON'))['close_benchmark'].last()
    # weekly_bench = (closes['close'] - opens['open']) / opens['open']
    weekly_bench = merged.groupby(pd.Grouper(key = 'date', freq = 'W-MON'))['gain_benchmark'].sum().reset_index(drop = True)
    weekly_bench.to_csv('weekly_benchmark.csv')

    plt.figure(figsize = (10,5))
    plot_melt = merged[['date', 'cumulative_benchmark', 'cumulative_trader']].melt('date', var_name = 'Legend', value_name = 'Ratio')
    sns.lineplot(x = 'date', y = 'Ratio', data = plot_melt, hue = 'Legend')
    plt.savefig('backtest_result.png')
    plt.show()

    T = len(merged)
    annual_profit = merged['cumulative_trader'].iloc[-1] ** (260 / T) - 1
    benchmark_annual_profit = merged['cumulative_benchmark'].iloc[-1] ** (260 / T) - 1
    relative_profit = annual_profit - benchmark_annual_profit
    beta = np.cov(weekly_profit, weekly_bench, bias = True)[0][1] / np.var(weekly_bench)
    win_loss_ratio = (merged['gain_trader'] > 0).sum() / (merged['gain_trader'] < 0).sum()
    risk_return_ratio = merged['gain_trader'][merged['gain_trader'] > 0].mean() / merged['gain_trader'][merged['gain_trader'] < 0].mean() * (-1)
    alpha = annual_profit - beta * benchmark_annual_profit

    rf = annual_interest_rate / 260
    loss = merged['gain_trader'] - rf
    numerator = np.sum(np.square(loss.loc[loss < 0]))
    downside_risk = np.sqrt(260) * np.sqrt(numerator / (T - 1))

    diff = merged['gain_trader'] - merged['gain_benchmark']
    info_ratio = diff.mean() * np.sqrt(260)
    track_error = np.std(diff)

    jensen = relative_profit - beta * (benchmark_annual_profit - annual_interest_rate)

    max_value = merged['cumulative_trader'][0]
    min_value = merged['cumulative_trader'][0]
    max_drawdown = 0
    for i in range(1, T):
        if merged['cumulative_trader'][i] < min_value:
            min_value = merged['cumulative_trader'][i]
        elif merged['cumulative_trader'][i] > max_value:
            drawdown = (max_value - min_value) / max_value
            if drawdown > max_drawdown:
                max_drawdown = drawdown
            max_value = merged['cumulative_trader'][i]
            min_value = max_value
    drawdown = (max_value - min_value) / max_value
    if drawdown > max_drawdown:
        max_drawdown = drawdown
    
    sharpe = (annual_profit - annual_interest_rate) / (np.std(merged['gain_trader']) * np.sqrt(260))
    sortino = (annual_profit - annual_interest_rate) / downside_risk
    treynor = (annual_profit - annual_interest_rate) / beta
    
    volatility = statistics.stdev(weekly_profit) * np.sqrt(52)

    R_squared = merged['gain_trader'].corr(merged['gain_benchmark']) ** 2
    semivar = np.std(merged['gain_trader'][merged['gain_trader'] < 0])

    return BacktestResult(annual_profit, benchmark_annual_profit, relative_profit, beta, win_loss_ratio, risk_return_ratio, alpha, downside_risk,
                          info_ratio, track_error, jensen, max_drawdown, sharpe, sortino, treynor, volatility, R_squared, semivar)

def csi300_stocks():
    # Returns all CSI300 index stocks, past and current.
    return query_SQL_csi300_weight()['code'].unique()

def random_stocks(stock_num, start_date, end_date = datetime.date.today().strftime('%Y%m%d')):

    query = f'SELECT DISTINCT codenum FROM market WHERE td BETWEEN {start_date} and {end_date}'
    
    stocks = pd.read_sql(query, mydb).reset_index(drop = True).astype('str')
    stocks_selected = []
    for i in range(stock_num):
        stock_ind = random.randint(0, len(stocks) - 1)
        stocks_selected.append(stocks['codenum'][stock_ind])
    return stocks_selected

def random_portofolio(start_date, end_date = datetime.date.today().strftime('%Y%m%d'), stock_num = 500):

    stocks_selected = random_stocks(stock_num, start_date, end_date = end_date)

    query = f'SELECT DISTINCT td FROM market WHERE td BETWEEN {start_date} and {end_date} ORDER BY td ASC'
    
    dates = pd.read_sql(query, mydb).reset_index(drop = True).astype('str')['td']

    random_port = []
    for date in dates:
        for stock in stocks_selected:
            random_port.append({'td': date, 'codenum': stock, 'weight': random.random() / stock_num})

    pd.DataFrame(random_port).to_csv('random_portofolio.csv', index = False)
    print('Random portofolio generated!')

if __name__ == '__main__':

    #start_date must be no earlier than 2002
    start_date = 20190101
    end_date = 20230101
    if end_date <= start_date:
        raise Exception('end_date <= start_date!')
    if start_date < 20020101:
        raise Exception('start_date too early!')

    random_portofolio(start_date, end_date = end_date)
    backtest('random_portofolio.csv')