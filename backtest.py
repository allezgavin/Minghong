import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import statistics
from global_var import *

class BacktestResult():
    def __init__(self, ap, bap, rp, b, wlr, rrr, a, dr, ir, te, j, md, sp, stn, tn, v, rs, sv, trn):
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
        self.turnover = trn
    
    def __str__(self):
        return f'Relative profit: {self.relative_profit}\nAlpha: {self.alpha}\nMax drawdown: {self.maximum_drawdown}\nIR: {self.info_ratio}\nTurnover ratio: {self.turnover}'

def backtest(portfolio_or_pathfile, annual_interest_rate = 0.0165, bench = True):
    if type(portfolio_or_pathfile) == str:
        port = pd.read_csv(portfolio_or_pathfile)
    elif type(portfolio_or_pathfile) == pd.DataFrame:
        port = portfolio_or_pathfile
    else:
        raise Exception('portfolio file type not supported! Please use .csv filepath or pd.DataFrame')
    port['td'] = port['td'].astype('str')
    port = port.sort_values('td', ascending = True)
    turnover = port.groupby('codenum')['weight'].agg(lambda x: (x - x.shift(1, fill_value = 0)).abs().mean()/2).sum()
    
    df = query_SQL_market(indicators = ['open', 'close', 'chg'])
    df['td'] = df['td'].astype('str')
    df = df[(df['td'] >= port['td'].min()) & (df['td'] <= port['td'].max())]
    
    df['rise'] = df['chg'] / 100
    
    df = pd.merge(df, port, how = 'left', on = ['td', 'codenum'])
    df.fillna(0, inplace = True)

    df = df[['td', 'codenum', 'rise', 'weight']]
    df['gain'] = df['rise'] * df['weight']
    gain_by_day = df.groupby('td').sum()['gain'].reset_index(drop=True)
    
    cumulative = (1 + gain_by_day).cumprod() / (1 + gain_by_day[0])
    
    day_total = pd.DataFrame()
    day_total['td'] = df['td'].unique()
    day_total['date'] = pd.to_datetime(day_total['td'], format = '%Y%m%d')
    day_total.sort_values('date', ascending = True, inplace = True)
    day_total['gain'] = gain_by_day
    day_total['cumulative'] = cumulative

    bench = query_SQL_csi300()

    #how = 'inner' only keeps the rows that have matching dates in both dataframes!!!
    merged = day_total.merge(bench, how = 'left', on = 'td')
    merged.columns = ['td', 'date', 'gain_trader', 'cumulative_trader', 'indexprice', 'gain_benchmark', 'cumulative_benchmark']
    merged['cumulative_benchmark'] = merged['cumulative_benchmark'] / merged['cumulative_benchmark'].iloc[0]
    merged.set_index('td', inplace = True)

    if len(merged) < len(bench):
        print('Warning: missing dates in market data!')
    if len(merged) < len(day_total):
        print('Warning: missing dates in CSI300 data!')

    # merged['gain_trader'].to_csv('daily_profit.csv')
    # merged['gain_benchmark'].to_csv('daily_benchmark.csv')
    # merged['cumulative_trader'].to_csv('cumulative_profit.csv')
    # merged['cumulative_benchmark'].to_csv('cumulative_benchmark.csv')

    weekly_profit = merged.groupby(pd.Grouper(key = 'date', freq = 'W-MON'))['gain_trader'].sum().reset_index(drop = True)
    # weekly_profit.to_csv('weekly_profit.csv')
    
    # opens = merged.groupby(pd.Grouper(key='date', freq='W-MON'))['open_benchmark'].first()
    # closes = merged.groupby(pd.Grouper(key='date', freq='W-MON'))['close_benchmark'].last()
    # weekly_bench = (closes['close'] - opens['open']) / opens['open']
    weekly_bench = merged.groupby(pd.Grouper(key = 'date', freq = 'W-MON'))['gain_benchmark'].sum().reset_index(drop = True)
    # weekly_bench.to_csv('weekly_benchmark.csv')

    plt.figure(figsize = (10,5))
    plot_melt = merged[['date', 'cumulative_benchmark', 'cumulative_trader']].melt('date', var_name = 'Legend', value_name = 'Ratio')
    sns.lineplot(x = 'date', y = 'Ratio', data = plot_melt, hue = 'Legend')
    plt.savefig('backtest_result.png')
    plt.show()

    plt.figure(figsize = (10,5))
    plot_melt = merged[['date', 'cumulative_benchmark', 'cumulative_trader']].melt('date', var_name = 'Legend', value_name = 'Ratio')
    sns.lineplot(x = 'date', y = 'Ratio', data = plot_melt, hue = 'Legend')
    plt.yscale('log')
    plt.savefig('backtest_result_log.png')
    plt.show()

    merged['cumulative_hedge'] = merged['cumulative_trader'] - merged['cumulative_benchmark'] + 1
    plt.figure(figsize = (10,5))
    sns.lineplot(x = 'date', y = 'cumulative_hedge', data = merged)
    plt.savefig('backtest_hedge.png')
    plt.show()

    merged['cumulative_hedge'] = merged['cumulative_trader'] - merged['cumulative_benchmark'] + 1
    plt.figure(figsize = (10,5))
    sns.lineplot(x = 'date', y = 'cumulative_hedge', data = merged)
    plt.yscale('log')
    plt.savefig('backtest_hedge_log.png')
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
    track_error = np.std(diff)
    info_ratio = diff.mean() / track_error
    track_error = track_error * np.sqrt(260)

    jensen = relative_profit - beta * (benchmark_annual_profit - annual_interest_rate)

    max_value = merged['cumulative_hedge'][0]
    min_value = merged['cumulative_hedge'][0]
    max_drawdown = 0
    for i in range(1, T):
        if merged['cumulative_hedge'][i] < min_value:
            min_value = merged['cumulative_hedge'][i]
        elif merged['cumulative_hedge'][i] > max_value:
            drawdown = (max_value - min_value) / max_value
            if drawdown > max_drawdown:
                max_drawdown = drawdown
            max_value = merged['cumulative_hedge'][i]
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
                          info_ratio, track_error, jensen, max_drawdown, sharpe, sortino, treynor, volatility, R_squared, semivar, turnover)

def random_portfolio(stock_num):
    query = f'SELECT DISTINCT td FROM market WHERE td BETWEEN {start_date} and {end_date} ORDER BY td ASC'
    dates = pd.read_sql(query, mydb).reset_index(drop = True).astype('str')['td']

    random_port = []
    for date in dates:
        for stock in stocks:
            random_port.append({'td': date, 'codenum': stock, 'weight': random.random() / stock_num})

    pd.DataFrame(random_port).to_csv('random_portfolio.csv', index = False)
    print('Random portfolio generated!')

if __name__ == '__main__':
    # random_portfolio(300)
    backtest('backtest_portfolio.csv')