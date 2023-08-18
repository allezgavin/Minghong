from select_factors import update_factor, reselect_factors, select_factors
from optimization import backtest_portfolio
from backtest import backtest, csi300_stocks
from global_var import *

if __name__ == '__main__':
    
    stocks = csi300_stocks()
    # update_factor(start_date - 10000, period, stocks = stocks) # Use reselect_factors() instead if new factors are added.
    # reselect_factors(start_date - 10000, period = period, stocks = stocks)
    select_factors()

    # cannot execute at once
    backtest_portfolio(period, start_date)
    result = backtest('backtest_portfolio.csv')
    print(result)