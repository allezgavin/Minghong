from select_factors import update_factor, reselect_factors, select_factors
from optimization import backtest_portfolio
from backtest import backtest
from global_var import *

if __name__ == '__main__':
    
    # update_factor() # Use reselect_factors() instead if new factors are added.
    # reselect_factors()
    # select_factors()

    # # cannot execute at once
    # backtest_portfolio()
    result = backtest('backtest_portfolio.csv')
    print(result)