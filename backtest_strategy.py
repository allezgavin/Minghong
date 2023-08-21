from select_factors import update_factor, reselect_factors, select_factors, factor_regression_history
from backtest import backtest
from global_var import *

if __name__ == '__main__':
    
    # update_factor() # Use reselect_factors() instead if new factors are added.
    # reselect_factors()

    select_factors()
    factor_regression_history()

    from optimization import backtest_portfolio
    backtest_portfolio()
    result = backtest('backtest_portfolio.csv')
    print(result)