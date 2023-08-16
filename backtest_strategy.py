from select_factors import update_factor, reselect_factors
from optimization import backtest_portfolio
from backtest import backtest, csi300_stocks
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="please consider using SQLAlchemy")
warnings.filterwarnings("ignore", category=UserWarning, message="The default dtype for empty Series will be 'object'")

def less_than_one_year_earlier(td):
    date_str = str(td)
    date_obj = datetime.strptime(date_str, '%Y%m%d')
    current_date = datetime.today()
    time_difference = current_date - date_obj
    return time_difference.days < 365

# 调仓频率
period = 5

if __name__ == '__main__':
    start_date = 20190101

    if less_than_one_year_earlier(start_date):
        raise ValueError('start_date must be one year earlier to allow calculation of all factors!')
    
    # stocks = csi300_stocks()
    # update_factor(start_date - 10000, period, stocks = stocks) # Use reselect_factors() instead if new factors are added.
    #reselect_factors(start_date - 10000, period = period, stocks = stocks)
    backtest_portfolio(period, start_date)
    result = backtest('backtest_portfolio.csv')
    print(result)