from select_factors import update_factor_history, reselect_factors
from optimization import backtest_portofolio
from backtest import backtest, csi300_stocks
from datetime import datetime

def less_than_one_year_earlier(td):
    date_str = str(td)
    date_obj = datetime.strptime(date_str, '%Y%m%d')
    current_date = datetime.today()
    time_difference = current_date - date_obj
    return time_difference.days < 365

if __name__ == '__main__':
    start_date = 20220101

    if less_than_one_year_earlier(start_date):
        raise ValueError('start_date must be one year earlier to allow calculation of all factors!')
    
    stocks = csi300_stocks()

    #update_factor_history(start_date, stocks = stocks) # Use reselect_factors() instead if new factors are added.
    reselect_factors(start_date)
    backtest_portofolio(start_date, stocks = stocks)
    backtest('backtest_portofolio.csv')