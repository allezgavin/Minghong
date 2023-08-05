from factor_test import *
from scipy.optimize import minimize
from datetime import timedelta

mydb = mysql.connector.connect(
        host="172.31.50.91",
        user="guest",
        password="MH#123456",
        database="astocks"
        )

factors = pd.read_csv('factors.csv')
selected_cols = pd.read_csv('factors_selected.csv').filter(like = 'factor_').columns
factor_return = pd.read_csv('factor_return.csv', index_col = 'td')
residual_df = pd.read_csv('residual.csv', index_col = 'td')
bench_query = "SELECT td, code, weight / 100 AS weight FROM indexweight WHERE indexnum = '000300.SH';"
bench_weight = pd.read_sql(bench_query, mydb)
industry_info = query_SQL_company().set_index('codenum')
current_holding = pd.Series()

def set_variables(td = (datetime.datetime.now() + timedelta(days=1)).strftime('%Y%m%d'), backtest = False):
    global X, stock_num, expected_f, F, V, p_B, S, current_holding
    current_exposure = factors.loc[factors['td'] == td]

    if len(current_exposure) == 0:
        raise Exception('{td} is not a trading day!')
    
    X = current_exposure.set_index('codenum')[selected_cols]
    stock_num = X.shape[0]

    residual_var = residual_df[residual_df.index < td].groupby('codenum').var()
    residual_var = residual_var.reindex(X.index).fillna(0.001)
    Delta = np.diag(residual_var['residual'].values)

    # Filter the rows prior to td (for backtesting)
    history_factor_return = factor_return.loc[factor_return.index < td]
    expected_f = history_factor_return.iloc[:, 1:].mean() #exlude the intercept

    F = history_factor_return.iloc[:, 1:].cov()

    V = X @ F @ X.T + Delta
    
    bench_td_max = bench_weight[bench_weight['td'] <= td]['td'].max()
    concurrent_bench_weight = bench_weight[bench_weight['td'] == bench_td_max]
    p_B = concurrent_bench_weight.set_index('code')['weight'].reindex(X.index).fillna(0)

    industry_df = industry_info.reindex(X.index).fillna('unknown')
    S = pd.get_dummies(industry_df)

    if backtest:
        current_holding = current_holding.reindex(X.index).fillna(0)
    else:
        current_holding = pd.read_csv('current_holding.csv')
        current_holding = pd.Series(data = current_holding.iloc[:, 1], index = current_holding['codenum'])
        current_holding = current_holding.reindex(X.index).fillna(0)


# Define the objective function
def objective_function(weight):
    risk_tolerance_coef = 1

    expected_stock_return = X @ expected_f.T
    port_return = weight @ expected_stock_return

    exchange = weight - current_holding
    selling_fee = 0.00127 * (exchange[exchange < 0].sum())
    buying_fee = 0.00027 * (exchange[exchange > 0].sum())
    # Must calculate a minimum fee of 5 CNY in actual trading, and a minimum exchange amount of 100 shares.
    exchange_fee = selling_fee + buying_fee

    return - port_return + exchange_fee + risk_tolerance_coef * weight @ V @ weight.T

def weight_sum_constraint(weight):
    return np.sum(weight) - 1

def industry_neutral_constraint(weight):
    return S.T @ (weight - p_B)

def positive_max_exposure_constraint(weight):
    max_factor_exposure = 0.1
    return max_factor_exposure - X.T @ weight

def negative_max_exposure_constraint(weight):
    max_factor_exposure = 0.1
    return X.T @ weight + max_factor_exposure


def optimize(backtest = False):

    # Define the constraints as dictionary
    constraints = ({'type': 'eq', 'fun': weight_sum_constraint},
                {'type': 'eq', 'fun': industry_neutral_constraint},
                {'type': 'ineq', 'fun': positive_max_exposure_constraint},
                {'type': 'ineq', 'fun': negative_max_exposure_constraint})
    
    # Define the bounds for h (each element between 0 and 1)
    bounds = [(0, 1)] * stock_num

    initial_guess = np.ones(stock_num) / stock_num

    # Solve the optimization problem
    result = minimize(objective_function, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    # Retrieve the optimal value of h
    optimal_weight = result.x
    optimal_weight = pd.Series(data = optimal_weight, index = X.index)

     # Print the results
    print("Optimal weight:")
    print(optimal_weight.sort_values(ascending = False).head(20))

    global current_holding

    if backtest:
        current_holding = optimal_weight
    else:
        optimal_weight.to_csv('current_holding.csv')

    return optimal_weight

def backtest_iteration(td):
    set_variables(td = td, backtest = True)
    print('optimization begins')
    optimal_weight = optimize(backtest = True)
    return optimal_weight

def backtest_portfolio(start_date, end_date = datetime.date.today().strftime('%Y%m%d')):
    all_td = pd.read_csv('factors.csv')['td'].unique()
    all_td = [td for td in all_td if td >= start_date and td <= end_date]
    portfolio = pd.DataFrame(columns = ['td', 'codenum', 'weight'])
    portfolio.to_csv('backtest_portfolio.csv', index = False)
    for td in all_td:
        print('Calculating portfolio:', td)
        optimal_weight = backtest_iteration(td)
        td_port = pd.DataFrame()
        td_port['td'] = [td for i in range(len(optimal_weight))]
        td_port['codenum'] = optimal_weight.index
        td_port['weight'] = optimal_weight.values
        previous_port = pd.read_csv('backtest_portfolio.csv')
        pd.concat([previous_port, td_port]).to_csv('backtest_portfolio.csv', index = False)

if __name__ == '__main__':
    #backtest_portfolio(20220101, end_date=20220120)
    backtest_iteration(20220110)