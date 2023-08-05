from factor_test import *
from datetime import timedelta
from cvxopt import matrix, solvers
from backtest import backtest, query_SQL_csi300_weight
solvers.options['show_progress'] = False

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
bench_weight = query_SQL_csi300_weight()
industry_info = query_SQL_company().set_index('codenum')
current_holding = pd.Series()

def set_variables(td = (datetime.datetime.now() + timedelta(days=1)).strftime('%Y%m%d')):
    global X, stock_num, expected_f, F, V, p_B, S
    current_exposure = factors.loc[factors['td'] == td]

    if len(current_exposure) == 0:
        raise Exception('missing data for the specified date!')
    
    X = current_exposure.set_index('codenum')[selected_cols]
    stock_num = X.shape[0]

    residual_var = residual_df[residual_df.index <= td].groupby('codenum').var()
    residual_var = residual_var.reindex(X.index).fillna(0.001)
    Delta = np.diag(residual_var['residual'].values)

    # Filter the rows prior to td (for backtesting)
    history_factor_return = factor_return.loc[factor_return.index <= td]
    expected_f = history_factor_return.iloc[:, 1:].mean() #exlude the intercept

    F = history_factor_return.iloc[:, 1:].cov()

    V = X @ F @ X.T + Delta
    
    bench_td_max = bench_weight[bench_weight['td'] <= td]['td'].max()
    concurrent_bench_weight = bench_weight[bench_weight['td'] == bench_td_max]
    p_B = concurrent_bench_weight.set_index('code')['weight'].reindex(X.index).fillna(0)

    industry_df = industry_info.reindex(X.index).fillna('unknown')
    S = pd.get_dummies(industry_df)

def optimize():
    x_k = 0.05 # Maximum factor exposure
    k = 1 # Risk aversion coefficient
    # Have not considered transaction costs

    # Objective function
    P = matrix(np.array(2 * k * V))
    q = matrix(np.array(-X @ expected_f))

    # Maxium factor exposure constraint and minimum weight constraint
    G = np.vstack((X.T, -X.T)) # maximum factor exposure constraint
    h = np.ones((2 * X.shape[1], 1)) * x_k

    G = np.vstack((G, np.eye(X.shape[0]) * -1)) # minimum weight constraint
    h = np.vstack((h, p_B.values.reshape(-1, 1)))
    G = matrix(G)
    h = matrix(h)

    # Industry neutral constraint and weight sum constraint
    last_row = np.ones((1, S.shape[0]))
    # A = matrix(np.vstack((S.T, last_row)))
    # b = matrix(np.zeros((S.shape[1] + 1, 1)))

    A = matrix(last_row)
    b = matrix(np.zeros((1,1)))
    sol = solvers.qp(P, q, G, h, A, b)
    optimal_weight = (p_B + list(sol['x']))
    return optimal_weight

def backtest_iteration(td):
    set_variables(td = td)
    optimal_weight = optimize()
    return optimal_weight

def backtest_portfolio(start_date, end_date = datetime.date.today().strftime('%Y%m%d')):
    all_td = pd.read_csv('factors.csv')['td'].unique()
    all_td = [td for td in all_td if td >= start_date and td <= end_date]
    portfolio = pd.DataFrame(columns = ['td', 'codenum', 'weight'])
    for td in all_td:
        print('Optimizing', td)
        optimal_weight = backtest_iteration(td)
        td_port = pd.DataFrame()
        td_port['td'] = [td for i in range(len(optimal_weight))]
        td_port['codenum'] = optimal_weight.index
        td_port['weight'] = optimal_weight.values
        portfolio = pd.concat([portfolio, td_port])
    portfolio.to_csv('backtest_portfolio.csv', index = False)

if __name__ == '__main__':
    pass