from factor_test import *
from datetime import timedelta
from scipy.stats import ttest_1samp
from cvxopt import matrix, solvers
from backtest import query_SQL_csi300_weight
solvers.options['show_progress'] = False

mydb = mysql.connector.connect(
        host="172.31.50.91",
        user="guest",
        password="MH#123456",
        database="astocks"
        )

factors_selected = pd.read_csv('factors_selected.csv').fillna(0)
factor_return = pd.read_csv('factor_return.csv', index_col = 'td')
residual_df = pd.read_csv('residual.csv', index_col = 'td')
bench_weight = query_SQL_csi300_weight()
industry_info = query_SQL_company().set_index('codenum')
current_holding = pd.Series()

def set_variables(td = (datetime.datetime.now() + timedelta(days=1)).strftime('%Y%m%d')):
    global X, Xa, Xr, stock_num, expected_f, F, V, p_B, S
    
    X = factors_selected[factors_selected['td'] == td].set_index('codenum')
    stock_num = X.shape[0]

    if stock_num == 0:
        raise Exception('missing data for the specified date!')

    residual_var = residual_df[residual_df.index <= td].groupby('codenum').var()
    residual_var = residual_var.reindex(X.index).fillna(0.001)
    Delta = np.diag(residual_var['residual'].values)

    # Filter the rows prior to td (for backtesting)
    history_factor_return = factor_return.loc[factor_return.index <= td].iloc[-130:] # Past 6 months

    # Distinguish alpha and risk factors
    alpha_factors = []
    risk_factors = []

    for factor in X.columns:
        t_statistic, p_value = ttest_1samp(X[factor], 0)

        # Assuming a significance level of 0.05
        if p_value < 0.05:
            alpha_factors.append(factor)
        else:
            risk_factors.append(factor)
    
    Xa = X[alpha_factors]
    Xr = X[risk_factors]
    
    expected_f = history_factor_return.iloc[:, 1:].mean() #exlude the intercept

    F = history_factor_return.iloc[:, 1:].cov()

    V = X @ F @ X.T + Delta
    
    bench_td_max = bench_weight[bench_weight['td'] <= td]['td'].max()
    concurrent_bench_weight = bench_weight[bench_weight['td'] == bench_td_max]
    p_B = concurrent_bench_weight.set_index('code')['weight'].reindex(X.index).fillna(0)

    industry_df = industry_info.reindex(X.index).fillna('unknown')
    S = pd.get_dummies(industry_df['industry'], drop_first = True)

def optimize(current_holding = np.zeros(len(p_B), 1)):
    x_k = 0.05 # Maximum alpha factor exposure
    k = 1 # Risk aversion coefficient
    
    # Objective function
    P = matrix(np.array(2 * k * V))
    q = matrix(np.array(-X @ expected_f))

    # Inequality constraint
    # Maximum alpha factor exposure constraint
    G1 = np.vstack((Xa.T, -Xa.T))
    h1 = np.ones((2 * Xa.shape[1], 1)) * x_k

    # Minimum weight constraint
    G2 = np.eye(X.shape[0]) * -1
    h2 = p_B.values.reshape(-1, 1)

    G = matrix(np.vstack(G1, G2))
    h = matrix(np.vstack(h1, h2))

    # Equality constraint
    # Industry neutral constraint
    A1 = S.T
    b1 = np.zeros((S.shape[1], 1))

    # Zero risk factor exposure constraint
    A2 = Xr.T
    b2 = np.zeros((Xr.shape[1]), 1)

    # Weight sum constraint
    A3 = np.ones((1, S.shape[0]))
    b3 = np.zeros((1, 1))

    A = matrix(np.vstack(A1, A2, A3))
    b = matrix(np.vstack(b1, b2, b3))


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