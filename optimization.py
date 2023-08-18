import multiprocessing
from factor_test import *
from datetime import timedelta
from scipy.stats import ttest_1samp
from cvxopt import matrix, solvers
from backtest import query_SQL_csi300_weight
from global_var import *
solvers.options['show_progress'] = False

def EWMA(time_series, halflife, step = 1):
    time_series = time_series / 1 # Convert to float
    if len(time_series.shape) == 2:
        mat = pd.DataFrame(np.zeros_like(time_series)).reindex_like(time_series)
        for j in range(mat.shape[1]):
            mat.iloc[:, j] = EWMA(time_series.iloc[:, j], halflife, step = step)
        return mat

    alpha = 1 - 2 ** (-1/halflife)
    ewma = np.zeros_like(time_series)  # Initialize with zeros of the same shape
    ewma[0:step] = time_series.iloc[0:step] * alpha  # First values
    
    for i in range(step, len(ewma)):
        ewma[i] = (1 - alpha) * ewma[i-step] + alpha * time_series[i]
    
    return pd.Series(ewma).reindex_like(time_series)

def EWMV(time_series, halflife, step = 1):
    time_series = time_series / 1
    if len(time_series.shape) == 2:
        mat = pd.DataFrame(np.zeros_like(time_series)).reindex_like(time_series)
        for j in range(mat.shape[1]):
            mat.iloc[:, j] = EWMV(time_series.iloc[:, j], halflife, step = step)
        return mat

    alpha = 1 - 2 ** (-1/halflife)
    ewmv = np.zeros_like(time_series)  # Initialize with zeros of the same shape
    ewma_series = EWMA(time_series, halflife)
    
    for i in range(step, len(time_series)):
        ewmv[i] = (1 - alpha) * ewmv[i-step] + alpha * (time_series[i] - ewma_series.iloc[i-step]) ** 2
    
    return pd.Series(ewmv).reindex_like(time_series)

factors_selected = pd.read_csv('factors_selected.csv').fillna(0)
factor_return = pd.read_csv('factor_return.csv', index_col = 'td')
residual_df = pd.read_csv('residual.csv', index_col = 'td')
bench_weight = query_SQL_csi300_weight()
industry_info = query_SQL_company().set_index('codenum')
all_style_factors = get_style_factors().keys()
all_alpha_factors = get_alpha_factors().keys()
factor_cols = factors_selected.filter(like = 'factor_').columns.to_list()
style_factor_cols = [style_factor for style_factor in factor_cols if style_factor in all_style_factors]
alpha_factor_cols = [alpha_factor for alpha_factor in factor_cols if alpha_factor in all_alpha_factors]

factor_return_pred = pd.concat([EWMA(factor_return[alpha_factor_cols], 36, step = period),
                                EWMA(factor_return[style_factor_cols], 18, step = period)], axis = 1)[factor_cols]
residual_var_pred = residual_df.groupby('codenum')['weight'].transform(EWMV, halflife=36, step=1) # Past 36 days (not 36 periods)


def set_variables(td = (datetime.datetime.now() + timedelta(days=1)).strftime('%Y%m%d')):
    global X, Xa, Xb, stock_num, expected_Xf, F, V, p_B, S
    
    X = factors_selected[factors_selected['td'] == td].set_index('codenum')[factor_cols]
    stock_num = X.shape[0]

    if stock_num == 0:
        raise ValueError('missing data for the specified date!')
    if X.isna().all().any():
        raise ValueError('missing factor for the specific date!')
    X = X.fillna(0)

    Xa = X[alpha_factor_cols]
    Xb = X[style_factor_cols]

    residual_var = residual_var_pred[residual_var_pred['td'] == td]
    residual_var = residual_var.reindex(X.index).fillna(residual_var.mean())
    Delta = np.diag(residual_var['residual'].values)

    expected_fa = factor_return_pred[factor_return_pred.index == td][alpha_factor_cols]
    expected_fb = factor_return_pred[factor_return_pred.index == td][style_factor_cols]

    expected_Xf = Xa @ expected_fa + Xb @ expected_fb

    # Assuming F is constant over time
    F = factor_return.loc[factor_return.index <= td].iloc[:, 1:].cov()
    V = X @ F @ X.T + Delta
    
    bench_td_max = bench_weight[bench_weight['td'] <= td]['td'].max()
    concurrent_bench_weight = bench_weight[bench_weight['td'] == bench_td_max]
    p_B = concurrent_bench_weight.set_index('code')['weight'].reindex(X.index).fillna(0)

    industry_df = industry_info.reindex(X.index).fillna('unknown')
    S = pd.get_dummies(industry_df['industry'], drop_first = True)

def optimize(must_full = False):
    x_k = 1 # Maximum alpha factor exposure
    k = 10 # Risk aversion coefficient
    
    # Objective function
    P = matrix(np.array(2 * k * V))
    q = matrix(np.array(-expected_Xf))

    # Inequality constraint
    # Maximum alpha factor exposure constraint
    G1 = np.vstack((Xa.T, -Xa.T))
    h1 = np.ones((2 * Xa.shape[1], 1)) * x_k

    # Minimum weight constraint
    G2 = np.eye(X.shape[0]) * -1
    h2 = p_B.values.reshape(-1, 1)

    G = matrix(np.vstack((G1, G2)))
    h = matrix(np.vstack((h1, h2)))

    # Equality constraint
    # Industry neutral constraint
    A1 = S.T
    b1 = np.zeros((S.shape[1], 1))

    # Zero risk factor exposure constraint
    A2 = Xb.T
    b2 = np.zeros((Xb.shape[1], 1))

    A = matrix(np.vstack((A1, A2)))
    b = matrix(np.vstack((b1, b2)))

    # Weight sum constraint
    if must_full:
        A3 = np.ones((1, S.shape[0]))
        b3 = np.zeros((1, 1))
        A = matrix(np.vstack((A, A3)))
        b = matrix(np.vstack((b, b3)))
    
    sol = solvers.qp(P, q, G, h, A, b)
    optimal_weight = (p_B + list(sol['x']))
    optimal_weight[optimal_weight < 0.01 / len(optimal_weight)] = 0

    optimal_excess_exposure = X.T @ list(sol['x'])
    return optimal_weight, optimal_excess_exposure

def backtest_iteration(td):
    print('Optimizing', td)
    set_variables(td = td)

    try:
        optimal_weight, optimal_exposure = optimize()
        return optimal_weight, optimal_exposure
    except ValueError:
        print(f'Missing data for {td}')
        return np.nan, np.nan

def backtest_portfolio():
    all_td = pd.read_csv('factor_return.csv')['td'].unique()
    all_td = [td for td in all_td if td >= int(start_date) and td <= int(end_date)]
    portfolio = pd.DataFrame(columns = ['td', 'codenum', 'weight'])
    exposure = pd.DataFrame(columns = ['td'] + factor_cols)

    with multiprocessing.Pool(processes=6) as pool:
        results = pool.starmap(backtest_iteration, [(period, all_td[i * period]) for i in range(len(all_td) // period)])
    
    for i in range(len(results)):
        optimal_weight = results[i][0]
        optimal_exposure = results[i][1]

        # Missing data
        try:
            len(optimal_weight)
        except:
            continue

        period_port = pd.DataFrame(columns = ['td', 'codenum', 'weight'])
        period_port['td'] = [all_td[n + i * period] for n in range(period) for j in range(len(optimal_weight))]
        period_port['codenum'] = list(optimal_weight.index) * period
        period_port['weight'] = list(optimal_weight.values) * period
        portfolio = pd.concat([portfolio, period_port])

        period_exposure = pd.DataFrame(columns = ['td'] + factor_cols)
        period_exposure['td'] = [all_td[n + i * period] for n in range(period)]
        period_exposure[factor_cols] = [list(optimal_exposure)] * period
        exposure = pd.concat([exposure, period_exposure])

    portfolio.to_csv('backtest_portfolio.csv', index = False)
    exposure.to_csv('backtest_exposure.csv', index = False)
    print('Backtest portofolio generated!')

if __name__ == '__main__':
    print(backtest_iteration(20,20230104).sort_values(ascending = False))
    #backtest_portfolio(200, 20190101)