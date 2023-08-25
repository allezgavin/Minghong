import multiprocessing
from factor_test import *
from datetime import timedelta
from scipy.stats import ttest_1samp
from cvxopt import matrix, solvers
from backtest import query_SQL_indexweight
from global_var import *
solvers.options['show_progress'] = False

def EWMA(time_series, halflife, step = 1):
    
    if len(time_series.shape) == 2:
        mat = pd.DataFrame(np.zeros_like(time_series), index = time_series.index, columns = time_series.columns)
        for j in range(mat.shape[1]):
            mat.iloc[:, j] = EWMA(time_series.iloc[:, j], halflife, step = step)
        return mat
    
    time_series = (time_series / 1).fillna(0) # Convert to float

    alpha = 1 - 2 ** (-1/halflife)
    ewma = np.zeros_like(time_series)  # Initialize with zeros of the same shape
    ewma[0:step] = time_series.iloc[0:step] * alpha  # First values
    
    for i in range(step, len(ewma)):
        ewma[i] = (1 - alpha) * ewma[i-step] + alpha * time_series.iloc[i]
    
    return ewma

def EWMV(time_series, halflife, step = 1):
    
    if len(time_series.shape) == 2:
        mat = pd.DataFrame(np.zeros_like(time_series), index = time_series.index, columns = time_series.columns)
        for j in range(mat.shape[1]):
            mat.iloc[:, j] = EWMV(time_series.iloc[:, j], halflife, step = step)
        return mat

    time_series = (time_series / 1).fillna(0)

    alpha = 1 - 2 ** (-1/halflife)
    ewmv = np.zeros_like(time_series)  # Initialize with zeros of the same shape
    ewma_series = EWMA(time_series, halflife)
    
    for i in range(step, len(time_series)):
        ewmv[i] = (1 - alpha) * ewmv[i-step] + alpha * (time_series.iloc[i] - ewma_series[i-step]) ** 2
    
    return ewmv


print('Reading factor_return.csv...')
factor_return = pd.read_csv('factor_return.csv', index_col = 'td')

print('Reading factors_selected.csv...')
factors_selected = pd.read_csv('factors_selected.csv')

print('Reading residual.csv...')
residual_df = pd.read_csv('residual.csv')

print('Querying indexweight table...')
bench_weight = query_SQL_indexweight()
industry_info = query_SQL_company().set_index('codenum')
all_style_factors = ['factor_' + key for key in get_style_factors().keys()]
all_alpha_factors = ['factor_' + key for key in get_alpha_factors().keys()]
factor_cols = factors_selected.filter(like = 'factor_').columns.to_list()
style_factor_cols = [style_factor for style_factor in factor_cols if style_factor in all_style_factors]
alpha_factor_cols = [alpha_factor for alpha_factor in factor_cols if alpha_factor in all_alpha_factors]
size_factor_col = ['factor_ln_market_cap']

print('Calculating EWMA of factor return...')
factor_return_pred = pd.concat([EWMA(factor_return[alpha_factor_cols], 36, step = period),
                                EWMA(factor_return[style_factor_cols], 18, step = period),
                                EWMA(factor_return[size_factor_col], 18, step = period)], axis = 1)[factor_cols]
residual_df['residual_var_pred'] = residual_df.groupby('codenum')['residual'].transform(EWMV, halflife=36, step=1) # Past 36 days (not 36 periods)

IC_series = pd.read_csv(f'IC_series.csv', index_col='td')

print('Calculating EWMA of IC...')
IC_pred = pd.concat([EWMA(IC_series[alpha_factor_cols], 36, step = period),
                    EWMA(IC_series[style_factor_cols], 18, step = period),
                    EWMA(IC_series[size_factor_col], 18, step = period)], axis = 1)[factor_cols].T
print('Tables all set!')

def set_variables(td = (datetime.datetime.now() + timedelta(days=1)).strftime('%Y%m%d')):
    global X, Xa, Xb, Xs, expected_Xf, F, V, p_B, S, sub_rows

    bench_td_max = bench_weight[bench_weight['td'] <= td]['td'].max()
    concurrent_bench_weight = bench_weight[bench_weight['td'] == bench_td_max]
    p_B = concurrent_bench_weight.set_index('code')['weight']

    X = factors_selected[factors_selected['td'] == td].set_index('codenum')[factor_cols]
    X = X.reindex(p_B.index).dropna(axis=0, how='all') # Some stocks may be missing due to missing data/dataframe merging in calc_factors()
    p_B = p_B.loc[X.index]

    if X.shape[0] == 0:
        print('Missing data!')
        raise Exception('Missing data for the specified date!')
    if X.isna().all().any():
        print('Missing factor!')
        raise Exception('Missing factor for the specific date!')
    X = X.fillna(0)

    sub_rows = (X @ IC_pred[td]).sort_values(ascending = False).iloc[:qp_size].index

    Xa = X[alpha_factor_cols]
    Xb = X[style_factor_cols]
    Xs = X[size_factor_col]

    residual_var = residual_df[residual_df['td'] == td].set_index('codenum')['residual_var_pred']
    residual_var = residual_var.reindex(sub_rows).fillna(residual_var.mean()) # Residual data may be missing due to null values in 'gain_next' col
    Delta = np.diag(residual_var.values)

    expected_fa = factor_return_pred[factor_return_pred.index == td][alpha_factor_cols].T
    expected_fb = factor_return_pred[factor_return_pred.index == td][style_factor_cols].T

    expected_Xf = Xa.loc[sub_rows] @ expected_fa + Xb.loc[sub_rows] @ expected_fb

    # Assuming F is constant over time
    F = factor_return.loc[factor_return.index <= td].iloc[:, 1:].cov().fillna(0) # only one non null value would lead to a null value in the cov matrix
    V = X.loc[sub_rows] @ F @ X.loc[sub_rows].T + Delta

    industry_df = industry_info.reindex(X.index).fillna('unknown')
    S = pd.get_dummies(industry_df['industry'], drop_first = True)

def loose_qp(P,q,G,h,A,b):
    tol_list = np.logspace(-7, -1, 20)
    for tol in tol_list:
        G_mod = matrix(np.vstack((G, A, -A)))
        h_mod = matrix(np.vstack((h, b+tol, -b+tol)))

        try:
            sol = solvers.qp(P, q, G_mod, h_mod)
            return sol
        except ValueError:
            pass
    raise Exception('QP failed!')

def optimize(k, x_a, x_b, must_full = True, loose = True):
    # Objective function
    P = matrix(np.array(2 * k * V))
    q = matrix(np.array(-expected_Xf))

    # Inequality constraint
    # Maximum alpha factor exposure constraint
    G1 = np.vstack((Xa.loc[sub_rows].T, -Xa.loc[sub_rows].T))
    bench_alpha_exposure = Xa.T @ p_B
    h1 = np.ones((2 * Xa.shape[1], 1)) * x_a + np.concatenate([bench_alpha_exposure, -bench_alpha_exposure]).reshape(-1,1)

    # Maximum style factor exposure constraint
    G2 = np.vstack((Xb.loc[sub_rows].T, -Xb.loc[sub_rows].T))
    bench_style_exposure = Xb.T @ p_B
    h2 = np.ones((2 * Xb.shape[1], 1)) * x_b + np.concatenate([bench_style_exposure, -bench_style_exposure]).reshape(-1,1)

    # Minimum weight constraint
    G3 = np.eye(qp_size) * -1
    h3 = np.zeros((qp_size, 1))

    G = matrix(np.vstack((G1, G2, G3)))
    h = matrix(np.vstack((h1, h2, h3)))

    # Equality constraint
    # Industry neutral constraint
    A1 = S.loc[sub_rows].T
    mask = A1.any(axis = 1)
    A1 = A1.loc[mask]
    bench_industry = (S.T @ p_B).loc[mask]
    bench_industry = bench_industry / bench_industry.sum()
    b1 = np.array(bench_industry).reshape(-1,1)

    # Zero market cap exposure constraint
    A2 = Xs.loc[sub_rows].T
    b2 = np.array(Xs.T @ p_B).reshape(-1,1)

    A = matrix(np.vstack((A1, A2)))
    b = matrix(np.vstack((b1, b2)))

    # Weight sum constraint
    if must_full:
        A3 = np.ones((1, qp_size))
        b3 = np.ones((1, 1))
        A = matrix(np.vstack((A, A3)))
        b = matrix(np.vstack((b, b3)))

    try:
        sol = solvers.qp(P, q, G, h, A, b)
    except ValueError:
        if loose:
            warnings.warn('Constraints loosened!')
            sol = loose_qp(P, q, G, h, A, b)
        else:
            raise Exception('QP failed! Consider using loose=True')

    optimal_weight = pd.Series(data = sol['x'], index = sub_rows)
    optimal_weight[optimal_weight < (0.01 / qp_size)] = 0

    optimal_excess_exposure = X.loc[sub_rows].T @ optimal_weight - X.T @ p_B

    return optimal_weight, optimal_excess_exposure

def backtest_iteration(td, k = risk_coef, x_a = max_alpha_exposure, x_b = max_style_exposure):
    print('Optimizing', td)
    try:
        set_variables(td = td)
    except Exception:
        print(f'Missing data for {td}')
        return np.nan, np.nan
        
    optimal_weight, optimal_exposure = optimize(k, x_a, x_b)

    return optimal_weight, optimal_exposure

def backtest_portfolio(k = risk_coef, x_a = max_alpha_exposure, x_b = max_style_exposure, multiprocess = False):
    print(f'Optimizing Portfolio.\nk = {k}\nx_a = {x_a}\nx_b = {x_b}\nmultiprocess = {multiprocess}\n')
    all_td = pd.read_csv('factor_return.csv')['td'].unique()
    all_td = [td for td in all_td if td >= int(start_date) and td <= int(end_date)]
    portfolio = pd.DataFrame(columns = ['td', 'codenum', 'weight'])
    exposure = pd.DataFrame(columns = ['td'] + factor_cols)

    if multiprocess:
        with multiprocessing.Pool(processes=6) as pool:
            results = pool.starmap(backtest_iteration, [(all_td[i * period], k, x_a, x_b) for i in range(len(all_td) // period)])
    else:
        results = []
        for i in range(len(all_td) // period):
            results.append(backtest_iteration(all_td[i * period], k, x_a, x_b))

    for i in range(len(results)):
        optimal_weight = results[i][0]
        optimal_exposure = results[i][1]

        # Missing data
        try:
            len(optimal_weight)
        except:
            continue

        period_port = pd.DataFrame(columns = ['td', 'codenum', 'weight'])
        # +1 day between optimization and purchase of the stock
        period_port['td'] = [all_td[n + i * period] + 1 for n in range(period) for j in range(len(optimal_weight))]
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

def param_search(control = True):
    # x_a_range = np.linspace(0.2, 2, 5)
    # x_b_range = np.linspace(0, 0.5, 5)
    # k_range = np.logspace(-1, 2, 5)
    x_a_range = np.array([])
    x_b_range = np.array([])
    k_range = np.logspace(-3, 1.5, 15)
    x_a_range = np.unique(np.append(x_a_range, max_alpha_exposure))
    x_b_range = np.unique(np.append(x_b_range, max_style_exposure))
    k_range = np.unique(np.append(k_range, risk_coef))

    if control:
        test_combo = [(k, max_alpha_exposure, max_style_exposure) for k in k_range]
        test_combo += [(risk_coef, x_a, max_style_exposure) for x_a in x_a_range]
        test_combo += [(risk_coef, max_alpha_exposure, x_b) for x_b in x_b_range]
    else:
        test_combo = [(k, x_a, x_b) for k in k_range for x_a in x_a_range for x_b in x_b_range]

    index = pd.MultiIndex.from_tuples(test_combo, names=['k', 'x_a', 'x_b'])
    result = pd.DataFrame(index=index, columns=['Alpha', 'IR', 'Max_drawdown'])

    for combo in test_combo:
        backtest_portfolio(*combo)
        backtest_result = backtest('backtest_portfolio.csv', transaction_fee=0.0005, fig=False)
        result.loc[combo] = [backtest_result.alpha, backtest_result.info_ratio, backtest_result.maximum_drawdown]

    result.to_csv('param_search.csv')
    return result

if __name__ == '__main__':
    # set_variables(td = 20160729)
    # for each in [X, Xa, Xb, Xs, expected_Xf, F, V, p_B, S, sub_rows, qp_size]:
    #     print(each)

    # t = backtest_iteration(20160729)
    # print(t[0].sort_values(ascending = False))
    # print(t[0].sum())
    # print(t[1])
    
    # backtest_portfolio()

    param_search()