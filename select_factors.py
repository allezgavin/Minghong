from factor_test import *
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from skopt import BayesSearchCV
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

def select_factors():
    factor_pool = pd.read_csv('factors.csv')
    selected_td = factor_pool['td'].unique()[-780:] # past 3 years
    factor_pool = factor_pool[factor_pool['td'].isin(selected_td)]

    factor_cols = factor_pool.filter(like = 'factor_').columns

    factor_pool = factor_pool.dropna(subset = ['gain_next']).fillna(0) #Drops the most current date because gain_next is to be predicted
    factor_pool['td'].astype('str')

    # Long term regression
    print('Long term regression starts!')

    final_model = Lasso(alpha = 0.0003)
    final_model.fit(factor_pool[factor_cols], factor_pool['gain_next'])
    long_term_factors_selected = set([factor_cols[i] for i in range(len(factor_cols)) if final_model.coef_[i] != 0])

    # Short term (daily) regression
    print('Short term regression starts!')
    selected_td = factor_pool['td'].unique()[-130:] # the past 6 months
    factor_pool = factor_pool[factor_pool['td'].isin(selected_td)]
    recent = factor_pool[factor_pool['td'] == factor_pool['td'].max()]

    short_term_factors_selected = set()
    short_term_regress_coef = np.array([0 for i in range(len(factor_cols))])
    for td, sub_df in factor_pool.groupby('td'):
        final_model = Lasso(alpha = 0.0003)
        final_model.fit(sub_df[factor_cols], sub_df['gain_next'])
        short_term_regress_coef = np.vstack((short_term_regress_coef, final_model.coef_))
    threshold = 80 #80 out of 130 days
    nonzero_num = np.count_nonzero(short_term_regress_coef, axis = 0)
    for i in range(len(factor_cols)):
        if nonzero_num[i] > threshold:
            short_term_factors_selected.add(factor_cols[i])
    
    # Long-short term Comparison
    factors_selected = long_term_factors_selected.union(short_term_factors_selected)
    factors_selected.add('factor_ln_market_cap') # Must be included for market cap neutralization
    factors_selected = list(factors_selected)
    print('All factors selected:', factors_selected)
    print('Long-term factors - short-term factors:', long_term_factors_selected.difference(short_term_factors_selected))
    print('Short-term factors - long-term factors:', short_term_factors_selected.difference(long_term_factors_selected))
    print('Intersection:', long_term_factors_selected.intersection(short_term_factors_selected))
    print('Maximum variance inflation factor:', max([variance_inflation_factor(recent[factors_selected].values, i) for i in range(len(factors_selected))]))

    # Write to file
    factor_pool = pd.read_csv('factors.csv')
    factor_pool = factor_pool[['td', 'codenum', 'gain_next'] + factors_selected]
    factor_pool.to_csv('factors_selected.csv', index = False)
    print('factors_selected.csv generated!')

def factor_regression_history():

    factors_selected = pd.read_csv('factors_selected.csv').dropna(subset = ['gain_next']).fillna(0)
    selected_cols = list(factors_selected.filter(like = 'factor_').columns)
    recent = factors_selected[factors_selected['td'] == factors_selected['td'].max()]

    # Calculate factor return history
    # Coarse hyperparameter search
    print('Coarse parameter search starts!')
    alpha_range = np.logspace(0, 6, 7)
    best_score = -np.inf
    for alpha in alpha_range:
        model = Ridge(alpha=alpha)
        scores = cross_val_score(model, recent[selected_cols], recent['gain_next'], cv=KFold(n_splits=3), scoring='neg_mean_squared_error')
        avg_score = np.mean(scores)
        
        if avg_score > best_score:
            best_score = avg_score
            best_alpha = alpha
    print(f'Coarse search result: {best_alpha}')
    
    param_space = {
        'alpha': (best_alpha / 10, best_alpha * 10, 'log-uniform')  # Log-uniform prior for alpha
    }
    # Define the search strategy using Bayesian Optimization
    # Finer Search
    print('Bayes parameter search starts!')
    bayes_cv = BayesSearchCV(
        Ridge(),
        param_space,
        n_iter=30,  # Number of iterations for the optimization
        cv=KFold(n_splits=3),
        scoring='neg_mean_squared_error',
        n_jobs=-1,  # Use all available CPU cores
    )
    bayes_cv.fit(recent[selected_cols], recent['gain_next'])
    print('Fine-tuned lambda:', bayes_cv.best_params_['alpha'])

    factor_return_df = pd.DataFrame()
    pred_v_real = pd.DataFrame()
    residual_df = pd.DataFrame(index = factors_selected.index, columns = ['td', 'codenum', 'residual'])

    for td, sub_df in factors_selected.groupby('td'):
        ridge = Ridge(alpha = bayes_cv.best_params_['alpha'])
        ridge.fit(sub_df[selected_cols], sub_df['gain_next'])
        factor_return_df = pd.concat([factor_return_df, pd.DataFrame([(td, ridge.intercept_, *ridge.coef_)])])
        sub_df['prediction'] = ridge.predict(sub_df[selected_cols])
        residual = sub_df['gain_next'] - sub_df['prediction']
        residual_df.loc[sub_df.index, 'td'] = td
        residual_df.loc[sub_df.index, 'codenum'] = sub_df['codenum']
        residual_df.loc[sub_df.index, 'residual'] = residual
        pred_v_real = pd.concat([pred_v_real, sub_df[['prediction', 'gain_next']]])
    factor_return_df.columns = ['td', 'intercept'] + selected_cols

    r2 = r2_score(pred_v_real['gain_next'], pred_v_real['prediction'])
    n = len(pred_v_real)
    p = factor_return_df.shape[0] * factor_return_df.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    print('Adjusted r-squared:', adjusted_r2)

    factor_return_df.to_csv('factor_return.csv', index = False)
    print('factor_return.csv created!')

    residual_df.to_csv('residual.csv', index = False)
    print('residual.csv created!')

def update_factor_history(start_date, stocks = []):
    selected_cols = pd.read_csv('factors_selected.csv').filter(like = 'factor_').columns
    selected_factors = [col.replace('factor_', '') for col in selected_cols]
    calc_factors(start_date, factors = {factor: get_all_factors()[factor] for factor in selected_factors}, stocks = stocks)[['td', 'codenum', 'gain_next'] + selected_cols].to_csv('selected_factors.csv')
    print('Selected factors updated!')
    factor_regression_history()

def reselect_factors(start_date, stocks = []):
    calc_all_factors(start_date, stocks = stocks)
    select_factors()
    factor_regression_history()

if __name__ == '__main__':

    start_date = 20210101
    end_date = 20230730

    if start_date >= end_date:
        raise ValueError('Date Error!')
    
    # random.seed(4)
    # stocks_tested = list(set(list(random_stocks(300, start_date, end_date)) + list(csi300_stocks())))
    # reselect_factors(start_date, stocks = stocks_tested)

    #select_factors()
    factor_regression_history()