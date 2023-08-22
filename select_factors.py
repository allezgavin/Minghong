from factor_test import *
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from skopt import BayesSearchCV
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from global_var import period

def select_factors():
    factor_pool = pd.read_csv('factors.csv')
    # factor_cols = factor_pool.filter(like = 'factor_').columns.to_list()
    all_style_factors = ['factor_' + key for key in get_style_factors().keys()]
    all_alpha_factors = ['factor_' + key for key in get_alpha_factors().keys()]
    all_style_and_alpha_factors = all_style_factors + all_alpha_factors
    factor_pool['gain_next_sum'] = factor_pool.groupby('codenum')['gain_next'].transform(lambda x: x.rolling(window = period).sum().shift(-period+1))

    selected_td = factor_pool['td'].unique()[-390:] # past 1.5 years
    factor_pool = factor_pool[factor_pool['td'].isin(selected_td)]

    factor_pool = factor_pool.dropna(subset = ['gain_next_sum']).fillna(0) #Drops the most current dates because gain_next is to be predicted
    factor_pool['td'].astype('str')

    # Long term regression
    print('Long term regression starts!')
    alpha_list = np.logspace(0, -7, 20) * period ** 2
    for alpha in alpha_list:
        long_term_factors_selected = set()
        final_model = Lasso(alpha = alpha)
        final_model.fit(factor_pool[all_alpha_factors], factor_pool['gain_next_sum'])
        long_term_factors_selected = set([all_alpha_factors[i] for i in range(len(all_alpha_factors)) if final_model.coef_[i] != 0])
        if len(long_term_factors_selected) > 10:
            break

    # Short term (daily) regression
    print('Short term regression starts!')
    selected_td = factor_pool['td'].unique()[-260:] # the past year
    factor_pool_recent = factor_pool[factor_pool['td'].isin(selected_td)]

    alpha_list = np.logspace(-3, -5, 20) * period ** 2
    for alpha in alpha_list:
        short_term_factors_selected = set()
        short_term_regress_coef = np.array([0 for i in range(len(all_style_and_alpha_factors))])
        for td, sub_df in factor_pool_recent.groupby('td'):
            final_model = Lasso(alpha = alpha)
            final_model.fit(sub_df[all_style_and_alpha_factors], sub_df['gain_next_sum'])
            short_term_regress_coef = np.vstack((short_term_regress_coef, final_model.coef_))
        threshold = 160 - period
        nonzero_num = np.count_nonzero(short_term_regress_coef, axis = 0)
        for i in range(len(all_style_and_alpha_factors)):
            if nonzero_num[i] > threshold:
                short_term_factors_selected.add(all_style_and_alpha_factors[i])
        if len(short_term_factors_selected) > 15:
            break
    
    # Long-short term Comparison
    factors_selected = long_term_factors_selected.union(short_term_factors_selected)
    factors_selected.add('factor_ln_market_cap') # Must be included for market cap neutralization
    factors_selected = list(factors_selected)
    print('\nAll factors selected:', factors_selected)
    print('\nLong-term factors - short-term factors:', long_term_factors_selected.difference(short_term_factors_selected))
    print('\nShort-term factors - long-term factors:', short_term_factors_selected.difference(long_term_factors_selected))
    print('\nIntersection:', long_term_factors_selected.intersection(short_term_factors_selected))
    print('\nMaximum variance inflation factor:', max([variance_inflation_factor(factor_pool[factors_selected].values, i) for i in range(len(factors_selected))]))

    # Write to file
    factor_pool = pd.read_csv('factors.csv')
    factor_pool = factor_pool[['td', 'codenum', 'gain_next'] + factors_selected]
    factor_pool.to_csv('factors_selected.csv', index = False)
    print('factors_selected.csv generated!')

def factor_regression_history(factors_selected = 'factors_selected.csv'):
    if type(factors_selected) == str:
        factors_selected = pd.read_csv(factors_selected)
        to_csv = True
    elif type(factors_selected) == pd.DataFrame:
        to_csv = False
    else:
        raise ValueError('Invalid input for factors_selected!')
    
    factors_selected['gain_next_sum'] = factors_selected.groupby('codenum')['gain_next'].transform(lambda x: x.rolling(window = period).sum().shift(-period+1))
    factors_selected = factors_selected.dropna(subset = ['gain_next_sum']).fillna(0)
    selected_cols = list(factors_selected.filter(like = 'factor_').columns)
    recent = factors_selected[factors_selected['td'] == factors_selected['td'].max()]

    # Calculate factor return history
    # Coarse hyperparameter search
    print('Coarse parameter search starts!')
    alpha_range = np.logspace(0, 6, 7)
    best_score = -np.inf
    for alpha in alpha_range:
        model = Ridge(alpha=alpha)
        scores = cross_val_score(model, recent[selected_cols], recent['gain_next_sum'], cv=KFold(n_splits=3), scoring='neg_mean_squared_error')
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
    bayes_cv.fit(recent[selected_cols], recent['gain_next_sum'])
    print('Fine-tuned lambda:', bayes_cv.best_params_['alpha'])

    factor_return_df = pd.DataFrame()
    pred_v_real = pd.DataFrame()
    residual_df = pd.DataFrame(index = factors_selected.index, columns = ['td', 'codenum', 'residual'])

    for td, sub_df in factors_selected.groupby('td'):
        ridge = Ridge(alpha = bayes_cv.best_params_['alpha'])
        ridge.fit(sub_df[selected_cols], sub_df['gain_next_sum'])
        factor_return_df = pd.concat([factor_return_df, pd.DataFrame([(td, ridge.intercept_, *ridge.coef_)])])
        sub_df['prediction'] = ridge.predict(sub_df[selected_cols])
        residual = sub_df['gain_next_sum'] - sub_df['prediction']
        residual_df.loc[sub_df.index, 'td'] = td
        residual_df.loc[sub_df.index, 'codenum'] = sub_df['codenum']
        residual_df.loc[sub_df.index, 'residual'] = residual
        pred_v_real = pd.concat([pred_v_real, sub_df[['prediction', 'gain_next_sum']]])
    factor_return_df.columns = ['td', 'intercept'] + selected_cols

    r2 = r2_score(pred_v_real['gain_next_sum'], pred_v_real['prediction'])
    n = len(pred_v_real)
    p = factor_return_df.shape[0] * factor_return_df.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    print('Adjusted r-squared:', adjusted_r2)

    if to_csv:
        factor_return_df.replace(0, np.nan).to_csv('factor_return.csv', index = False)
        print('factor_return.csv created!')

        residual_df.to_csv('residual.csv', index = False)
        print('residual.csv created!')

    else:
        return factor_return_df

def update_factor():
    selected_cols = pd.read_csv('factors_selected.csv').filter(like = 'factor_').columns
    selected_factors = [col.replace('factor_', '') for col in selected_cols]
    calc_factors(factors = {factor: get_all_factors()[factor] for factor in selected_factors}, stocks = stocks)[['td', 'codenum', 'gain_next'] + selected_cols].to_csv('selected_factors.csv')
    factor_regression_history()
    print('Selected factors updated!')

def reselect_factors():
    calc_factors()
    select_factors()
    factor_regression_history()

if __name__ == '__main__':
    #reselect_factors()

    select_factors()
    factor_regression_history()