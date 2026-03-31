import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
from statsmodels.api import OLS
import sklearn.model_selection as skm
import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler
from ISLP import load_data
from ISLP.models import ModelSpec as MS
from functools import partial

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from ISLP.models import (Stepwise, sklearn_selected, sklearn_selection_path)
from l0bnb import fit_path

# 1. Sub-selection methods( forward selection)
Hitters = load_data('Hitters')
np.isnan(Hitters['Salary']).sum() # 59 missing values
Hitters = Hitters.dropna()
Hitters.shape # 263 rows and 20 columns
def nCp(sigma2, estimator, X, Y): #Choosing the best model using Cp statistic, the smaller the better.
    "Negarive Cp statistic for estimator"
    n, p = X.shape
    Yhat = estimator.predict(X)
    RSS = np.sum((Y - Yhat) ** 2)
    return (RSS + 2 * p * sigma2) / n
#Estimate the residual variance (our first argument in our nCp function) using the full model:
design = MS(Hitters.columns.drop('Salary')).fit(Hitters)
Y = np.array(Hitters['Salary'])
X = design.transform(Hitters)
sigma2 = OLS(Y, X).fit().scale #Create neg_Cp as our scorer for model selection
neg_Cp = partial(nCp, sigma2) #Partial func freezes the 1st argument with our estimate of the residual variance, so that we can call it with just an estimator as an argument.
strategy = Stepwise.first_peak(design, direction = 'forward', max_terms = len(design.terms))
hitters_MSE = sklearn_selected(OLS, strategy)
hitters_MSE.fit(Hitters, Y)
hitters_MSE.selected_state_
hitters_Cp = sklearn_selected(OLS, strategy, scoring = neg_Cp)
hitters_Cp.fit(Hitters, Y)
hitters_Cp.selected_state_
# Cross validation (Alternative)
strategy = Stepwise.fixed_steps(design, len(design.terms), direction = 'forward')
full_path = sklearn_selection_path(OLS, strategy)
full_path.fit(Hitters, Y)
Yhat_in = full_path.predict(Hitters)
#print(Yhat_in.shape)
mse_fig, ax = subplots(figsize = (8, 8))
insample_mse = ((Yhat_in - Y[:, None]) ** 2).mean(0)
n_steps = insample_mse.shape[0]
ax.plot(np.arange(n_steps), insample_mse, 'k', label = 'In-sample')
ax.set_ylabel('MSE', fontsize = 20)
ax.set_xlabel('# steps of forward stepwise', fontsize = 20)
ax.set_xticks(np.arange(n_steps)[::2])
ax.legend()
ax.set_ylim(50000, 250000)
K = 5 # number of folds for cross validation
kfold = skm.KFold(K, shuffle = True, random_state = 0)
Yhat_cv = skm.cross_val_predict(full_path, Hitters, Y, cv = kfold)
#print(Yhat_cv.shape)
cv_mse = []
for train_idx, test_idx in kfold.split(Y):
    errors = (Yhat_cv[test_idx] - Y[test_idx, None]) ** 2
    cv_mse.append(errors.mean(0)) #Column means
cv_mse = np.array(cv_mse).T
#print(cv_mse.shape)
ax.errorbar(np.arange(n_steps), cv_mse.mean(1), cv_mse.std(1)/np.sqrt(K), label = 'Cross-validated', c = 'r')
ax.set_ylim([50000, 250000])
ax.legend()
mse_fig
# Using the validation set approach
validation = skm.ShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 0)
for train_idx, test_idx in validation.split(Y):
    full_path.fit(Hitters.iloc[train_idx], Y[train_idx])
    Yhat_val = full_path.predict(Hitters.iloc[test_idx])
    errors = (Yhat_val - Y[test_idx, None]) ** 2
    validation_mse = errors.mean(0)
ax.plot(np.arange(n_steps), validation_mse, label = 'Validation set', c = 'b', linestyle = '--')
ax.set_xticks(np.arange(n_steps)[::2])
ax.set_ylim([50000, 250000])
ax.legend()
mse_fig
# Best subset selection using l0bnb
D = design.fit_transform(Hitters)
D = D.drop('intercept', axis = 1)
X = np.asarray(D)
#path_ = fit_path(X, Y, max_nonzeros=X.shape[1]) #returns a list whose values include the fitted coefficients and other few attributes
#primt(path[0].shape) # 263 rows and 19 columns, the first column is the intercept

#2. Ridge regression and the lasso
# Ridge regression
Xs = X - X.mean(0)[None,:]
X_scale = X.std(0)
Xs = Xs / X_scale[None,:]
lambdas = 10**np.linspace(8, -2, 100)/ Y.std()
soln_array = skl.ElasticNet.path(Xs, Y, l1_ratio = 0, alphas = lambdas)[1]
soln_array.shape # 19 rows and 100 columns, each column is the solution for a different value of lambda
soln_path = pd.DataFrame(soln_array.T, columns = D.columns, index = np.log(lambdas))
#Transposed the matrix and turned it into a dataframe to facilitate viewing and plotting
soln_path.index.name = 'negative log(lambda)'
path_fig, ax = subplots(figsize = (8, 8))
soln_path.plot(ax = ax, legend = False)
ax.set_xlabel('$-\log(\lambda)$', fontsize = 20)
ax.set_ylabel('Standardized coefficient', fontsize = 20)
ax.legend(loc = 'upper left')
beta_hat = soln_path.loc[soln_path.index[39]]
#print(lambdas[39], beta_hat) # coefficients at the 40th step
np.linalg.norm(beta_hat) # l2 norm of the coefficients at the 40th step
ridge = skl.ElasticNet(alpha = lambdas[59], l1_ratio = 0)
scaler = StandardScaler(with_mean = True, with_std = True)
pipe = Pipeline(steps = [('scaler', scaler), ('ridge', ridge)])
pipe.fit(X, Y)
np.linalg.norm(ridge.coef_)
# Estimating test errors of ridge regression
validation = skm.ShuffleSplit(n_splits = 1, test_size = 0.5, random_state = 0)
ridge.alpha = 0.01
results = skm.cross_validate(ridge, X, Y, scoring = 'neg_mean_squared_error', cv = validation)
results['test_score'] # test MSE for ridge regression with lambda = 0.01
param_grid = {'ridge_alpha': lambdas}
grid = skm.GridSearchCV(pipe, param_grid, cv = validation, scoring = 'neg_mean_squared_error')
grid.fit(X, Y)
grid.best_params_['ridge_alpha'] # best lambda value for ridge regression
grid.best_estimator_ # best ridge regression model
ridge_fig, ax = subplots(figsize = (8, 8))
ax.errorbar(-np.log(lambdas), -grid.cv_results_['mean_test_score'], yerr = grid.cv_results_['std_test_score']/np.sqrt(K))
ax.set_ylim([50000, 250000])
ax.set_xlabel('$-\log(\lambda)$', fontsize = 20)
ax.set_ylabel('Cross-validated MSE', fontsize = 20)
# Fast cross validation for solution path
ridgeCV = skl.ElasticNetCV(alphas = lambdas, l1_ratio = 0, cv = kfold)
pipeCV = Pipeline(steps = [('scaler', scaler), ('ridgeCV', ridgeCV)])
pipeCV.fit(X, Y)
tuned_ridge = pipeCV.named_steps['ridge']
ridgeCv_fig, ax = subplots(figsize = (8, 8))
ax.errorbar(-np.log(lambdas), tuned_ridge.mse_path_.mean(1), yerr = tuned_ridge.mse_path_.std(1)/np.sqrt(K))
ax.axvline(-np.log(tuned_ridge.alpha_), c = 'k', linestyle = '--')
ax.set_ylim([50000, 250000])
ax.set_xlabel('$-\log(\lambda)$', fontsize = 20)
ax.set_ylabel('Cross-validated MSE', fontsize = 20)
np.min(tuned_ridge.mse_path_.mean(1)) # minimum cross validated MSE for ridge regression
tuned_ridge.coef_ # coefficients for ridge regression with the best lambda value
# Evaluate test error of cross validated ridge
outer_valid = skm.ShuffleSplit(n_splits = 1, test_size = 0.25, random_state = 1)
innner_cv = skm.KFold(n_splits = 5, shuffle = True, random_state = 2)
ridgeCV = skl.ElasticNetCV(alphas = lambdas, l1_ratio = 0, cv = innner_cv)
pipeCV = Pipeline(steps = [('scaler', scaler), ('ridgeCV', ridgeCV)])
results = skm.cross_validate(pipeCV, X, Y, cv = outer_valid, scoring = 'neg_mean_squared_error')
-results['test_score'] # test MSE for cross validated ridge regression

# Lasso regression. The code is similar to ridge regression, except that we set l1_ratio = 1 in the ElasticNet function.

#3. Principal components regression and partial least squares
# Principal components regression
pca = PCA(n_components= 2)
linreg = skl.LinearRegression()
pipe = Pipeline([('pca', pca), ('linreg', linreg), ('scaler', scaler)])
pipe.fit(X, Y)
pipe.named_steps['linreg'].coef_ # coefficients for PCR with 2 principal components
param_grid = {'pca__n_components' : range(1, 20)}
grid = skm.GridSearchCV(pipe, param_grid, cv = kfold, scoring = 'neg_mean_squared_error')
grid.fit(X, Y)
pcr_fig, ax = subplots(figsize = (8, 8))
n_comp = param_grid['pca__n_components']
ax.errorbar(n_comp, -grid.cv_results_['mean_test_score'], grid.cv_results_['std_test_score']/np.sqrt(K))
ax.set_ylabel('Cross-validated MSE', fontsize = 20)
ax.set_xlabel('# principal components', fontsize = 20)
ax.set_xticks(n_comp[::2])
ax.set_ylim([50000, 250000])
Xn = np.zeros((X.shape[0], 1))
cv_null = skm.cross_validate(linreg, Xn, Y, cv = kfold, scoring = 'neg_mean_squared_error')
-cv_null['test_score'].mean()
pipe.named_steps['pca'].explained_variance_ratio_ #Amount of info about the predictors captured using M principal components
# Prtial least squares
pls = PLSRegression(n_components = 2, scale = True)
pls.fit(X, Y)
param_grid = {'n_components' : range(1, 20)}
grid = skm.GridSearchCV(pls, param_grid, cv = kfold, scoring = 'neg_mean_squared_error')
grid.fit(X, Y)
pls_fig, ax = subplots(figsize = (8, 8))
n_comp = param_grid['n_components']
ax.errorbar(n_comp, -grid.cv_results_['mean_test_score'], grid.cv_results_['std_test_score']/np.sqrt(K))
ax.set_ylabel('Cross-validated MSE', fontsize = 20)
ax.set_xlabel('# principal components', fontsize = 20)
ax.set_xticks(n_comp[::2])
ax.set_ylim([50000, 250000])