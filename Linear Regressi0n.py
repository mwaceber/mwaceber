import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from matplotlib.pyplot import subplots
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize, poly)

# 1. Simple Linear Regression
# We gonna use Boston dataset for this chapter to predict the median value(medv) using
# 13 predictors eg: average no of rooms per house(rmvar), age, percentage of lower status population(lstat) etc.

Boston = load_data('Boston')
X = pd.DataFrame({'Intercepts': np.ones(Boston.shape[0]), 'Lstat': Boston['lstat']})
#print(X[:4]). Extract the response and fit it to the model
y = Boston['medv']
model = sm.OLS(y, X) #Specifize the model
results = model.fit() # Fit the model
# summarize () from ISLP package gives a summary of the fitted model including 
# the parameter estimates, standard errors, t-statistics, and p-values for each predictor variable.
#print(summarize(results))
# Lets try use occupy multiple predictors for the model

design = MS(['lstat'])
design = design.fit(Boston) # Checks if the predictors exist in the dataset 
X = design.transform(Boston) # Constructs the model matrix
# Another way is by combining the two processes
design = MS(['lstat'])
X = design.fit_transform(Boston)
y = Boston['medv']
model = sm.OLS(y, X)
results = model.fit()
#Now lets compute predicted mean,cofine intervals and prediction intervals for a range of lstat values with age fixed at its mean value
new_df = pd.DataFrame({'lstat':[5, 10, 15]})
mewX = design.transform(new_df)
new_pred = results.get_prediction(mewX)
# print(new_pred.predicted_mean)
# print(new_pred.conf_int(alpha = 0.05)) # confidence intervals
#print(new_pred.conf_int(obs = True, alpha = 0.05)) # prediction intervals

# Defining a function to plot the fitted regression line 
def abline(ax, b, m, *args, **kwargs):
    """Add an intercept and slope to an axis."""
    xlim = ax.get_xlim()
    ylim = [m * xlim[0] + b, m * xlim[1] + b]
    ax.plot(xlim, ylim, *args, **kwargs)

ax = Boston.plot.scatter('lstat', 'medv')
abline(ax, results.params[0], results.params[1], 'r--', linewidth = 3)
ax  = subplots(figsize = (8, 8))[1]
ax.scatter(results.fittedvalues, results.resid)
ax.set_xlabel('Fitted Values')
ax.set_ylabel('Residuals')
ax.axhline(0, c = 'k', ls= '--')

infl = results.get_influence()
ax = subplots(figsize = (8, 8))[1]
ax.scatter(np.arange(X.shape[0]), infl.hat_matrix_diag)
ax.set_xlabel('Index')
ax.set_ylabel('Leverage')
np.argmax(infl.hat_matrix_diag) # index of the observation with the highest leverage
# plot.show()

# 2. Multiple Linear Regression
terms = Boston.columns.drop('medv')
X = MS(terms).fit_transform(Boston)
model = sm.OLS(y, X)
results = model.fit()

#3. Multivariate Goodness of Fit
vals = [VIF(X,i) for i in range(1, X.shape[1])]
vif = pd.DataFrame({'vif': vals}, index = X.columns[1:])
#for repetitive operations
vals = []
for i in range(1, X.shape[1]):
    vals.append(VIF(X.values, i))
# For interaction terms, include tuple eg: ('lstat', 'age') in the model specification

#4. Non-linear Transformations of the Predictors
X = MS([poly('lstat', degree = 2), 'age']).fit_transform(Boston)
model1 = sm.OLS(y, X)
results1 = model1.fit() # Poly creates a basis matrix for inclusion in the model matrix
anova_lm(results, results1) # Compare the two models using anova