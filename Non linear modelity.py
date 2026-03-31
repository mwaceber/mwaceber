import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize, poly)
from statsmodels.stats.anova import anova_lm

from pygam import (s as s_gam, l as l_gam, f as f_gam, LinearGAM, LogisticGAM)
from ISLP.transforms import (BSpline, NaturalSpline)
from ISLP.models import bs, ns
from ISLP.pygam import (approx_lam, degrees_of_freedom, plot as plot_gam, anova as anova_gam)

# 1.Polynomial regression and step function
Wage = load_data ('Wage')
y = Wage['wage']
age = Wage['age']
poly_age = MS([poly('age', degree= 4)]).fit(Wage)
M = sm.OLS(y, poly_age.transform(Wage)).fit()
summarize(M)
age_grid = np.linspace(age.min(), age.max(), 100)
#age_df = pd.DataFrame({'age', age_grid})
#def plot_wage_fit(age_df, basis, titles):
#    X = basis.transform(Wage)
#    Xnew = basis.transform(age_df)
#    M = sm.OLS(y, X).fit()
#    preds = M.get_prediction(Xnew)
#    bands = preds.conf_int(alpha = 0.05)
#    fig, ax = subplots(figsize = (8, 8))
#    ax.scatter(age, y, facecolor = 'gray', alpha = 0.5)
#    for val, ls in zip([preds.predicted_mean, bands[:, 0], bands[:, 1]], ['b', 'r--', 'r--']):
#        ax.plot(age_df, val, ls, linewidth = 3)
#    ax.set_title(titles, fontsize = 20)
#    ax.set_xlabel('Age', fontsize = 20)
#    ax.set_ylabel('Wage', fontsize = 20)
#    return ax
#plot_wage_fit(age_df, poly_age, 'Degree-4 Polynomial')
models = [MS([poly('age', degree= d)])for d in range(1, 6)]
Xs = [model.fit_transform(Wage) for model in models]
anova_lm(*[sm.OLS(y, X_).fit() for X_ in Xs])
# Alternative
X = poly_age.transform(Wage)
high_earn = Wage['high_earn'] = y > 250
glm = sm.GLM(y > 250, X, family = sm.families.Binomial())
B = glm.fit()
summarize(B)
#newX = poly_age.transform(age_df)
#preds = B.get_prediction(newX)
#bands = preds.conf_int(alpha = 0.05)
fig, ax = subplots(figsize = (8, 8))
rng = np.random.default_rng(0)
ax.scatter(age + 0.2*rng.uniform(size = y.shape[0]),np.where(high_earn, 0.198, 0.002), fc = 'gray', marker= '|')
#for val, ls in zip([preds.predicted_mean, bands[:,0], bands[:,1]], ['b', 'r--', 'r--']):
#    ax.plot(age_df.values, val, ls, linewidth = 3)
#ax.set_title('Degree-4 Polynomial', fontsize = 20)
ax.set_xlabel('Age', fontsize = 20)
ax.set_ylabel('P(Wage > 250)', fontsize = 20)
ax.set_ylim([0, 0.2])
cut_age = pd.qcut(age, 4) # Step functions
summarize(sm.OLS(y, pd.get_dummies(cut_age)).fit())

#2. Splines
bs_ = BSpline(internal_knots= [25, 40, 60], intercept = True).fit(age)
bs_age = bs_. transform(age)
bs_age.shape
bs_age = MS([bs('age', internal_knots = [25, 40, 60])])
Xbs = bs_age.fit_transform(Wage)
M = sm.OLS(y, Xbs).fit()
summarize(M)
BSpline(df = 6).fit(age).internal_knots_ #df func identifies the complexity of the spline (by columns)
bs_age0 = MS([bs('age', df=3, degree = 0)]).fit(Wage) #to fit a natural spline, we use ns() instead of bs()
Xbs0 = bs_age0.transform(Wage)
summarize(sm.OLS(y, Xbs0).fit())

#3. Smoothing splines and GAMs
x_age = np.array(age).reshape(-1,1)
gam = LinearGAM(s_gam(0, lam = 0.6))
gam.fit(x_age, y)
fig, ax = subplots(figsize = (8, 8))
ax.scatter(age, y, facecolor = 'gray', alpha = 0.5)
for lam in np.logspace(-2, 6, 5) :
    gam = LinearGAM(s_gam(0, lam = lam)).fit(x_age, y)
    ax.plot(age_grid, gam.predict(age_grid), label = '{:.1e}'.format(lam), linewidth = 3)
ax.set_xlabel('Age', fontsize = 20)
ax.set_ylabel('Wage', fontsize = 20)
ax.legend(title = '\\lambda$')
gam_opt = gam.gridsearch(x_age, y)
ax.plot(age_grid, gam_opt.predict(age_grid), label = 'Grid search', linewidth = 4)
ax.legend()
fig
# Alternative in fixing the degree of freedom of the smoothing spline
age_term = gam.terms[0]
lam_4 = approx_lam(x_age, age_term, 4)
age_term.lam = lam_4
degrees_of_freedom(x_age, age_term) #Answer is 4 degrees
fig, ax = subplots(figsize = (8, 8))
ax.scatter(x_age, y, facecolor = 'gray', alpha = 0.3)
for degfre in [1, 3, 4, 8, 15]:
    lam = approx_lam(x_age, age_term, degfre + 1)
    age_term.lam = lam
    gam.fit(x_age, y)
    ax.plot(age_grid, gam.predict(age_grid), label = '{:d}'.format(degfre), linewidth = 4)
ax.set_xlabel('Age', fontsize = 20)
ax.set_ylabel('Wage', fontsize = 20)
ax.legend(title= 'Degrees of freedom')
# Create a new prediction matrix using GAM's features for multivariate regression models
ns_age = NaturalSpline(df = 4).fit(age)
ns_year = NaturalSpline(df = 5).fit(Wage['year'])
Xs = [ns_age.transform(age), ns_year.transform(Wage['year']), pd.get_dummies(Wage['education']).values]
X_bh = np.hstack(Xs)
gam_bh = sm.OLS(y, X_bh).fit()
# Now we create the matrix
age_grid = np.linspace(age.min(), age.max(), 100)
X_age_bh = X_bh.copy()[:100]
X_age_bh[:] = X_bh[:].mean(0)[None,:]
X_age_bh[:, :4] = ns_age.transform(age_grid)
preds = gam_bh.get_prediction(X_age_bh)
bounds_age = preds.conf_int(alpha = 0.05)
partial_age = preds.predicted_mean
center = partial_age.mean()
partial_age -= center
bounds_age -= center
fig, ax = subplots(figsize = (8, 8))
ax.plot(age_grid, partial_age, 'b', linewidth = 3)
ax.plot(age_grid, bounds_age[:, 0], 'r--', linewidth = 3)
ax.plot(age_grid, bounds_age[:, 1], 'r--', linewidth = 3)
ax.set_xlabel('Age')
ax.set_ylabel('Effects on Wage')
ax.set_title('Partial dependence of age on wage', fontsize = 20)
# Convert the categorial series 'education' to its araay representation
gam_full = LinearGAM(s_gam(0) + s_gam(1, n_splines = 7) + f_gam(2, lam = 0))
Xgam = np.column_stack([age, Wage['year'], Wage['education'].cat.codes])
gam_full = gam_full.fit(Xgam, y)
fig, ax = subplots(figsize = (8, 8))
plot_gam(gam_full, 0, ax = ax)
ax.set_xlabel('Age')
ax.set_ylabel('Effect on wage')
ax.set_title('Partial dependence of age on wage - default lam = 0.6', fontsize = 20)
age_term = gam_full.terms[0]
age_term.lam = approx_lam(Xgam, age_term, df = 4+1)
year_term = gam_full.terms[1]
year_term.lam = approx_lam(Xgam, year_term, df = 4+1)
gam_full = gam_full.fit(Xgam, y)
# ANOVA Tests for Additive models. Anova tests determines which of the used models is the best
gam_0 = LinearGAM(age_term + f_gam(2, lam = 0))
gam_0.fit(Xgam, y)
gam_linear = LinearGAM(age_term + l_gam(1, lam=0) + f_gam(2, lam=0))
gam_linear.fit(Xgam, y)
gam_logit = LogisticGAM(age_term + l_gam(1, lam=0) + f_gam(2, lam=0))
gam_logit.fit(Xgam, high_earn)
anova_gam(gam_0, gam_linear, gam_full, gam_logit)
# 4. Local Regression
lowess = sm.nonparametric.lowess
fig, ax = subplots(figsize = (8,8))
ax.scatter(age, y, facecolor = 'gray', alpha = 0.5)
for span in [0.2, 0.5]:
    fitted = lowess(y, age, frac = span, swals = age_grid)
    ax.plot(age_grid, fitted, label = '{:.1f}'. format(span), linewidth = 4)
ax.set_xlabel('Age', fontsize = 20)
ax.set_ylabel('Wage', fontsize = 20)
ax.legend(title = 'span', fontsize = 15)