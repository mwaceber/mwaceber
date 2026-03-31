import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from matplotlib.pyplot import subplots
from ISLP import load_data
from ISLP.models import (ModelSpec as MS , summarize)
import statsmodels.api as sm

from ISLP import confusion_table
from ISLP.models import contrast
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis as LDA , QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. Logistic Regression
Smarket = load_data("Smarket")
allvars = Smarket.columns.drop(['Today', 'Direction', 'Year'])
design = MS(allvars)
X = design.fit_transform(Smarket)
y = Smarket.Direction == 'Up'
#To specify a logistic regression model, we can use the GLM function from statsmodels with the family argument set to Binomial
glm = sm.GLM(y, X, family = sm.families.Binomial())
results = glm.fit()
#print(summarize(results))
#Create a vector of class predics based on whether the predicted probability of a market increase is greater than or less than 0.5
probs = results.predict()
labels = np.array(['Down'] * 1250)
labels[probs > 0.5] = 'Up'
#print(confusion_table(labels ,Smarket.Direction))
# To increase accuracy of the logistic regression, we fit the model using part of the data and 
# then examine how well it predicts the held out data
train = (Smarket. Year < 2005)
Smarket_train = Smarket.loc[train]
Smarket_test = Smarket.loc[~train]
X_train, X_test = X.loc[train], X.loc[~train]
y_train, y_test = y.loc[train], y.loc[~train]
glm_train = sm.GLM(y_train, X_train, family = sm.families.Binomial())
results_train = glm_train.fit()
probs = results_train.predict(exog = X_test)
D = Smarket.Direction
L_train, L_test = D.loc[train], D.loc[~train]
labels= np.array(['Down'] * 252)
labels[probs > 0.5] = 'Up'
#print(confusion_table(labels, L_test))
# Compute the test set error rate
#print (np.mean(labels == L_test), np.mean(labels != L_test))
# Now lets use predictors with a more direct relationship with the response variable
model = MS(['Lag1', 'Lag2'])
X = model.fit_transform(Smarket)
X_train, X_test = X.loc[train], X.loc[~train]
y_train, y_test = y.loc[train], y.loc[~train]
glm_train = sm.GLM(y_train, X_train, family = sm.families.Binomial())
results = glm_train.fit()
probs = results.predict(exog=X_test)
labels = np.array(['Down'] * 252)
labels[probs > 0.5] = 'Up'
#print(confusion_table(labels, L_test))

#2. Linear Discriminant Analysis
lda = LDA(store_covariance= True)
X_train, X_test = [M.drop(columns = ['intercept'])for M in [X_train, X_test]]
lda.fit(X_train, L_train)
lda_pred = lda.predict(X_test)
#print(confusion_table(lda_pred, L_test))
# Estimate probability for each class at each point in training set.
# Applying a threshold of 0.5 to the predicted probabilities allows to recreate predictions in lda_pred
lda_prob = lda.predict_proba(X_test) # predict_proba() estimates the probability of each class for each observation in the test set. The output is a 2D array where each row corresponds to an observation and each column corresponds to a class. The values
np.all(np.where(lda_prob[:,1] >= 0.5, 'Up','Down') == lda_pred)
# Quadratic Discriminant Analysis operations are similar as LDA but with a different model specification

# 4. Naive Bayes
NB = GaussianNB()
NB.fit(X_train, L_train)
X_train[L_train == 'Down'].mean() # mean of each predictor for the 'Down' class
X_train[L_train == 'Down'].var() # variance of each predictor for the 'Down' class
nb_labels = NB.predict(X_test)
#print(confusion_table(nb_labels, L_test))

#5. K-Nearest Neighbors
caravan = load_data('Caravan')
purchase = caravan.Purchase
purchase.value_counts()
feature_df = caravan.drop(columns = ['Purchase'])
scaler = StandardScaler( with_mean = True, with_std = True, copy = True)
X_std = scaler.fit_transform(feature_df)
feature_std = pd.DataFrame(X_std, columns = feature_df.columns)
(X_train, X_test, y_train, y_test) = train_test_split(feature_std, purchase, test_size = 1000, random_state = 0)
knn1 = KNeighborsClassifier(n_neighbors = 1)
knn1_pred = knn1.fit(X_train, y_train).predict(X_test)
np.mean(y_test != knn1_pred), np.mean(y_test != 'No')
confusion_table(knn1_pred, y_test)

#No of neighbours in KNN (Tuning parameters/ Hyperparameters)
for K in range (1,6):
    knn = KNeighborsClassifier(n_neighbors = K)
    knn_pred = knn.fit(X_train, y_train).predict(X_test)
    C = confusion_table(knn_pred, y_test)
    templ = ('K = {0:d}: # predicted to rent: {1:>2},' + 
            ' # who did rent {2:d}, accuracy {3:.1%}' )
    pred = C.loc['Yes'].sum()
    did_rent = C.loc['Yes', 'Yes']
    #print(templ.format(K, pred, did_rent, did_rent / pred))

logit = LogisticRegression(C = 1e10, solver = 'liblinear')
logit.fit(X_train, y_train)
logit_pred = logit.predict_proba(X_test)
logit_labels = np.where(logit_pred[:,1] > 0.25, 'Yes', 'No')
#print(confusion_table(logit_labels, y_test))

#Linear vs Poisson Regression
Bike = load_data('Bikeshare')
X = MS(['mnth', 'hr', 'workingday', 'weathersit', 'temp']).fit_transform(Bike)
Y = Bike['bikers']
ans = sm.OLS(Y, X).fit()
summarize(ans)
# Alternative coding to include hr[0] and mnth[Jan]
hr_encode = contrast('hr', 'sum')
mnth_encode = contrast('mnth', 'sum')
X2 = MS([mnth_encode, hr_encode, 'workingday', 'weathersit', 'temp']).fit_transform(Bike)
ans2 = sm.OLS(Y, X2).fit()
S2 = summarize(ans2)
np.sum(ans.fittedvalues - ans2.fittedvalues)**2 # the fitted values are the same for both models
coef_month = S2[S2.index.str.contains('mnth')]['coef']
months = Bike['mnth'].dtype.categories
coef_month = pd.concat([coef_month, pd.Series([-coef_month.sum()], index = ['mnth[Dec]'])])
fig_month, ax_month = subplots(figsize = (8, 8))
x_month = np.arange(coef_month.shape[0])
ax_month.plot(x_month, coef_month, marker = 'o', ms = 10)
ax_month.set_xticks(x_month)
ax_month.set_xticklabels([l[5] for l in coef_month.index], fontsize = 20)
ax_month.set_xlabel('Month', fontsize = 20)
ax_month.set_ylabel('Coefficient', fontsize = 20)
# Now for the hours
coef_hr = S2[S2.index.str.contains('hr')]['coef']
coef_hr = coef_hr.reindex(['hr[{0}]'.format(h) for h in range(23)])
coef_hr = pd.concat([coef_hr, pd.Series([-coef_hr.sum()], index = ['hr[23]'])])
fig_hr, ax_hr = subplots(figsize = (8, 8))
x_hr = np.arange(coef_hr.shape[0])
ax_hr.plot(x_hr, coef_hr, marker = 'o', ms = 10)
ax_hr.set_xticks(x_hr[::2])
ax_hr.set_xticklabels(range(24)[::2], fontsize = 20)
ax_hr.set_xlabel('Hour', fontsize = 20)
ax_hr.set_ylabel('Coefficient', fontsize = 20) 
# Poisson regression
ans_poi = sm.GLM(Y, X2, family = sm.families.Poisson()).fit()
S_pois=summarize(ans_poi)
coef_month = S_pois[S_pois.index.str.contains('mnth')]['coef']
coef_month = pd.concat([coef_month, pd.Series([-coef_month.sum()], index = ['mnth[Dec]'])])
coef_hr = S_pois[S_pois.index.str.contains('hr')]['coef']
coef_hr = pd.concat([coef_hr, pd.Series([-coef_hr.sum()], index = ['hr[23]'])])
fig_pois, (ax_month_pois, ax_hr_pois) = subplots(1, 2, figsize = (16, 8))
ax_month.plot(x_month, coef_month, marker = 'o', ms = 10)
ax_month.set_xticks(x_month)
ax_month.set_xticklabels([l[5] for l in coef_month.index], fontsize = 20)
ax_month.set_xlabel('Month', fontsize = 20)
ax_month.set_ylabel('Coefficient', fontsize = 20)
ax_hr.plot(x_hr, coef_hr, marker = 'o', ms = 10)
ax_hr.set_xticklabels(range(24)[::2], fontsize = 20)
ax_hr.set_xlabel('Hour', fontsize = 20)
ax_hr.set_ylabel('Coefficient', fontsize = 20)
#Compare the fitted values from the linear and Poisson regression models
fig, ax = subplots(figsize = (8, 8))
ax.scatter(ans.fittedvalues, ans_poi.fittedvalues, s = 0.5)
ax.set_xlabel('Linear Regression Fit', fontsize = 20)
ax.set_ylabel('Poisson Regression Fit', fontsize = 20)
ax.axline([0,0], c = 'k', ls = '--', linewidth = 3, slope = 1)
#plot.show()