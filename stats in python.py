import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.stats.anova import anova_lm
from ISLP import load_data
from ISLP.models import (ModelSpec as ms, summarize, poly)

Boston = load_data('Boston')
x= pd.DataFrame({"intercepts": np.ones(Boston.shape[0]), "lstat": Boston.lstat})
#print(x[:4])
y = Boston["medv"]
model = sm.OLS(y, x)
results = model.fit()
#print(summarize(results))
#To specify a model with only a single feature,
design = ms(["lstat"])
design = design.fit(Boston)
x = design.transform(Boston)
#print(x[:4])
#Confidence intervals for the coefficients and predicting new points
new_df = pd.DataFrame({"lstat": [5, 10, 15]})
new_x = design.transform(new_df)
#print(new_x)
#Now we compute prediction at new_x
new_prediction = results.get_prediction(new_x)
new_prediction.predicted_mean
#Multiple linear regression
X = ms(["lstat", "age"]).fit_transform(Boston)
model1 = sm.OLS(y, X)
results1 = model1.fit()
print(summarize(results1))
