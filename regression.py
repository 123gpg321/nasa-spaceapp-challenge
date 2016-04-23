#https://github.com/justmarkham/DAT4/blob/master/notebooks/08_linear_regression.ipynb
#https://gist.github.com/komasaru/75bd0abfbe95814c50bb
#https://www.flightradar24.com/data/flights/w64306

# imports
import pandas as pd
import matplotlib.pyplot as plt

# read data into a DataFrame
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
data.head()

feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]
y = data.Sales

# follow the usual sklearn pattern: import, instantiate, fit
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X, y)

zip(feature_cols, lm.coef_)

# predict for a new observation
lm.predict([100, 25, 25])

# calculate the R-squared
lm.score(X, y)
#0.89721063817895208