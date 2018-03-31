# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values # IVs
y = dataset.iloc[:, 4].values # DV

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting MLR to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() # create object
regressor.fit(X_train, y_train) # using fit method, fit the multiple regressor to training set

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
# add b0 column based from the formula: y = b0 + b1X1 + ... + bnXn 
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1) # axis = 1 for column, 0 for row

# start backward elimination
X_opt = X[:, [0, 1, 2, 3, 4, 5]] # create new optimal matrix of features, this will create a high impact on the dependent variable
# BE STEP 2 new regressor new object for sm class ORDINARY LEAST SQUARES
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # Fit the full model with all possible predictors
regressor_OLS.summary()

# start backward elimination
X_opt = X[:, [0, 1, 3, 4, 5]] # create new optimal matrix of features, this will create a high impact on the dependent variable
# BE STEP 2 new regressor new object for sm class ORDINARY LEAST SQUARES
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # Fit the full model with all possible predictors
regressor_OLS.summary()

# start backward elimination
X_opt = X[:, [0, 3, 4, 5]] # create new optimal matrix of features, this will create a high impact on the dependent variable
# BE STEP 2 new regressor new object for sm class ORDINARY LEAST SQUARES
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # Fit the full model with all possible predictors
regressor_OLS.summary()

# start backward elimination
X_opt = X[:, [0, 3, 5]] # create new optimal matrix of features, this will create a high impact on the dependent variable
# BE STEP 2 new regressor new object for sm class ORDINARY LEAST SQUARES
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # Fit the full model with all possible predictors
regressor_OLS.summary()

# start backward elimination
X_opt = X[:, [0, 3]] # create new optimal matrix of features, this will create a high impact on the dependent variable
# BE STEP 2 new regressor new object for sm class ORDINARY LEAST SQUARES
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # Fit the full model with all possible predictors
regressor_OLS.summary()