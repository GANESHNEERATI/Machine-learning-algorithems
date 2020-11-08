# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:38:38 2019

@author: neeratig
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#importing dataset
dataset=pd.read_csv('50_Startups.csv')
#spilting independent and dependent variable
X=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 4].values
#encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lableencoder=LabelEncoder()
X[:, 3]=lableencoder.fit_transform(X[:, 3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()
#avoiding dummy variable trap
X=X[:, 1:]
#spiting data into training and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#fitting multiple linear regression to training set
from sklearn.linear_model import LinearRegression
regress=LinearRegression()
regress.fit(X_train,y_train)
#predicting test set result
y_predict=regress.predict(X_test)
#building optimal using backelimination method
import statsmodels.regression.linear_model as lm
#adding column one column of integer type
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt=X[:, [0,1,2,3,4,5]]
regressor_ols = lm.OLS(endog = y, exog = X_opt).fit()

regressor_ols.summary()

#removing x2 ND PERFORMING BACKELIMINATION AGAIN
X_opt=X[:, [0,1,3,4,5]]
regressor_ols=lm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()
#removing
X_opt=X[:, [0,3,4,5]]
regressor_ols=lm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()
#removing
X_opt=X[:, [0,3,5]]
regressor_ols=lm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()
#removing
X_opt=X[:, [0,3]]
regressor_ols=lm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()

#predicting result after backelimination
X1=X_opt;

y1=y;
from sklearn.model_selection import train_test_split
X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regress=LinearRegression()
regress.fit(X1_train,y1_train)
#predicting test set result
y1_predict=regress.predict(X1_test)






