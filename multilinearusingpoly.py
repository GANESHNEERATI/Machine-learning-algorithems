# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:19:30 2019

@author: neeratig
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#importing boston housing dataset
from sklearn.datasets import load_boston
boston_dataset=load_boston()
boston=pd.DataFrame(boston_dataset.data,columns=boston_dataset.feature_names)
boston.head()
boston['MEDV']=boston_dataset.target
boston.isnull().sum()
#exploatary data analysis
#sns.set(rc={'figure.figure_size':(11.7,8.27)})
sns.distplot(boston['MEDV'],bins=30)
plt.show()
#orrelation matrix that measures the linear relationships between the variables.

correlation_matrix=boston.corr().round(2)
# annot = True to print the values inside the square
sns.heatmap(annot=True,data=correlation_matrix)
plt.figure(figsize=(20,5))

features=['LSTAT','RM']
target=boston['MEDV']

for i,col in enumerate(features):
    plt.subplot(1,len(features),i+1)
    x=boston[col]
    y=target
    plt.scatter(x,y,marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')
    
    
#prepairing trining data model
X=pd.DataFrame(np.c_[boston['LSTAT'],boston['RM']],columns=['LSTAT','RM'])
y=boston['MEDV']
#spliting data into training and testing set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#applying polynominal regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
def create_polynominal_regression(degree):
    "creates a model for polynominal regression for given degreee"
    poly_features=PolynomialFeatures(degree=degree)
    #transforming the exisiting feature higher degree feature
    X_train_poly=poly_features.fit_transform(X_train)
    #fit the transformed feature to linear regression
    poly_model=LinearRegression()
    poly_model.fit(X_train_poly,y_train)
    #predicting on traing data set
    y_train_predict=poly_model.predict(X_train_poly)
    #predicting test dataset
    y_test_predict=poly_model.predict(poly_features.fit_transform(X_test))
    #evaluating model on training dataset
    rmse_train=np.sqrt(mean_squared_error(y_train,y_train_predict))
    r2_train=r2_score(y_train,y_train_predict)
    #evaluating model on test dataset
    rmse_test=np.sqrt(mean_squared_error(y_test,y_test_predict))
    r2_test=r2_score(y_test,y_test_predict)
    print('model performence for training set')
    print('------------------------------------------')
    print('RMSE IS{}'.format(rmse_train))
    print('Rc is{}'.format(r2_train))
    print("\n")
    print('model performence for test set')
    print('------------------------------------------')
    print('RMSE IS{}'.format(rmse_test))
    print('Rc is{}'.format(r2_test))

create_polynominal_regression(3)
    
    
         
