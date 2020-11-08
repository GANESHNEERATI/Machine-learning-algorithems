# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:18:01 2019

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

#fitting linear regression to traing set
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

regress=LinearRegression()
regress.fit(X_train,y_train)
#predictig and model evaluation for training set
y_train_predict=regress.predict(X_train)
rmse=(np.sqrt(mean_squared_error(y_train,y_train_predict)))
r2=r2_score(y_train,y_train_predict)
print("model performence for traing set")
print("----------------------------------------------")
print("rmse is {}".format(rmse))
print("r2 score is{}".format(r2))

#predicting and model evaluation for testing set
y_test_predict=regress.predict(X_test)
rmse=(np.sqrt(mean_squared_error(y_test,y_test_predict)))
r2=r2_score(y_test,y_test_predict)
print("model performence for traing set")
print("----------------------------------------------")
print("rmse is {}".format(rmse))
print("r2 score is{}".format(r2))



