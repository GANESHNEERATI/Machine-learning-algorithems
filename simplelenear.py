# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:03:02 2019

@author: neeratig
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Salary_Data.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

#spilting data into training set and testting set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

#fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression

regress=LinearRegression()
regress.fit(X_train,y_train)

#visualizing training set
viz_train=plt
viz_train.scatter(X_train,y_train,c='red')
viz_train.plot(X_train,regress.predict(X_train),color='blue')
viz_train.title('sal vs experience(training set)')
viz_train.xlabel('year of experience')
viz_train.ylabel('salary')
viz_train.show()
#visualizing test set
viz_test = plt
viz_test.scatter(X_test, y_test, color='red')
viz_test.plot(X_train,regress.predict(X_train),color='blue')
viz_test.title('Salary VS Experience (Test set)')
viz_test.xlabel('Year of Experience')
viz_test.ylabel('Salary')
viz_test.show()
#predicting salary

y_predict=regress.predict(X_test)


#predicting single value
sal_pre=np.array(20).reshape(-1,1)
regress.predict(sal_pre)