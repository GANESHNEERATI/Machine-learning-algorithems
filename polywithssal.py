# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 17:18:55 2019

@author: neeratig
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('https://s3.us-west-2.amazonaws.com/public.gamelab.fun/dataset/position_salaries.csv')
X=dataset.iloc[:, 1].values
y=dataset.iloc[:, 2].values
X=X.reshape(-1,1)
y=y.reshape(-1,1)







from sklearn.model_selection import  train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)





from sklearn.linear_model import LinearRegression
regress=LinearRegression()
regress.fit(X_train,y_train)
def viz_linear():
    viz_train=plt
    viz_train.scatter(X_train,y_train,color='red')
    viz_train.plot(X_train,regress.predict(X_train),color='blue')
    viz_train.title('position vs saalry')
    viz_train.xlabel('position')
    viz_train.ylabel('salary')
    viz_train.show()
    return


viz_linear()

#polynominal regression

from sklearn.preprocessing import PolynomialFeatures

def viz_poly(degree):
    poly_reg=PolynomialFeatures(degree=degree)
    X_poly=poly_reg.fit_transform(X)
    poly_reg.fit(X_poly,y)
    regress=LinearRegression()
    regress.fit(X_poly,y)
    #visualization
    plt.scatter(X, y, color = 'red')
    plt.plot(X, p_reg.predict(poly_reg.fit_transform(X)))
    plt.show()
    
    X_grid = np.arange(min(X), max(X), 0.1)
    X_grid = X_grid.reshape(len(X_grid), 1)
    plt.scatter(X, y, color ='red')
    plt.plot(X_grid, regress.predict(poly_reg.fit_transform(X_grid)), color ='green')
    plt.title('Truth or Bluff (Polynomial Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
    
    posi=np.array(6).reshape(-1,1)
    print( regress.predict(poly_reg.fit_transform(posi)))

    return


viz_poly(4)

    



    
   
    
