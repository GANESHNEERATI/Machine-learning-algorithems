# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:18:44 2019

@author: neeratig
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x=2-3*np.random.normal(0,1,20)
y=x-2*(x**2)+0.5*(x**3)+np.random.normal(-3,3,20)
#transforming data to include another axis
x=x[:, np.newaxis]
y=y[:, np.newaxis]

#ditting linear expression

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)
y_predict=model.predict(x)

plt.scatter(x,y,s=10)
plt.plot(x,y_predict,c='r')
plt.show()