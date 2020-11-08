# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 12:13:14 2019

@author: neeratig
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#import dataset
dataset=pd.read_csv('position_salaries.csv')
X=dataset.iloc[:, 1:2].values
y=dataset.iloc[:,2].values
y=np.reshape(y,(-1,1))
##featurew scaling
from sklearn.preprocessing import StandardScaler
Sc_x=StandardScaler()
Sc_y=StandardScaler()
X=Sc_x.fit_transform(X)
y=Sc_y.fit_transform(y)

#fitting SVR
from sklearn.svm import SVR
regress=SVR(kernel='rbf')

regress.fit(X,y)

#visualize 
plt.scatter(X, y, color='red')
plt.plot(X, regress.predict(X), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

#printing more precisely
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,regress.predict(X_grid),color='blue')
plt.title('truth or bluf')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

#predicting salary
y_pred = Sc_y.inverse_transform(
	regress.predict(
		Sc_x.transform(
			np.array([[5]])
		)
	)
)





