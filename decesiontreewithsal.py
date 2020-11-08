# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:27:35 2019

@author: neeratig
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('position_salaries.csv')


X=dataset.iloc[:, 1].values
X=np.reshape(X,(-1,1))
y=dataset.iloc[:, 2].values
y=np.reshape(y,(-1,1))


"""from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
"""

#fitting decesion tree to dataset
from sklearn.tree import DecisionTreeClassifier
regress=DecisionTreeClassifier(random_state=0,criterion='entropy')
regress.fit(X,y)

y_predict=regress.predict(np.array(6.5).reshape(-1,1))

#visualization
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,regress.predict(X_grid),color='blue')
plt.title('Decesion  tree for positional salaries')
plt.xlabel('position')
plt.ylabel('salaries')
plt.show()

#printing decesion tree
from sklearn import tree
import pydot
from sklearn.externals.six import StringIO
dot_data=StringIO()
tree.export_graphviz(regress,out_file=dot_data)
graph=pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf("salaries1.pdf")