# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 14:27:13 2019

@author: neeratig
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection  import train_test_split

dataset=load_iris()
X=dataset.data
y=dataset.target
y=np.reshape(y,(-1,1))
#divinding training and testing data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#fitting decesion tree model
regress=DecisionTreeClassifier(criterion='entropy')

regress.fit(X_train,y_train)
y_predict=regress.predict(X_test)


from sklearn.metrics import accuracy_score
print('accuracy score on train data',accuracy_score(y_true=y_train,y_pred=regress.predict(X_train)))
print('accuracy score on test data',accuracy_score(y_true=y_test,y_pred=y_predict))

#imporve the accuracy by tuning 


regress=DecisionTreeClassifier(criterion='entropy',min_samples_split=50)
regress.fit(X_train,y_train)
print('accuracy score on train data',accuracy_score(y_true=y_train,y_pred=regress.predict(X_train)))
print('accuracy score on test data',accuracy_score(y_true=y_test,y_pred=y_predict))

"""X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regress.predict(X_grid),color='blue')
plt.show()"""

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO  
import pydot 



dot_data = StringIO() 
tree.export_graphviz(regress, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 

graph[0].write_pdf("iris.pdf")  # must access graph's first element

