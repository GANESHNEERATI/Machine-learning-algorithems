# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:58:58 2019

@author: NEERATI GANESH
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importing dataset
dataset=pd.read_csv('diabetes2.csv')
dataset.head()
dataset.info()
dataset.describe()

sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.countplot(x='Outcome',data=dataset)
sns.distplot(dataset['Age'].dropna(),kde=True)
dataset.corr()
sns.heatmap(dataset.corr())
sns.pairplot(dataset)

plt.subplot(figsize=(20,15))
sns.boxplot(x='Age',y='BMI',data=dataset)

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,8].values
y=np.reshape(y,(-1,1))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


from sklearn.linear_model import LogisticRegression
regress=LogisticRegression()
regress.fit(X_train,y_train)
y_predict=regress.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_predict))

confusion_matrix(y_test,y_predict)