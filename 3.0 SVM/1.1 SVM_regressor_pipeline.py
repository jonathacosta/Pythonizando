#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 14:54:27 2021
@author: Jonatha Costa
Código básico utilizando o Support Vector Machine como regressor
"""

# from sklearn.model_selection import GridSearchCV
# from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
import numpy as np

#%% ****** 1.0 Preprocessing ******
''' Basic data for initial tests. '''
X=np.arange(0,110,10).reshape(-1,1)
y=np.array([ 0.94, 0.96, 1.0, 1.05, 1.07, 1.09, 1.14, 1.17, 1.21, 1.24, 1.28])


#%% ****** 2.0 Learning ******
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True,
                                                    test_size=0.3, 
                                                    random_state=123)
# 2.1 Model flow using pipeline
model = make_pipeline(MinMaxScaler(), SVR(kernel='linear'),verbose=True)
# 2.2 Learning
model.fit(X_train.reshape(-1,1),y_train.ravel())

#%% ****** 3.0 Evaluation ******
y_pred = model.predict(X_test)
print('Mean Square Error (MSE):',mean_squared_error(y_test, y_pred) )
print('Root Mean Squared Erro (RMSE):',mean_squared_error(y_test, y_pred)*0.5 )
print('Mean Absolute Percent Error(MAPE):', 
      round( np.mean( abs( y_test - y_pred/y_test))*100,4),'%')

#%% ****** 4.0 Prediction ******
y_pred = model.predict(X_test)
print(f'Valor predito para próximo período:{y_pred}')
#%% ****** 5.0 Graphics results ******
import matplotlib.pyplot as plt

plt.plot(X_test,y_test,'b',label='Refers')
plt.plot(X_test,y_pred,'r',label = 'Prediction')
plt.legend()
plt.grid()
plt.show()