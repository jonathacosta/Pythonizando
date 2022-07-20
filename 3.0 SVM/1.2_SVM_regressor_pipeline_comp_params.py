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

# 2.2 Learning
def set_params(kernel):
    model = make_pipeline(MinMaxScaler(), SVR(kernel=kernel),verbose=False)
    model.fit(X_train.reshape(-1,1),y_train.ravel())
    y_pred = model.predict(X_test)
    return y_pred

#%% ****** 3.0 Evaluation ******

#%% ****** 4.0 Prediction ******

#%% ****** 5.0 Graphics results ******
import matplotlib.pyplot as plt
lst = ['rbf', 'poly','sigmoid','linear' ]
plt.figure(figsize=(10,6))
plt.plot(X_test,y_test,'b',label='Refers')
for i in lst:    
    y_pred = set_params(i)    
    plt.plot(X_test,y_pred,label = f'{i}') 
    print(f'\nKernel {i}:')
    print('Mean Square Error (MSE):',mean_squared_error(y_test, y_pred) )
    print('Root Mean Squared Erro (RMSE):',np.sqrt(mean_squared_error(y_test, y_pred)))
    print('Mean Absolute Percent Error(MAPE):', 
          round( np.mean( abs( y_test - y_pred/y_test))*100,4),'%')
plt.title('SVM e resultados com diferentes kernels')
plt.legend()
plt.grid()
plt.show()