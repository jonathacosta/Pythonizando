#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 14:54:27 2021
@author: Jonatha Costa
Código básico utilizando o Support Vector Machine como regressor
"""

# from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
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
def set_params(max_depth, X_test,y_test):
    model = make_pipeline(MinMaxScaler(),
                         DecisionTreeRegressor(max_depth = max_depth ,random_state=1),
                         verbose=False)
    model.fit(X_train.reshape(-1,1),y_train.ravel())
    y_pred = model.predict(X_test)
    acc = model.score(X_test,y_test)

    return y_pred, acc

#%% ****** 3.0 Evaluation ******

#%% ****** 4.0 Prediction ******

#%% ****** 5.0 Graphics results ******
import matplotlib.pyplot as plt
lst = np.arange(1,10)  # max_depth

plt.figure(figsize=(10,6))
plt.plot(X_test,y_test,'b',label='Refers')
mse=1e3

for i in lst:        
    y_pred, acc = set_params(i,X_test,y_test)
    
    # Critério de parada    
    if mean_squared_error(y_test, y_pred) < mse:
        mse = mean_squared_error(y_test, y_pred)    
    else:
        print(f'Resultados convergentes a partir da iteração {i} de {max(lst)}.')
        break   
    # plots
    plt.plot(X_test,y_pred,label = f'{i}') 
    print(f'\nMax_depth {i}:')
    print('Accuracy score(ACC):',acc)
    print('Mean Square Error (MSE):',mean_squared_error(y_test, y_pred) )
    print('Root Mean Squared Erro (RMSE):',mean_squared_error(y_test, y_pred)*0.5 )
    print('Mean Absolute Percent Error(MAPE):', 
          round( np.mean( abs( y_test - y_pred/y_test))*100,4),'%')
print('\nResults without cross-validation.')    
plt.title('Tree Decision e resultados com diferentes parametros para DeptMax')
plt.legend()
plt.grid()
plt.show()