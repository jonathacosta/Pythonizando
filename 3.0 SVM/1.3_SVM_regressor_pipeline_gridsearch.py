#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 14:54:27 2021
@author: Jonatha Costa
Código básico utilizando o Support Vector Machine como regressor
"""

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
import numpy as np

#% ****** 1.0 Preprocessing ******
''' Basic data for initial tests. '''
X=np.arange(0,110,10).reshape(-1,1)
y=np.array([ 0.94, 0.96, 1.0, 1.05, 1.07, 1.09, 1.14, 1.17, 1.21, 1.24, 1.28]).reshape(-1,1)


#% ****** 2.0 Learning ******
# Split subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True,
                                                    test_size=0.3, 
                                                    random_state=123)
# 2.1 Model flow using pipeline

model = Pipeline([ 
        ('Normalizador', MinMaxScaler()),                       # Normalizador  
        ('svr',SVR(kernel='rbf'))
        ],verbose=False,)

# 2.2 Learning
model.fit(X_train.reshape(-1,1),y_train.ravel())

#% ****** 3.0 Evaluation ******
def eval(X,y,model,dataset): 
    '''Método calcula os erros de um dataset '''    
    y_pred = model.predict(X).reshape(-1,1)
    print(f'\nEVALUATION - {dataset} date_set:')
    print('Mean Square Error (MSE):',mean_squared_error(y, y_pred) )
    print('Root Mean Squared Erro (RMSE):',mean_squared_error(y, y_pred)**0.5 )
    print('Mean Absolute Error(MAE):', mean_absolute_error(y, y_pred))
    print('R-squared Error:',r2_score(y, y_pred))
    
print(model.get_params())
eval(X_train,y_train,model,'train')
# eval(X_test,y_test,model,'test')

#% ****** 4.0 Prediction ******
y_pred = model.predict(X_test)
print(f'Valor predito para próximo período:{y_pred[-1]}')
print()
#% ****** 5.0 Graphics results ******

def graf(X_test,y_test,y_pred):
    import matplotlib.pyplot as plt
    plt.plot(X_test,y_test,'b',label='Refers')
    plt.plot(X_test,y_pred,'r',label = 'Prediction')
    plt.legend()
    plt.grid()
    plt.show()
    
y_pred = model.predict(X_test).reshape(-1,1)
graf(X_test,y_test,y_pred)    

#%% ****** 6.0 Parameters optimization hyperparams ******
# Ajuste com Grid search
parametros = {'svr__kernel': ('rbf','linear','poly','sigmoid',),
              'svr__C':list(np.arange(0.5,10,0.5)),
              'svr__gamma': [1e-7, 1e-4],
              'svr__epsilon':[0.1,0.2,0.5,0.3],
              }
grid = GridSearchCV(model, cv=2, param_grid = parametros,verbose=True)
grid.fit(X_train,y_train.ravel())

# Evaluation
print()
print(grid.best_estimator_)
eval(X_train,y_train,grid,'train')
eval(X_test,y_test,grid,'test')

# Previsão 
y_pred = grid.predict(X_test)
print(f'\nValor predito para próximo período:{y_pred[-1]}')
print()

# Gráfico   
y_pred = grid.predict(X_test).reshape(-1,1)
graf(X_test,y_test,y_pred)  
print('\nResultados grid treino:\n',grid.cv_results_['mean_test_score'].ravel())
print('\nGrid.best_params',grid.best_params_ )
print('\nGrid_best_estimador',grid.best_estimator_)
