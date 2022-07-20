#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 14:54:27 2021
@author: Jonatha Costa
Código básico utilizando o Support Vector Machine como regressor
"""
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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

''' Report '''
n_tops=10
def report(results, n_top=n_tops):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")


''' Ajuste com Grid search '''
parametros = {
               'svr__kernel': ('rbf','linear','sigmoid',),
              
               # 'svr__C':list(np.arange(0.5,10,0.5)),
               'svr__gamma': [1e-7, 1e-4]
              # 'svr__epsilon':[0.1,0.2,0.5,0.3],
              }

from sklearn.metrics import make_scorer
aval = make_scorer((mean_squared_error),greater_is_better=False)

grid = GridSearchCV(model,cv=2, param_grid = parametros,verbose=True, 
                    # scoring = aval 
                    )
grid.fit(X_train,y_train.ravel())

'''  Ajuste com Random search'''
param_dist = {'svr__kernel': ('rbf','linear','poly','sigmoid',),
               'svr__C':list(np.arange(0.5,10,0.5)),
              'svr__gamma': [1e-7, 1e-4],
              'svr__epsilon':[0.1,0.2,0.5,0.3],
              }
random_search = RandomizedSearchCV(model, cv=2, param_distributions = param_dist, n_iter=10)
random_search.fit(X_train,y_train.ravel())

# report(random_search.cv_results_)
report(grid.cv_results_)


# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import LinearSVC
# grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},
#                     scoring=ftwo_scorer)



#%% Resultados

print(grid.best_estimator_,
      # grid.best_params_,
      )
print(random_search.best_estimator_,
      # random_search.best_params_,
      )


#%%

import pandas as pd
pd.set_option('display.max_colwidth', 0)
res = pd.DataFrame()
res["rank"] = grid.cv_results_["rank_test_score"]
res["mean_test_score"] = grid.cv_results_["mean_test_score"]
res["std_test_score"] = grid.cv_results_["std_test_score"]
res['Kernel'] = grid.cv_results_["param_svr__kernel"]
res['Parâmetros'] = grid.cv_results_["params"]
res.sort_values(by = "rank",inplace=True)
res = res.set_index('rank')
# res.groupby(['rank','std_test_score']).describe()

print(res[:50])


