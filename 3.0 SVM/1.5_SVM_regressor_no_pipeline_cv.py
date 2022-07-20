#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 14:54:27 2021
@author: Jonatha Costa
Código básico utilizando o Support Vector Machine como regressor
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import numpy as np
from rotinas_auxiliares import eval_jrc,graf
from sklearn.model_selection import cross_val_score,cross_val_predict


#%% ****** 1.0 Preprocessing ******
''' Basic data for initial tests. '''
X=np.arange(0,110,10).reshape(-1,1)
y=np.array([ 0.94, 0.96, 1.0, 1.05, 1.07, 1.09, 1.14, 1.17, 1.21, 1.24, 1.28]).reshape(-1,1)


#%% ****** 2.0 Learning ******
# Separa subsets - Cross validation
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, 
                                                    test_size=0.3, 
                                                    random_state = 123)
# Normaliza os datasets
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Modelo
model =  SVR(kernel='rbf')
# 2.2 Learning
model.fit(X_train.reshape(-1,1),y_train.ravel())

# 3.0 Train
per=''
y_pred = model.predict(X_train).reshape(-1,1)
graf(X_train,y_train,y_pred,scaler,f'Train set - período: {per}')        
print('Resultados com hiperparâmetros default do estimador!')
print(eval_jrc(X_train,y_train,model,'train'))



# 4.0 Test
y_pred = model.predict(X_test).reshape(-1,1)
graf(X_test,y_test,y_pred,scaler,f'Test set - período: {per}') 
print('Resultados com hiperparâmetros default do estimador!')
print(eval_jrc(X_test,y_test,model,'test'))



# =============================================================================
# 
#%% =============================================================================

r1 = cross_val_score(model, X_train, y_train.ravel(), 
                         cv=2,
                         # scoring='neg_root_mean_squared_error',
                        )

r2 = cross_val_predict(model, X_train, y_train.ravel(), 
                         cv=2,
                         # scoring='neg_root_mean_squared_error',
                        )

