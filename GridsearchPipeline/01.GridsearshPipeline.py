#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Código básico de utilização do pipeline e gridsearh
'''
# import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# =============================================================================
# # 1.0 Base de dados
# =============================================================================
iris = datasets.load_iris()
X,y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    shuffle=True, 
                                                    random_state=1,
                                                    stratify=y)

# =============================================================================
# # 2.0 pipeline
# =============================================================================
pipe = Pipeline([ 
        ('Normalizador', StandardScaler()),
        ('knnClassif', KNeighborsClassifier(n_neighbors=2)),   # Classificador
        ],verbose=False)

# =============================================================================
# #  3.0 Ajuste comm Grid search
# =============================================================================
n=20
parametros = {
    'knnClassif__n_neighbors': list(range(1,n+1))
              }
grid = GridSearchCV(pipe, cv=2, n_jobs=1, param_grid = parametros, scoring='accuracy')

# 3.1 pipe.fit - treino do pipe
grid.fit(X_train, y_train)
grid.cv_results_
print('Resultados grid treino:\n',grid.cv_results_['mean_test_score'].reshape(-1,1))
print('\n',grid.best_params_ )
clf = grid.best_estimator_

# =============================================================================
# # 4.0 Predição
# =============================================================================
y_test_pred = grid.predict(X_test)
print('\nResultados previsão:')
print('y_predict:',y_test_pred,'\n','accuracy_score',accuracy_score(y_test, y_test_pred))
print(clf)