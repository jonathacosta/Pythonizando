#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Código básico de utilização do pipeline e gridsearh
'''
# import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

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
pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=9))

# =============================================================================
# #  3.0 Ajuste search e validação
# =============================================================================
# 3.1 pipe.fit - treino do pipe
pipe.fit(X_train, y_train)
# 3.2 Validação do conjunto de treino
score = cross_val_score(pipe, X_train, y_train, cv=5,verbose=0)
print('Validação k-fold:\n',score)

# =============================================================================
# # 4.0 Predição
# =============================================================================
y_test_pred = pipe.predict(X_test)
print('\ny_predict:',y_test_pred,'\n' ,'accuracy_score',accuracy_score(y_test, y_test_pred))


# Comentários
'''
A opção Pipeline, a seguir permite aplicar rótulos aos métodos para indexá-los
quando na utilização do gridsearh

pipe = Pipeline([ 
        ('Normalizador', StandardScaler()),
        ('knnClassif', KNeighborsClassifier(n_neighbors=5)),   # Classificador
        ],verbose=False)
'''

