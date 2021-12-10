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
# # 2.0 Sem pipeline
# =============================================================================
scaler = StandardScaler().fit(X_train)      # Define parâmetros do normalizador a partir da base de treino
X_train = scaler.transform(X_train)         # Normaliza X_train
X_test = scaler.transform(X_test) 
knn = KNeighborsClassifier(n_neighbors=10)  # Classificador com n vizinhos

# =============================================================================
# #  3.0 Ajuste search
# =============================================================================

# 3.1 estimador.fit - treino do estimador
knn.fit(X_train, y_train)

# 3.2 Validação do conjunto de treino
score = cross_val_score(knn, X_train, y_train, cv=5,verbose=0)
print('Validação k-fold:\n',score)

# =============================================================================
# # 4.0 Predição
# =============================================================================
y_test_pred = knn.predict(X_test)
print('\ny_predict:',y_test_pred,'\n' ,'accuracy_score',accuracy_score(y_test, y_test_pred))
