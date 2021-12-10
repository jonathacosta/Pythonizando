#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:06:29 2021
@author: Rede Neural básica utilizando o tensorflow
COMENTÁRIOS:
   1) A rede Adaline utiliza:
       * 'linear' como função de ativação
       * 'mse' como loss 
       * 'accuracy' como métrica de acerto
       * 'sgd' ou outro como otimizador
   
   2) A rede Regressão Logística utiliza:
       * 'sigmoid' como função de ativação
       * 'binary_crossentropy' como loss 
       * 'accuracy' como métrica de acerto
       * 'sgd' ou outro como otimizador
       
   3) A rede Regressão Linear utiliza:
         * 'linear' como função de ativação
         * 'mse' como loss 
         * 'mse' como métrica de acerto
         * 'sgd' ou outro como otimizador   
         Tem como saída valores escalares preditos em lugar de classes.
         (4.3 Binary classification),(4.4 Accuracy_score and confusion_matrix)
         São etapas exclusivas de classificadores. Não de regressores!
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error                  # just for regressor
from sklearn.metrics import accuracy_score,confusion_matrix   # Just for classificator

#%% =============================================================================
# 1.0 Preprocessing
# =============================================================================

data = np.genfromtxt('https://raw.githubusercontent.com/JonathaCosta/Consulta-Deep-Learning/master/L05_grad-descent/code/datasets/linreg-data.csv',delimiter=',')
X, y = data[1: , 1:3], data[1:, 3]
print('X.shape:', X.shape)
print('y.shape:', y.shape)

#%% =============================================================================
# 2.0 Learning
# =============================================================================

# Separa subsets - Cross validation
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, 
                                                    test_size=0.4, 
                                                    random_state = 123)

# Normaliza os datasets
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#%% =============================================================================
# 2.1 Rede Neural - arquitetura da rede
# =============================================================================
# Veja os comentários!
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(2,)))                            # Colunas de X
model.add(tf.keras.layers.Dense(1, kernel_initializer='zeros'))  # Camada oculta
model.add(tf.keras.layers.Activation('linear'))                  # Função ativação do Adaline
model.compile(loss='mse', optimizer='sgd', metrics=['mse']) 
model.summary()

#%% =============================================================================
# 3.0 Evaluation: fitness, loss e accuracy
# =============================================================================
# Ajuste da rede
model.fit(X_train, y_train, epochs=10, batch_size=5, verbose=1)

#%% =============================================================================
# 4.0 Prediction: teste
# =============================================================================
# 4.1 Loss and accuracy
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print('\nTest_loss',test_loss,'\nTest_accuracy',test_accuracy)

# 4.2 Prediction
y_pred = model.predict(X_test).ravel()
# print(y_pred.reshape(1,-1))

# =============================================================================
# #Etapa exclusiva para classficador!!!
# =============================================================================
# 4.3 Binary classification -  
# y_pred = np.where(y_pred > 0.5, 1, 0)
# print(y_pred)
# print(y_test)


# # 4.4 Accuracy_score and confusion_matrix
# print()
# print('Accuracy_score',accuracy_score(y_test, y_pred))
# print('Confusion_matrix\n',confusion_matrix(y_pred, y_test))


'''
COMENTÁRIOS:
    * O plot do X deve considerar a quantidade de colunas que o compõe.
    * O predict fora o X deve considerar um input estimado para X, como um novo
    X_test. 
    Nesse caso vale avaliar se deve-se treinar novamente o modelo com todo o dataset disponível.
'''