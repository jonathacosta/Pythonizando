#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:06:29 2021
@author: Rede Neural básica utilizando o tensorflow
COMENTÁRIOS:
   1) Regularization as method to previne overfitting
    A rede utiliza
     * 'linear' como função de ativação
     * 'mse' como loss 
     * 'mse' como métrica de acerto
     * 'sgd' ou outro como otimizador
     Regulariza kernel e bias com l1 / l2
         * pesos = kernel_regularizer = tf.keras.regularizers.l2(0.01),
                 deve ser ajustado em função do tamanho do minibatch
         * bias = bias_regularizer = tf.keras.regularizers.l2(0.01),)
         
   
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score,confusion_matrix   # Just for classificator

#%% =============================================================================
# 1.0 Preprocessing
# =============================================================================
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()
X_train.shape



#%% =============================================================================
# 2.0 Learning
# =============================================================================

# Separa subsets - Cross validation

# Normaliza os datasets
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Resultados da normalização - média ~ 0 e desvio ~ 1
print('X_train mean', np.mean(X_train))
print('X_train standard deviation', np.std(X_train))

print('X_train.shape', X_train.shape)
print('y_train.shape', y_train.shape)
print('X_test.shape', X_test.shape)
print('y_test.shape', y_test.shape)

#%% =============================================================================
# 2.1 Rede Neural - arquitetura da rede
# =============================================================================
# Veja os comentários!
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(min(X_train.shape))))                 # Colunas de X
model.add(tf.keras.layers.Dense(1,             
                                kernel_regularizer = tf.keras.regularizers.l2(0.01),
                                bias_regularizer = tf.keras.regularizers.l2(0.01),)
                                )   
model.add(tf.keras.layers.Activation('linear'))                      # Função ativação do Adaline
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
y_pred = model.predict(X_test)
# print(y_pred)

#%
# 4.3 Pesos
print()
print('Pesos:')
print(model.get_weights()[0])
print(model.get_weights()[1])


#%% 4.3 Classification multclass

# def to_class(x):
#     y_class=[]
#     for i in x:
#         y_class.append(np.argmax(i))
#     return np.array(y_class)

# y_pred = to_class(y_pred)
# y_test = to_class(y_test)
# print(y_pred)
# print(y_test)

#%%
# 4.4 Accuracy_score and confusion_matrix
# print()

# print('Accuracy_score',accuracy_score(y_test, y_pred))
# print('Confusion_matrix\n',confusion_matrix(y_pred, y_test))

