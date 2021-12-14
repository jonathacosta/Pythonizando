#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:06:29 2021
@author: Rede Neural básica utilizando o tensorflow
COMENTÁRIOS:
   1) Regularization l1/l2 as method to previne overfitting
    A rede utiliza
     * 'linear' como função de ativação
     * 'mse' como loss 
     * 'mse' como métrica de acerto
     * 'sgd' ou outro como otimizador
     Regulariza kernel e bias com l1 / l2
         * pesos = kernel_regularizer = tf.keras.regularizers.l2(0.01),
                 deve ser ajustado em função do tamanho do minibatch
         * bias = bias_regularizer = tf.keras.regularizers.l2(0.01),)
    2) Regularizion dropout as method to previne overfitting
        A rede utiliza
         * 'softmax' como função de ativação
         * 'sparse_categorical_crossentropy' como loss 
         * 'sparse_categorical_crossentropy' como métrica de acerto
         * 'sgd' ou outro como otimizador
"""

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
#%% =============================================================================
# 1.0 Preprocessing
# =============================================================================
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


#%% =============================================================================
# 2.0 Learning
# =============================================================================

# Separa subsets - Cross validation

# Normaliza os datasets -  IMAGENS
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshaping - transformação de dimensão
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

# Resultados da normalização / redimensionamente - média ~ 0 e desvio ~ 1
print('X_train.shape', X_train.shape)
print('y_train.shape', y_train.shape)
print('X_test.shape', X_test.shape)
print('y_test.shape', y_test.shape)

#%% =============================================================================
# 2.1 Rede Neural - arquitetura da rede - MLP
# =============================================================================
# Veja os comentários!
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(min(X_train.shape))))                 # Colunas de X
model.add(tf.keras.layers.Dense(128, activation = 'relu'))             
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))                        
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['sparse_categorical_accuracy']) 
model.summary()


#%% =============================================================================
# 3.0 Evaluation: fitness, loss e accuracy
# =============================================================================
# Ajuste da rede
history = model.fit(X_train, y_train, batch_size=10, validation_split=0.1, epochs=20)


#%% =============================================================================
# 4.0 Prediction: teste
# =============================================================================
# 4.1 Loss and accuracy
print('\nTrain results:')
test_loss, test_accuracy = model.evaluate(X_train, y_train, batch_size=100)
print('\nTest results:')
# test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=100)

#%% 5.0 Graphic results
import matplotlib.pyplot as plt

plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.title('model accuracy trainning')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['treino', 'validação_treino'], loc='upper left')
plt.show()