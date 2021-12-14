#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:06:29 2021
@author: Rede Neural básica utilizando o tensorflow
COMENTÁRIOS:
        
    1) Rede com:
        *  Regularização dropout
            > model.add(tf.keras.layers.Dropout(0.2))                               
        * Normalização das entradas interlayers
            > model.add(tf.keras.layers.BatchNormalization())                       
        * Normalização dos pesos interlayers
            > model.add(tf.keras.layers.Dense(units=128, activation='relu', kernel_initializer='GlorotNormal'))
        * A rede utiliza:
           * 'softmax' como função de ativação
           * 'sparse_categorical_crossentropy' como loss 
           * 'sparse_categorical_accuracy' como métrica de acerto
           * 'adam' ou outro como otimizador
        
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
model.add(tf.keras.Input(shape=(min(X_train.shape))))                 # Camada de entrada
  
model.add(tf.keras.layers.Dense(units=128, activation='relu',         # Neurônios da camada intermediária
                                # kernel_initializer='GlorotNormal'     # Normalização das pesos
                                ))
model.add(tf.keras.layers.Dropout(0.2))                               # Dropout
model.add(tf.keras.layers.BatchNormalization())                       # Batch normalization entradas das camadas intermediarias
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))      # Camada de saída                  
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy']) 
model.summary()


#%% =============================================================================
# 3.0 Evaluation: fitness, loss e accuracy
# =============================================================================
# Ajuste da rede
history = model.fit(X_train, y_train, batch_size=10, validation_split=0.1, epochs=10)


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