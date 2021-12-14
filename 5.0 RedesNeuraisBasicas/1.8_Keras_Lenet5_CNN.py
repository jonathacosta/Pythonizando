#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:06:29 2021
@author: Rede Neural básica utilizando o tensorflow
COMENTÁRIOS:
        
    1) Rede Lenet5 - CNN de referência.
        * Input
        * Convulução
        * Pooling
        * Convolução
        * Pooling
        * Layers full connected - dense layers
        
        
"""
import numpy as np
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

# Reshaping - transformação de dimensão 4d para keras - batchsize, rows, cols, channel
X_train = np.reshape(X_train,[60000,28,28,1])
X_test = np.reshape(X_test,[10000,28,28,1])

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
model.add(tf.keras.Input(shape=[28,28,1], ))                # Camada de entrada. Batch-size entra na treino!
model.add(tf.keras.layers.Conv2D(filters=6, 
                                 kernel_size=(5,5), 
                                 padding='same', 
                                 activation='tanh'
                                 ))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=2))
model.add(tf.keras.layers.Conv2D(filters=16, 
                                 kernel_size=(5,5), 
                                 strides=1, 
                                 padding='valid',   # saída reduzida para 10x10 
                                 activation='tanh'
                                 ))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=2))
model.add(tf.keras.layers.Flatten()) # Concatenar saídas num vetor
model.add(tf.keras.layers.Dense(units=120, activation='tanh'))
model.add(tf.keras.layers.Dense(units=84, activation='tanh'))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])


model.summary()


#%% =============================================================================
# 3.0 Evaluation: fitness, loss e accuracy
# =============================================================================
# Ajuste da rede
history = model.fit(X_train, y_train, batch_size=100, validation_split=0.1, epochs=5)


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