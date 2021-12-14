#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:06:29 2021
@author: Rede Neural básica utilizando o tensorflow
COMENTÁRIOS:
        
    1) Rede com LSTM RNN 
     * Análise de sentimentos
        
        
"""
import tensorflow as tf
from tensorflow.keras.datasets import imdb
#%% =============================================================================
# 1.0 Preprocessing
# =============================================================================
number_of_words = 20000
max_len = 100
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=number_of_words) # Dataset já está codificado.


#%% =============================================================================
# 2.0 Learning
# =============================================================================

# Padding  - uniformizar as palavras do dicionário
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)
print(X_train)
print(X_test)
#%% =============================================================================
# 2.1 Rede Neural - arquitetura da rede -RNN
# =============================================================================
# Veja os comentários!
X_input = tf.keras.layers.Input((100))

X = tf.keras.layers.Embedding(input_dim=number_of_words, output_dim=128)(X_input)
X = tf.keras.layers.LSTM(units=128, activation='tanh')(X) #return_sequences=True  - Input é vetor 3d
X = tf.keras.layers.Dense(units=1, activation='sigmoid')(X)

model = tf.keras.models.Model(inputs=X_input, outputs=X)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

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

plt.plot(history.history['val_accuracy'],label = 'val_accuracy')
plt.plot(history.history['accuracy'],label = 'accuracy' )
plt.plot(history.history['loss'],label='loss')
plt.title('model accuracy trainning')
plt.ylabel('accuracy/loss/val_accuracy')
plt.xlabel('epoch')
plt.legend(loc='upper left')
plt.show()