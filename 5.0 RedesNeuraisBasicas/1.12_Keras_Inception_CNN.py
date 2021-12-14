#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:06:29 2021
@author: Rede Neural básica utilizando o tensorflow
COMENTÁRIOS:
        
    1) Rede VGG16 - 
        
        
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
#%% =============================================================================
# 1.0 Preprocessing
# =============================================================================
(X_train, y_train), (X_test, y_test) = cifar100.load_data()


#%% =============================================================================
# 2.0 Learning
# =============================================================================
def inception_module(X, filters):
        
      # Retrieve Filters
      F1, F2, F3, F4, F5, F6 = filters
    
      conv_1x1 = tf.keras.layers.Conv2D(filters=F1, kernel_size=(1,1), padding='same', activation='relu')(X)
      
      conv_3x3 = tf.keras.layers.Conv2D(filters=F2, kernel_size=(1,1), padding='same', activation='relu')(X)
      conv_3x3 = tf.keras.layers.Conv2D(filters=F3, kernel_size=(3,3), padding='same', activation='relu')(conv_3x3)
    
      conv_5x5 = tf.keras.layers.Conv2D(filters=F4, kernel_size=(1,1), padding='same', activation='relu')(X)
      conv_5x5 = tf.keras.layers.Conv2D(filters=F5, kernel_size=(5,5), padding='same', activation='relu')(conv_5x5)
    
      pool_proj = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(1,1), padding='same')(X)
      pool_proj = tf.keras.layers.Conv2D(filters=F6, kernel_size=(1,1), padding='same', activation='relu')(pool_proj)
    
      X = tf.keras.layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3)
      
      return X


#%% =============================================================================
# 2.1 Rede Neural - arquitetura da rede - CNN
# =============================================================================
# Veja os comentários!
X_input = tf.keras.layers.Input((32,32,3))
X = tf.keras.layers.experimental.preprocessing.Resizing(224,224)(X_input)

# Steam
X = tf.keras.layers.Conv2D(filters=64, kernel_size=(7,7), padding='same', strides=(2,2), activation='relu')(X)
X = tf.keras.layers.MaxPool2D(pool_size=(3,3), padding='same', strides=(2,2))(X)
#X = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1), padding='same', strides=(1,1), activation='relu')(X)
X = tf.keras.layers.Conv2D(filters=192, kernel_size=(3,3), padding='same', strides=(1,1), activation='relu')(X)
X = tf.keras.layers.MaxPool2D(pool_size=(3,3), padding='same', strides=(2,2))(X)

# Inception x2
X = inception_module(X, [64,96,128,16,32,32])
X = inception_module(X, [128,128,192,32,96,64])

# Maxpool
X = tf.keras.layers.MaxPool2D(pool_size=(3,3), padding='same', strides=(2,2))(X)

# Inception x1
X = inception_module(X, [192,96,208,16,48,64])

# Tower 1
X1 = tf.keras.layers.AveragePooling2D(pool_size=(5,5), strides=(3,3))(X)
x1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1,1), padding='same', activation='relu')(X1)
X1 = tf.keras.layers.Flatten()(X1)
X1 = tf.keras.layers.Dense(units=1024, activation='relu')(X1)
X1 = tf.keras.layers.Dropout(0.7)(X1)
X1 = tf.keras.layers.Dense(units=100, activation='softmax')(X1)

# Inception x3
X = inception_module(X, [160,112,224,24,64,64])
X = inception_module(X, [128,128,256,24,64,64])
X = inception_module(X, [112,144,288,32,64,64])

# Tower 2
X2 = tf.keras.layers.AveragePooling2D(pool_size=(5,5), strides=(3,3))(X)
X2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1,1), padding='same', activation='relu')(X2)
X2 = tf.keras.layers.Flatten()(X2)
X2 = tf.keras.layers.Dense(units=1024, activation='relu')(X2)
X2 = tf.keras.layers.Dropout(0.7)(X2)
X2 = tf.keras.layers.Dense(units=100, activation='softmax')(X2)

# Inception x1
X = inception_module(X, [256,160,320,32,128,128])

# Maxpool
X = tf.keras.layers.MaxPool2D(pool_size=(3, 3), padding='same', strides=(2,2))(X)

# Inception x2
X = inception_module(X, [256,160,320,32,128,128])
X = inception_module(X, [384,192,384,48,128,128])

X = tf.keras.layers.GlobalAveragePooling2D()(X)

X = tf.keras.layers.Dropout(0.4)(X)

X = tf.keras.layers.Dense(units=100, activation='softmax')(X)

model = tf.keras.models.Model(inputs=X_input, outputs=[X, X1, X2], name='ResNet50')

opt = tf.keras.optimizers.Adam()
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