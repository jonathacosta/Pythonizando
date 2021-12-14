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
def identity_block(X, filters):
    
      # Retrieve Filters
      F1, F2, F3 = filters
    
      # Save the input value. You'll need this later to add back to the main path. 
      X_shortcut = X
    
      # First component of main path
      X = tf.keras.layers.Conv2D(filters=F1, kernel_size=(1,1), strides=(1,1), padding='valid')(X)
      X = tf.keras.layers.BatchNormalization()(X)
      X = tf.keras.layers.Activation('relu')(X)
    
      # Second component of main path
      X = tf.keras.layers.Conv2D(filters=F2, kernel_size=(3,3), strides=(1,1), padding='same')(X)
      X = tf.keras.layers.BatchNormalization()(X)
      X = tf.keras.layers.Activation('relu')(X)
    
      # Third component of main path
      X = tf.keras.layers.Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid')(X)
      X = tf.keras.layers.BatchNormalization()(X)
    
      # Final step: Add shortcut value to main path, and pass it through a RELU activation
      X = tf.keras.layers.Add()([X,X_shortcut])
      X = tf.keras.layers.Activation('relu')(X)
    
      return X

def convolutional_block(X, filters, s):
        
      # Retrieve Filters
      F1, F2, F3 = filters
        
      # Save the input value. You'll need this later to add back to the main path. 
      X_shortcut = X
    
      ##### MAIN PATH #####
      # First component of main path 
      X = tf.keras.layers.Conv2D(filters=F1, kernel_size=(1,1), strides=(s,s), padding='valid')(X)
      X = tf.keras.layers.BatchNormalization()(X)
      X = tf.keras.layers.Activation('relu')(X)
    
      # Second component of main path
      X = tf.keras.layers.Conv2D(filters=F2, kernel_size=(3,3), strides=(1,1), padding='same')(X)
      X = tf.keras.layers.BatchNormalization()(X)
      X = tf.keras.layers.Activation('relu')(X)
    
      # Third component of main path
      X = tf.keras.layers.Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid')(X)
      X = tf.keras.layers.BatchNormalization()(X)
    
      ##### SHORTCUT PATH ####
      X_shortcut = tf.keras.layers.Conv2D(filters=F3, kernel_size=(1,1), strides=(s,s), padding='valid')(X_shortcut)
      X_shortcut = tf.keras.layers.BatchNormalization()(X_shortcut)
    
      # Final step: Add shortcut value to main path, and pass it through a RELU activation
      X = tf.keras.layers.Add()([X,X_shortcut])
      X = tf.keras.layers.Activation('relu')(X)
    
      return X

#%% =============================================================================
# 2.1 Rede Neural - arquitetura da rede -CNN
# =============================================================================
# Veja os comentários!
X_input = tf.keras.layers.Input((32,32,3))
X = tf.keras.layers.experimental.preprocessing.Resizing(224,224)(X_input)

# Stage 1
X = tf.keras.layers.ZeroPadding2D((3,3))(X)
X = tf.keras.layers.Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding='valid')(X)
X = tf.keras.layers.BatchNormalization()(X)
X = tf.keras.layers.Activation('relu')(X)
X = tf.keras.layers.ZeroPadding2D((1,1))(X)
X = tf.keras.layers.MaxPooling2D((3,3), strides=(2,2))(X)

# Stage 2
X = convolutional_block(X, [64,64,256], 1)
X = identity_block(X, [64,64,256])
X = identity_block(X, [64,64,256])

# Stage 3
X = convolutional_block(X, [128,128,512], 2)
X = identity_block(X, [128,128,512])
X = identity_block(X, [128,128,512])
X = identity_block(X, [128,128,512])

# Stage 4 
X = convolutional_block(X, [256,256,1024], 2)
X = identity_block(X, [256,256,1024])
X = identity_block(X, [256,256,1024])
X = identity_block(X, [256,256,1024])
X = identity_block(X, [256,256,1024])
X = identity_block(X, [256,256,1024])

# Stage 5 
X = convolutional_block(X, [512,512,2048], 2)
X = identity_block(X, [512,512,2048])
X = identity_block(X, [512,512,2048])

# AVGPOOL
X = tf.keras.layers.GlobalAveragePooling2D()(X)

# output layer
X = tf.keras.layers.Flatten()(X)
X = tf.keras.layers.Dense(100, activation='softmax')(X)

# Create model
model = tf.keras.models.Model(inputs=X_input, outputs=X, name='ResNet50')

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