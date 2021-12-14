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
       
  3) A rede Regressão Linear utiliza:tach().zero_()
        self.linear.bias.detach().zero_()
        # Note: the trailing underscore
        # means "in-place operation" in the context
        # of PyTorch
        
    def forward(self, x):
        * 'linear' como função de ativação
        * 'mse' como loss 
        * 'mse' como métrica de acerto
        * 'sgd' ou outro como otimizador   
        Tem como saída valores escalares preditos em lugar de classes.
        (4.3 Binary classification),(4.4 Accuracy_score and confusion_matrix)
        São etapas exclusivas de classificadores. Não de regressores!
   4) A rede com multiclasses - softmax deve utilizar
        * 'sigmoid' como função de ativação
        * 'categorical_crossentropy' como loss 
        * 'categorical_accuracy'' como métrica de acerto
        * 'sgd' ou outro como otimizador
         
    
   
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix   # Just for classificator

#%% =============================================================================
# 1.0 Preprocessing
# =============================================================================
from sklearn import datasets
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Conversão do multiclass para 0,1 
y = tf.keras.utils.to_categorical(y)


#%% =============================================================================
# 2.0 Learning
# =============================================================================

# Separa subsets - Cross validation
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, 
                                                    test_size=0.4,
                                                    stratify=None,
                                                    random_state = 123)
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
model.add(tf.keras.Input(shape=(min(X.shape))))                   # Colunas de X
model.add(tf.keras.layers.Dense(3, kernel_initializer='zeros'))   # Quantidade de saídas dos neurônios.
model.add(tf.keras.layers.Activation('sigmoid'))                  # Função ativação do Adaline
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['categorical_accuracy']) 
model.summary()


#%% =============================================================================
# 3.0 Evaluation: fitness, loss e accuracy
# =============================================================================
# Ajuste da rede
history = model.fit(X_train, y_train, epochs=100, batch_size=5, verbose=1)

#%% =============================================================================
# 4.0 Prediction: teste
# =============================================================================
# 4.1 Loss and accuracy
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print('\nTest_loss',test_loss,'\nTest_accuracy',test_accuracy)

# 4.2 Prediction
y_pred = model.predict(X_test)
# print(y_pred)

#%% 4.3 Classification multclass

def to_class(x):
    y_class=[]
    for i in x:
        y_class.append(np.argmax(i))
    return np.array(y_class)

y_pred = to_class(y_pred)
y_test = to_class(y_test)
print(y_pred)
print(y_test)

#%%
# 4.4 Accuracy_score and confusion_matrix
# print()

print('Accuracy_score',accuracy_score(y_test, y_pred))
print('Confusion_matrix\n',confusion_matrix(y_pred, y_test))


# 4.5 Pesos
print()
print('Pesos:')
print(model.get_weights()[0])
print(model.get_weights()[1])

#%% 5.0 Graphic results
import matplotlib.pyplot as plt

plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['categorical_accuracy'],label='categorical_accuracy')
plt.title('model accuracy')
plt.ylabel('accuracy/loss')
plt.xlabel('epoch')
plt.legend(loc='upper left')
plt.show()