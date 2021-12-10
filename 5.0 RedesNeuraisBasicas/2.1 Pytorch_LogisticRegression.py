# -*- coding: utf-8 -*-
"""
   Utilizando o Pytorch:  
   A rede Regressão Logística utiliza:
       * 'sigmoid' como função de ativação
       * 'binary_crossentropy' como loss 
       * 'accuracy' como métrica de acerto
       * 'sgd' ou outro como otimizador
       """

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#%% ****** 1.0 Preparando o dataset ******

data = np.genfromtxt('https://raw.githubusercontent.com/JonathaCosta/Consulta-Deep-Learning/master/L07_logistic/code/data/toydata.txt', delimiter='\t')
X = data[:, :2].astype(np.float32)
y = data[:, 2].astype(np.int64)

# Separa subsets - Cross validation
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, 
                                                    test_size=0.25, 
                                                    random_state = 123)
# Normaliza os datasets
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


fig, ax = plt.subplots(1, 2, figsize=(7, 2.5))
ax[0].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1])
ax[0].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1])
ax[1].scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1])
ax[1].scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1])
plt.xlim([X[:, 0].min()-0.5, X[:, 0].max()+0.5])
plt.ylim([X[:, 1].min()-0.5, X[:, 1].max()+0.5])
plt.show()


#%% ****** 2.0 NN Logisct Regression ******

class LogisticRegression(torch.nn.Module):
   
    def __init__(self, num_features):
        ''' Método de iniciação do módulo NN do pytorch, função linear ,
        pesos e bias'''
        
        super().__init__()
        self.linear = torch.nn.Linear(num_features, 1)
        self.num_features = num_features

        self.linear.weight.detach().zero_()
        self.linear.bias.detach().zero_()

    def forward(self, x):
        ''' Método para processar a saida da camada 
        e aplicar função de ativação logistica '''
        
        logits = self.linear(x)
        probas = torch.sigmoid(logits)
        
        return probas
    
    
    def comp_accuracy(self,label_var, pred_probas):
        ''' Método para avaliação da acurácia '''
        
        pred_labels = torch.where((pred_probas > 0.5), 
                                  torch.tensor([1]), 
                                  torch.tensor([0])).view(-1)
        acc = torch.sum(pred_labels == label_var.view(-1)).float() / label_var.size(0)
        return acc


# 2.1 Instancia o modelo

model = LogisticRegression(num_features=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
#%% ****** 3.0 Treino e ajuste ******

num_epochs = 30
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)


for epoch in range(num_epochs):   
  
    out = model(X_train_tensor)                                           #### Compute outputs ####  
    cost = F.binary_cross_entropy(out, y_train_tensor, reduction='sum')   #### Compute gradients ####
    optimizer.zero_grad()
    cost.backward()    
    optimizer.step()                                                      #### Update weights ####  
    pred_probas = model(X_train_tensor)                                   #### Logging #### 
    acc = model.comp_accuracy(y_train_tensor, pred_probas)                #### Calculate accuracy      
    print('Epoch: %03d' % (epoch + 1), end="")
    print(' | Train ACC: %.3f' % acc, end="")
    print(' | Cost: %.3f' % F.binary_cross_entropy(pred_probas, y_train_tensor))
      
print('\nModel parameters:')
print('  Weights: %s' % model.linear.weight)
print('  Bias: %s' % model.linear.bias)



#%% ****** 4.0 Teste ******

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
pred_probas = model(X_test_tensor)
test_acc = model.comp_accuracy(y_test_tensor, pred_probas)
print('Test set accuracy: %.2f%%' % (test_acc*100))

#%% ****** 5.0 Graphic answer ******
# Code from Dalcimar apud Sebastian Raschka.
