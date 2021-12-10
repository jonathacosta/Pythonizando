# -*- coding: utf-8 -*-
"""
   Utilizando o Pytorch:  
   A rede Softmax Regression utiliza:
        * 'sigmoid' como função de ativação
        * 'binary_crossentropy' como loss 
        * 'accuracy' como métrica de acerto
        * 'sgd' ou outro como otimizador
        * modelo automatizado para reconhecer dimensão de x e classes. Verifique em:
            model = SoftmaxRegression(num_features=len(X.shape), num_classes=len(np.bincount(y)))  # colunas de x e classes   
   Nota 1:
       O tensor pred_probas contém duas colunas, ativação do somatório(X.w + b) e saída.
       pred_probas[0] - saída da função logística entre 0,1. 
       pred_probas[1] - saída da função logística entre 0,1, convertida em probabilidade pela softmax
       torch.argmax(pred_probas[i]) - o argmax extrai o maior valor do pred_probas[i] e atribue a classe.
           de modo que o resultado é o mesmo se utilizar pred_probas[0] ou pred_probas[1] por essa razão! 
       """

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F

#%% ****** 1.0 Preparando o dataset ******

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:,1:4:2]  # Take columns 2 and 4.
y = iris.target


# Separa subsets - Cross validation
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, 
                                                    test_size=0.2,
                                                    stratify=None,
                                                    random_state = 123)
# Normaliza os datasets
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


fig, ax = plt.subplots(1, 2, figsize=(7, 2.5))
ax[0].scatter(X_train[y_train == 2, 0], X_train[y_train == 2, 1])
ax[0].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], marker='v')
ax[0].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], marker='s')
ax[1].scatter(X_test[y_test == 2, 0], X_test[y_test == 2, 1])
ax[1].scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], marker='v')
ax[1].scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], marker='s')
plt.show()


#%% ****** 2.0 NN Softmax Regression ******

class SoftmaxRegression(torch.nn.Module):
   
    def __init__(self, num_features,num_classes):
        ''' Método de iniciação do módulo NN do pytorch, função linear ,
        pesos e bias'''
        
        super().__init__()
        self.linear = torch.nn.Linear(num_features, num_classes)
        self.num_features = num_features

        self.linear.weight.detach().zero_()
        self.linear.bias.detach().zero_()

    def forward(self, x):
        ''' Método para processar a saida da camada 
        e aplicar função de ativação logistica '''
        
        logits = self.linear(x)
        probas = F.softmax(logits,dim=1)
        
        return logits,probas
    
    
    def comp_accuracy(self,true_labels, pred_labels):
        ''' Train set'''
        accuracy = torch.sum(true_labels.view(-1).float() == pred_labels.float()).item() / true_labels.size(0)
        return accuracy


# 2.1 Instancia o modelo de acordo com a dimensão do dataset
# Converte o subset_train de arrays para tensor
X_train = torch.tensor(X_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.int)

model = SoftmaxRegression(num_features=len(X.shape), num_classes=len(np.bincount(y)))  # colunas de x e classes   
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)                                # lr = taxa de aprendizado

#%% ****** 3.0 Treino e ajuste ******

num_epochs = 50

for epoch in range(num_epochs):   
  
    #### Compute outputs ####  
    logits, probas = model(X_train) 
                               
    #### Compute gradients ####
    cost = F.cross_entropy(logits, y_train.long())
    optimizer.zero_grad()
    cost.backward()
    
    #### Update weights ####  
    optimizer.step()
    
    #### Logging ####                                                        
    logits, probas = model(X_train)  
                                 
    #### Calculate accuracy      
    acc = model.comp_accuracy(y_train, torch.argmax(probas, dim=1))                
    print('Epoch: %03d' % (epoch + 1), end="")
    print(' | Train ACC: %.3f' % acc, end="")
    print(' | Cost: %.3f' % F.cross_entropy(logits, y_train.long()))  

print('TRAINNING RESULTS:')    
print('  Model parameters:')
print('  Weights: %s' % model.linear.weight)
print('  Bias: %s' % model.linear.bias)

#%% ****** 4.0 Teste ******

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.int)
pred_probas = model(X_test)               # Ver nota em comentários!
test_acc = model.comp_accuracy(torch.argmax(pred_probas[1], dim=1),y_test)

print()
print('TESTING RESULTS:')    
print('Test set accuracy: %.2f%%' % (test_acc*100))


#%% ****** 5.0 Graphic answer ******
# Code from Dalcimar apud Sebastian Raschka.

from matplotlib.colors import ListedColormap
import numpy as np
def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', '^', 'o', 'x', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    tensor = torch.tensor(np.array([xx1.ravel(), xx2.ravel()]).T).float()
    logits, probas = classifier.forward(tensor)
    Z = np.argmax(probas.detach().numpy(), axis=1)

    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, color=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)
        
plot_decision_regions(X_test, y_test, classifier=model)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


print('\nDistribuição do target nas classes em treino e teste foi:')
print(y_train.bincount())
print(y_test.bincount())