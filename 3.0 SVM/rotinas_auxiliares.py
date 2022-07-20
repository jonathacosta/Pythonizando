#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 10:13:31 2021

@author: Costa,JR
Funções auxiliares para análise consumo
"""

from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt    
from sklearn.svm import SVR    
import pandas as pd
import numpy as np
from tqdm import tqdm

def dados_feaac(periodo='D'):
    '''Método import e realiza pré-tratamento dos dados de consumo disponíveis
    no link 'https://raw.githubusercontent.com/JonathaCosta/SmartGrid/main/Datasets/consumption_feaac.csv'.
    Os dados são agrupados pelo período diário('D') ou conforme informado ('D','W','S','Y') e então retornam
    separados em X e y.
    '''
    
    dataset = 'https://raw.githubusercontent.com/JonathaCosta/SmartGrid/main/Datasets/consumption_feaac.csv'
    dados = pd.read_csv(dataset,index_col=[0],
                        header=0, sep=',',
                        decimal=',',
                        parse_dates=True, infer_datetime_format=True)
    dados = dados.resample(str(periodo)).sum()
    dados = dados.rename(columns = {dados.columns[0]:'X'})
    dados['y'] = dados.X.shift(-1)
    dados.dropna(inplace=True)
    X = dados.X.values.reshape(-1,1)
    y = dados.y.values.reshape(-1,1)
    print(f'Dados de consumo com período {periodo}. Arquivo feeac!')
    return X,y    


def eval_jrc(X,y,model,dataset): 
    '''Método calcula os erros de um dataset '''    
    y_pred = model.predict(X).reshape(-1,1)
    print(f'\nEVALUATION - {dataset} date_set:')
    
    res = pd.DataFrame(     
        columns=['rank','RMSE','MAE','R2','kernel','C','gamma','epsilon'])        
    res.RMSE = (mean_squared_error(y, y_pred)**0.5).ravel()
    res.MAE = mean_absolute_error(y, y_pred)
    res.R2 = r2_score(y, y_pred)
    res.kernel = model.kernel
    res.C = model.C
    if model.gamma == 'scale' : 
        res.gamma = (1/((X.shape[1])*X.var()))
    else:        
        res.gamma = model.gamma
            
    res.epsilon = model.epsilon
    res['rank'] = 'único'
    res = res.set_index('rank')
    
    return res

def graf(X,y_true,y_pred,scaler,tipo):
    """Método exibe gráficos com os dados de x,y_true e y_pred """
    
    
    plt.scatter(scaler.inverse_transform(X),scaler.inverse_transform(y_true),
                color="darkorange", label=f'{tipo}')
    plt.scatter(scaler.inverse_transform(X),scaler.inverse_transform(y_pred),
                color="navy",label='Predições')
    plt.legend(framealpha=1, frameon=True);
    plt.ylabel('Consumo (kWh)')
    plt.title(tipo)   
    plt.style.use('ggplot')
    plt.legend(loc="upper left")
    plt.show()
    

def grid_jrc_svr(X_train,y_train,X_test,y_test,scaler):
    '''Método realiza busca exaustiva de hiperparâmetros
    do estimador SVR.
    Ranking realizado por RMSE.'''    
    
    C = np.arange(0.5,15,0.5)
    kernel=['rbf','linear','sigmoid','poly']
    gamma = [1/((X_train.shape[1])*X_train.var()),]
    epsilon = [0.1,0.2,0.5,0.3]
    
    l_rmse,l_mae,l_r2 = [],[],[]
    l_kernel,l_C,l_gamma,l_epsilon = [],[],[],[]
    print()
    for l in tqdm(epsilon):
        for k in gamma:
            for j in C:
                for i in kernel: 
                    model = SVR(kernel=i,C=j,gamma=k,epsilon=l).fit(X_train,y_train.ravel())
                    y_pred = model.predict(X_test)
                    rmse_iter = mean_squared_error(y_test, y_pred)**0.5
                    mae_iter = mean_absolute_error(y_test, y_pred)
                    r2_iter = r2_score(y_test, y_pred)
                    l_rmse.append(rmse_iter)
                    l_mae.append(mae_iter) 
                    l_r2.append(r2_iter)
                    l_kernel.append(i), l_C.append(j), 
                    l_gamma.append(k), l_epsilon.append(l)
                    
    res = pd.DataFrame(columns=['rank','RMSE','MAE','R2','kernel','C','gamma','epsilon'])        
    res.RMSE,res.MAE,res.R2 = l_rmse, l_mae, l_r2 
    res.kernel, res.C, res.gamma, res.epsilon = l_kernel,l_C, l_gamma, l_epsilon
    res.sort_values(by='RMSE',inplace=True)
    res['rank'] = np.arange(1,res.shape[0]+1)
    res = res.set_index('rank')
            
    idx = res.index[0]
    model = SVR(kernel = res.kernel[idx],
                C=res.C[idx], 
                gamma = res.gamma[idx],
                epsilon = res.epsilon[idx]
                ).fit(X_train,y_train.ravel())
    y_pred = model.predict(X_test).reshape(-1,1)
    graf(X_test, y_test, y_pred, scaler, 'Grid JRC')
    # print(f'{res.loc[idx]}\n')
    return res
    
def grid_sklearn(model,X_train,y_train,X_test,y_test,scaler):
    '''Método realiza a busca exaustiva de hiperparâmetros implementando
    gridsearchCV com validação cruzada k-fold default.'''
    parametros = {'kernel': ('rbf','linear','poly','sigmoid',),
                  'C' : np.arange(0.5,15,0.5),
                  'epsilon' : [0.1,0.2,0.5,0.3],
                  'gamma' : [1/((X_train.shape[1])*X_train.var())]
                   }
    
    grid = GridSearchCV(model,
                        cv=5,
                        param_grid = parametros,
                        scoring='neg_root_mean_squared_error',
                        # scoring='neg_mean_absolute_error',
                        verbose=1
                        ).fit(X_train,y_train.ravel())
                
    # Results sorting    
    res = pd.DataFrame(columns=['rank','mean_rmse','std_test_score'])
    
    res['rank'] = grid.cv_results_["rank_test_score"]
    res['mean_rmse'] = grid.cv_results_["mean_test_score"]*-1
    res['std_test_score'] = grid.cv_results_["std_test_score"]
    res['kernel'] = grid.cv_results_["param_kernel"]
    res['C'] = grid.cv_results_["param_C"]
    res['gamma'] = grid.cv_results_["param_gamma"]
    res['epsilon'] = grid.cv_results_["param_epsilon"]
        
    res.sort_values(by = "mean_rmse", ascending=True, inplace=True)
    res = res.set_index('rank')
    
    # print('\n',res.head())
  
    model = grid
    y_pred = model.predict(X_test).reshape(-1,1)
    graf(X_test, y_test, y_pred, scaler, 'GridSearhCV')
    # print(f'{res.loc[1]}\n')
    return res,grid



   