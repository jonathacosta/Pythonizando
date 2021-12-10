#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 12:31:10 2021
@author: Jonatha Costa
Code for dataset analyse assistence
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class Preprocess():
    '''
    JRC\n
    Classe contém métodos de apoio para o pré-processamento de dataset
    * __init__   : recebe dataset e retorna o Dataframe(df) 
    * períodos   : recebe df e agrupa para sia,semana e mes
    * train_test : separa o dataset em conjunto de treino e teste
    '''    
        
    def dataset_to_df(self,dataset,verbose=0):
        '''
        Método recebe um dataset e pré-processa-o para retornar um DataFrame.
        
        return dados

        '''
        try:
            dados = pd.read_csv(dataset,index_col=[0],
                                header=0, sep=',',
                                decimal=',',
                                parse_dates=True, infer_datetime_format=True)
            dados.asfreq(pd.infer_freq(dados.index))
            return dados
           
        except:
            try:
                dados = pd.read_csv(dataset,index_col=[0],
                                header=1, sep=';',
                                decimal=',',
                                parse_dates=True, infer_datetime_format=True)
                del dados[dados.columns[-1]]
                dados.asfreq(pd.infer_freq(dados.index))
                return dados
                   
            except:
                print('-'*70 +'\nFalha num dos parâmetros do dataset! \nAnalise-o diretamente na base de dados \n' + '-'*70)
                print('\n\n')
                return dados
            else:
                print('\nO dataset contém datas iguais.\nFrequência horária foi atribuida para o mesmo.')
            
                print('\n\nConfirme com resultado!')
          
        finally:
             
              if verbose ==1:
                  print('','-'*50,"\n \t\tDataFrame do Dataset original !\n",'-'*50)
                  print(dados.head())
                  if str(dados.index.freq) == 'None':
                      print('\nALERTA: Frequência do dataset não identificada automaticamente!')
                      print("Execute df.asfreq('defina manualmente o valor da frequencia') para corrigir isso!")
                  else:
                      print('\nDataset com frequência ',str(dados.index.freq))              
              
    def periodo(self,df,verbose=0):
        '''Método recebe um df e retorna 3 saídas com período:
        diario, semanal, mensal realizando a soma dos valores.
        
        return diario, semanal, mensal

        '''
        diario = df.resample('D').sum()
        semanal = df.resample('W').sum()
        mensal = df.resample('M').sum()    
        if verbose==1: print('\nPeríodos separados em dia, semanal e mes!\n')
        return diario, semanal, mensal    
    
    def sep_train_test(self, df, tamanho_treino = 0.8, embaralhar=False,verbose=0):
        '''Método recebe um df e retorna train_set e test_set.
        'tamanho_treino  -  0.8': 80% dos dados para treino e 20% para validação
        'embaralhar - False': não embaralhar os dados, no caso de análise de séries temporais.
        '''
        
        train , test = train_test_split(df,test_size = tamanho_treino, shuffle=embaralhar)
        if verbose ==1: print('\nDataset separado!\n',train.shape, test.shape)                  
        return train,test
    
    def normaliza_df(self,df,verbose=0):
        ''' Método normaliza o df entre os valore [0,1] mantendo o índice.
        '''
        index = df.index
        df_norm = MinMaxScaler().fit_transform(df)
        df_norm = pd.DataFrame(df_norm, index = index)        
        if verbose==1: print(df_norm)
        return df_norm

    def dataset(self,dados=None):
        '''Informe: feacc,fort17, fort18, ... '''
        disc_dataset={
        'bd1' :'https://raw.githubusercontent.com/JonathaCosta/3_SmartGrid/main/Datasets/consumption_feaac.csv',
        'bd2' :'https://raw.githubusercontent.com/JonathaCosta/3_SmartGrid/main/Datasets/2017_FORTALEZA(A305).csv',
        'bd3' :'https://raw.githubusercontent.com/JonathaCosta/3_SmartGrid/main/Datasets/2018_FORTALEZA%20(A305).csv',
                    }

        if dados == None: 
            df = self.dataset_to_df(disc_dataset['bd1'])
        else:
            df = self.dataset_to_df(disc_dataset[dados])
        return df