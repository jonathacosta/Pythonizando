#!/usr/bin/env python
# coding: utf-8
'''
Análise de séries temporais - introdução

Séries básicas utilizando:
* Autoregressão

Série utilizará os datasets públicos disponíveis na web e alocados no github do autor: JonathaCosta
dados='https://raw.githubusercontent.com/JonathaCosta/3_SmartGrid/main/Datasets/consumption_feaac.csv'
dados='https://raw.githubusercontent.com/JonathaCosta/1_Notebooks/main/Datasets/timeline.csv'
dados='https://raw.githubusercontent.com/JonathaCosta/1_Notebooks/main/Datasets/clima_india.csv'
'''
from datetime import timedelta
from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima.model import ARIMA

dados='https://raw.githubusercontent.com/JonathaCosta/1_Notebooks/main/Datasets/clima_india.csv'
dff = pd.read_csv(dados,index_col=0,parse_dates=True)
dff=dff.asfreq(pd.infer_freq(dff.index))    # Faz reconhecimento de frequência da série

# In[0]: Codes 
filtro_data = pd.to_datetime('2013-01-01')  # Filtro de data 2014 em diante
col=dff.columns[0]
df = dff.loc[filtro_data:,[col]].copy()

# Resposta gráfica
plt.figure(figsize=(10,4))
df.plot()
plt.title(f'Curvas de {col}',fontsize=20)
plt.ylabel('Fator u.m')
for year in range(2014,2017):
    plt.axvline(pd.to_datetime(str(year)),color='k',linestyle='--',alpha=0.5)

# Correlação
acf_plot = plot_acf(df,lags=100)
pacf_plot = plot_pacf(df)


# Separa o dataset em train e test
corte = int(len(df)*2/3) 
train_set = df[:corte]
test_set  = df[corte:]

# Modelo AR
model = ARIMA(train_set,order=(6,0,0))
start=time()
model_fit = model.fit()        # Ajuste
end=time()
print('Tempo de ajuste do modelo:',end-start)
print()
print(model_fit.summary())      # Resumo

# Previsões

prev_inicio = test_set.index[0]  # Data de previsão inicial
prev_fim = test_set.index[-1]   # Data de previsão final

prev = model_fit.predict(start=prev_inicio, end=prev_fim)
prev = prev.values.reshape(-1,1)  # Converte para formato array para operação a seguir
residuos = test_set - prev
residuos

# Análise gráfica dos Resíduos
plt.figure(figsize=(10,4))
plt.plot(residuos)
plt.title('Residuos do modelo AR', fontsize=20)
plt.ylabel('Erro',fontsize=16)
plt.axhline(0,color='r',linestyle='--',alpha=0.5)
for year in range(2014,2017):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'),color='k',linestyle='--',alpha=0.5)

# Análise gráfica previsão e valores
plt.figure(figsize=(10,4))
plt.plot(residuos)
plt.plot(test_set)
plt.legend(['Residuo','Trains_set'])
for year in range(test_set.index.year[0],test_set.index.year[-1]):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'),color='k',linestyle='--',alpha=0.5)

#  Erros
print('Mean Absolute Percent Error:', round( np.mean( abs( residuos/test_set)),4))
print('Root Mean Square Error:', np.sqrt(np.mean(residuos**2)))

# Média de previsões
# In[5]
previsoes_rolling = []

for end_date in test_set.index:
    train_set = df[:end_date - timedelta(days=1)]
    model = ARIMA(train_set,order=(1,0,0))
    model_fit = model.fit()
    prev = model_fit.predict(end_date)
    previsoes_rolling.append(prev)

previsoes_rolling = np.array(previsoes_rolling).reshape(-1,1)
residuos_rolling = test_set - previsoes_rolling
residuos_rolling.head()

#%%
plt.figure(figsize=(10,4))
plt.plot(residuos_rolling)
plt.plot(test_set)
plt.legend(['Residuo_rolling','Trains_set'])
for year in range(test_set.index.year[0],test_set.index.year[-1]):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'),color='k',linestyle='--',alpha=0.5)

#  Erros
print('Mean Absolute Percent Error:', round( np.mean( abs( residuos_rolling/test_set)),4))
print('Root Mean Square Error:', np.sqrt(np.mean(residuos_rolling**2)))
