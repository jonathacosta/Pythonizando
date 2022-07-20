#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 09:17:51 2021
@author: Jonatha Costa
Modelo ARIMA 


"""
# Bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

dataset='https://raw.githubusercontent.com/JonathaCosta/1_Notebooks/main/Datasets/clima_india.csv'
dados = pd.read_csv(dataset,index_col=0,parse_dates=True)
dados=dados.asfreq(pd.infer_freq(dados.index))    # Faz reconhecimento de frequência da série
print(f'Intervalo de dados: {dados.index.min()} a {dados.index.max()}')
df=dados.loc[:,['meantemp']].copy()

#%% Decompondo a série pra avaliar: tendência, sazonalidade e resíduo
resultado = seasonal_decompose(df)
fig = plt.figure(figsize=(8, 6))  
fig = resultado.plot()

# Teste de estacionariedade
result=adfuller(df.dropna())
if result[1] < 0.05: print('Série não estacionária!\n')
else: print('Série estacionária (p-valor > 0.05)!\n')
print(f'Teste ADF:{result[0]}')
print(f'p-valor:{result[1]}')

#%% Busca de melhores parâmetros com autoarima 
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

fit_arima = auto_arima(df, d=1, start_p=1, start_q=1, max_p=3, max_q=3,
                       seasonal=True, m=3, D=1, 
                       start_P=1, start_Q=1, 
                       max_P=2, max_Q=2, 
                       information_criterion='aic',
                       trace=True, 
                       error_action='ignore', 
                       stepwise=True)

#%% Modelo SARIMAX com os parâmetros do auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

order = fit_arima.order
seasonal_order = fit_arima.seasonal_order

# model = SARIMAX(df,order=(2,1,2),  seasonal_order=(1, 1, 2, 6))
model = SARIMAX(df,order=order,  seasonal_order=seasonal_order)
resultado_sarimax = model.fit()
resultado_sarimax.summary()

#%% Teste e previsões
# "backtesting" com predições de valores dos 12 meses anteriores
predicoes = resultado_sarimax.get_prediction(start=-12)
predicao_media = predicoes.predicted_mean

# Intervalos
intervalo_confianca = predicoes.conf_int()
limites_abaixo = intervalo_confianca.iloc[:,0]
limites_acima = intervalo_confianca.iloc[:,1]
limites_abaixo[0], limites_acima[0]
predicao_media[0]
datas_previsao = np.asarray(predicao_media.index)
datas = np.asarray(df.index)

# Respost gráfica
plt.figure(figsize=(10,6))
plt.plot(datas_previsao,predicao_media.values,color='red',label='prediction')
plt.fill_between(datas_previsao, limites_abaixo, limites_acima, color='red')
plt.plot(datas, df.values, label='real')
plt.legend()
plt.show()

# Erro
from sklearn.metrics import mean_squared_error
from math import sqrt

rmse_sarima = sqrt(mean_squared_error(df[-12:].values, predicao_media.values))
print(rmse_sarima)


#%% Forecast pra 12 values

forecast = resultado_sarimax.get_forecast(steps=12)
forecast_medio = forecast.predicted_mean
forecast_medio.size

#%%
intervalo_confianca_forecast = forecast.conf_int()
intervalo_abaixo_f = intervalo_confianca_forecast.iloc[:,0]
intervalo_acima_f = intervalo_confianca_forecast.iloc[:,1]

intervalo_abaixo_f[10], intervalo_acima_f[10]
forecast_medio[10]

datas_forecast = np.asarray(forecast_medio.index)
pred_dates=np.asarray(predicao_media.index)


dates=np.asarray(df.index)
plt.figure(figsize=(10,6))
plt.plot(datas_forecast,forecast_medio.values,color='green',label='forecast')
plt.fill_between(datas_forecast, intervalo_abaixo_f, intervalo_acima_f, color='lightgreen')

plt.plot(datas_previsao,predicao_media.values,color='red',label='prediction')
plt.fill_between(datas_previsao, limites_abaixo, limites_acima, color='pink')

plt.plot(dates,df.values, label='real')
plt.legend(loc='upper left')
plt.show()




















