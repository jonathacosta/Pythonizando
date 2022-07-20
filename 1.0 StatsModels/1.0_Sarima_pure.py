#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 12:31:10 2021
@author: Jonatha Costa
Módulo para análise de série temporial com 
pdmarima, gráficos, análise exploratória, predição e forecast
"""

from pmdarima import auto_arima
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error,mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
plt.rcParams["figure.figsize"] = (15,6)

class states_graf():   
    
    ''' Class apresenta gráficos estatísticos'''

        
    def states_graf(self,df):     
        from pandas.plotting import autocorrelation_plot     
               
        result=(seasonal_decompose(df, extrapolate_trend = 'freq'))        
        result.plot()    
        
        plt.figure()
        autocorrelation_plot(df, ax=plt.gca())
        
        plt.figure()
        plt.subplot(2,2,1)
        plot_acf(df,ax=plt.gca())
        plt.subplot(2,2,2)
        plot_pacf(df,ax=plt.gca())
        plt.subplot(2,2,3)
        df.plot(kind='hist',grid=True,ax=plt.gca())
        plt.subplot(2,2,4)
        df.plot(kind='kde',grid=True,ax=plt.gca())
        plt.show()   
        
class sarima():
        
    ''' Class métodos contém os métodos ARIMA, SARIMA e outros'''
    def __init__(self):
        states_graf.__init__(self)            
       
    def auto_arima(self,df,m=3,max_p=4, max_q=4, max_P=4, max_Q=4, trace=False):
        '''
        Método calcula o melhor ajusta pdq x PDQ. '''
        
        fit_arima = auto_arima(df, 
                               d=1, start_p=1, start_q=1,
                               D=1, start_P=1, start_Q=1,
                               max_p = max_p, max_q = max_q,
                               max_P = max_P, max_Q = max_Q, 
                               seasonal=True, m = m, 
                               trace = trace, 
                               information_criterion='aic',
                               error_action='ignore', 
                               stepwise=False)
        return fit_arima.order , fit_arima.seasonal_order
    
    def sarima_model(self, df, order,seasonal_order):
                                    
        model = SARIMAX(df, order=order,  seasonal_order=seasonal_order)
        modelo_sarima = model.fit()
        return modelo_sarima, modelo_sarima.summary()
       
    def sarima_predicao(self,df,modelo_sarima,past,graf=0):
        # Previsões retroativas
        predicoes = modelo_sarima.get_prediction(start=-past)
        predicao_media = predicoes.predicted_mean
        
        # Intervalos
        intervalo_confianca = predicoes.conf_int()
        limites_abaixo = intervalo_confianca.iloc[:,0]
        limites_acima = intervalo_confianca.iloc[:,1]
        limites_abaixo[0], limites_acima[0], predicao_media[0]
        
        # Erro predição
        erro_predicao = round(mean_squared_error(df[-past: ].values, predicao_media.values),4)
        rmse = round(erro_predicao**0.5,4)
        mape = round(mean_absolute_error(df[-past:].values, predicao_media.values),4)
        print(f'Erros de predição para {past} períodos retroativos.\n mse:{erro_predicao}\n rmse:{rmse}\n mape:{mape}')             

              
        if graf==1:
            self.graf_sarima(df,
                             erro_predicao=erro_predicao,  
                             predicao_media=predicao_media, 
                             limites_abaixo=limites_abaixo, 
                             limites_acima=limites_acima)

        return predicao_media,intervalo_confianca
        plt.plot(predicao_media)
        plt.plot(intervalo_confianca)
        
    def sarima_forecast(self,df,modelo_sarima,future, graf=0):
        # Forecast
        forecast = modelo_sarima.get_forecast(steps=future)
        forecast_medio = forecast.predicted_mean

        # Intervalos            
        intervalo_confianca_forecast = forecast.conf_int()
        intervalo_abaixo_f = intervalo_confianca_forecast.iloc[:,0]  # linhas 0:n e coluna 0
        intervalo_acima_f = intervalo_confianca_forecast.iloc[:,1]   # linhas 0:n e coluna 1
               
        if graf==1:
            self.graf_sarima(df,
                             forecast_medio=forecast_medio,
                             intervalo_abaixo_f=intervalo_abaixo_f,
                             intervalo_acima_f=intervalo_acima_f,
                             )
        
        return forecast_medio,intervalo_confianca_forecast

            
    def sarima_full(self,df, modelo_sarima, past, future, conf=0, graf=0):
        # Previsões retroativas
        predicoes = modelo_sarima.get_prediction(start=-past)
        predicao_media = predicoes.predicted_mean
        
        # Intervalos
        intervalo_confianca = predicoes.conf_int()
        limites_abaixo = intervalo_confianca.iloc[:,0]
        limites_acima = intervalo_confianca.iloc[:,1]
        limites_abaixo[0], limites_acima[0], predicao_media[0]
             
        # Forecast
        forecast = modelo_sarima.get_forecast(steps=future)
        forecast_medio = forecast.predicted_mean
          
        # Intervalos            
        intervalo_confianca_forecast = forecast.conf_int()
        intervalo_abaixo_f = intervalo_confianca_forecast.iloc[:,0]  # linhas 0:n e coluna 0
        intervalo_acima_f = intervalo_confianca_forecast.iloc[:,1]   # linhas 0:n e coluna 1
                    
        # Erro predição
        erro_predicao = (mean_squared_error(df[-past:].values, predicao_media.values))**0.5
        
        if graf==1:
            self.graf_sarima(df, erro_predicao, conf,
                             forecast_medio, intervalo_abaixo_f, intervalo_acima_f,
                             predicao_media, limites_abaixo, limites_acima)
             
    def graf_sarima(self,df,
                    erro_predicao=None,conf=None, 
                    forecast_medio=None,
                    intervalo_abaixo_f=None, intervalo_acima_f=None,
                    predicao_media=None,
                    limites_abaixo=None, limites_acima=None):
        
        plt.figure()
              
        # Bloco dados reais
        datas = np.asarray(df.index)               
        plt.plot(datas, df.values, label='real')           
        
        for year in range(df.index.year[0],df.index.year[-1]+1):
            plt.axvline(pd.to_datetime(str(year)+'-01-01'),color='k',linestyle='--',alpha=0.5)

        # Bloco predição
        try:
            datas_previsao = np.asarray(predicao_media.index)
            plt.plot(datas_previsao, predicao_media.values, color='red',label='prediction')
            plt.title(f'Dados reais e predição utilizando Sarima. Erro (rmse) = {round(erro_predicao,3)}.')

            if conf==1:
                plt.fill_between(datas_previsao, limites_abaixo, limites_acima, color='red')                     
        except:
            pass
               
        # Bloco forecast
        try:
            datas_forecast = np.asarray(forecast_medio.index)
            plt.plot(datas_forecast,forecast_medio.values,color='green',label='forecast')
            plt.title('Dados reais e forecast utilizando Sarima')
            if conf==1:
                plt.fill_between(datas_forecast, intervalo_abaixo_f, intervalo_acima_f, color='lightgreen')              
        except:
            pass
        
        plt.legend()    
        plt.show()


def escopo():
    pass

if '__name__==__main__':
    print('oi')