# -*- coding: utf-8 -*-
"""
Created on Sun May 26 20:52:58 2024

@author: hyt
"""
from predict_v2 import predict
from lstm_train import lstm_train
import pylab as plt
import pandas as pd
import joblib
import matplotlib.dates as mdates
from trading_strategy.machinelearning_trading import buy_and_sell_in_one_day
from trading_strategy.machinelearning_trading import buy_and_select_when_sell
from trading_strategy.machinelearning_trading import buy_and_sell_in_5_days

model='randomforest'

if model=='randomforest':
    mse_train, mse_test, y_train, train_pred, y_test, test_pred, mtr, whole_predict= predict('randomforest',datapath = 'stock_data/002948.SZ.csv',startdate = "2020-01-11",enddate = "2024-4-29")
    buy_and_sell_in_5_days(whole_predict,'log/rf_log_B5D.txt') 
    
    plt.figure(figsize=(12,8))

   
    plt.plot(pd.to_datetime(whole_predict['Date']),whole_predict['pred_close'],label = 'pred')
    plt.plot(pd.to_datetime(whole_predict['Date']),whole_predict['act_close'],label = 'actual')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xlabel('Date')
    plt.ylabel('Value')
     
     
    plt.gcf().autofmt_xdate()
    plt.title('Predictions vs Actual Data')
    plt.legend()
    plt.show()


    print('MSE ON TRAIN:'+str(mse_train))
    print('MSE ON TEST:'+str(mse_test))
 
elif model=='svm':    
    mse_train, mse_test, y_train, train_pred, y_test, test_pred, mtr, whole_predict= predict('svm',datapath = 'stock_data/002948.SZ.csv',startdate = "2020-01-11",enddate = "2024-4-29")
    buy_and_sell_in_5_days(whole_predict,'log/svm_log_B5D.txt')

    plt.figure(figsize=(12,8))

   
    plt.plot(pd.to_datetime(whole_predict['Date']),whole_predict['pred_close'],label = 'pred')
    plt.plot(pd.to_datetime(whole_predict['Date']),whole_predict['act_close'],label = 'actual')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xlabel('Date')
    plt.ylabel('Value')
     
     
    plt.gcf().autofmt_xdate()
    plt.title('Predictions vs Actual Data')
    plt.legend()
    plt.show()


    print('MSE ON TRAIN:'+str(mse_train))
    print('MSE ON TEST:'+str(mse_test))

elif model=='gru':
    mse_train, mse_test, y_train, train_pred, y_test, test_pred, scaled_data, whole_predict= predict('gru',datapath = 'stock_data/002948.SZ.csv',startdate = "2020-01-11",enddate = "2024-4-29")
    buy_and_sell_in_5_days(whole_predict,'log/gru_log_B5D.txt')

    plt.figure(figsize=(12,8))

   
    plt.plot(pd.to_datetime(whole_predict['Date']),whole_predict['pred_close'],label = 'pred')
    plt.plot(pd.to_datetime(whole_predict['Date']),whole_predict['act_close'],label = 'actual')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xlabel('Date')
    plt.ylabel('Value')
     
     
    plt.gcf().autofmt_xdate()
    plt.title('Predictions vs Actual Data')
    plt.legend()
    plt.show()


    print('MSE ON TRAIN:'+str(mse_train))
    print('MSE ON TEST:'+str(mse_test))








