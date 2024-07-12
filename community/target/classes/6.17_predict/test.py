# -*- coding: utf-8 -*-
"""
Created on Fri May 24 18:52:29 2024

@author: hyt
"""

#####线性回归
from predict_v2 import predict
import pylab as plt
import pandas as pd
import joblib
import matplotlib.dates as mdates
from models.linear_regression import linearregression_train
#from trading import buy_and_sell_in_one_day
from trading_strategy.machinelearning_trading import buy_and_select_when_sell
#from machinelearning_trading import buy_and_sell_in_5_days
from trading_strategy.machinelearning_trading import buy_and_sell_in_5_days

####训练

linearregression_train(datapath = 'stock_data/^HSI.csv',startdate = "2010-01-01",enddate = "2020-06-30")
###预测

mse,actual,pred,whole_predict = predict('linearregression',datapath = 'stock_data/^HSI.csv',startdate = "2020-06-01",enddate = "2023-3-30")
print(whole_predict);

buy_and_sell_in_5_days(whole_predict,'log/linear_regre_B5D.txt')
#buy_and_select_when_sell(whole_predict,'log/linear_regre_HSI.txt')
plt.figure(figsize = (24,8))
plt.plot(actual['Date'],actual['Close'], label ='actual')
plt.plot(pred['Date'],pred['Close'],label = 'pred' )
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xlabel('Date')
plt.ylabel('Value')
plt.gcf().autofmt_xdate()
plt.title('Predictions vs Actual Data')
plt.show()
print("Mean Squared Error at prediction:", mse)


