# -*- coding: utf-8 -*-
"""
Created on Fri May 24 20:47:03 2024

@author: hyt
"""

######训练lstm
from predict_v2 import predict
from lstm_train import lstm_train
import pylab as plt
import pandas as pd
import joblib
import matplotlib.dates as mdates
from trading_strategy.machinelearning_trading import buy_and_sell_in_one_day
from trading_strategy.machinelearning_trading import buy_and_select_when_sell
from trading_strategy.machinelearning_trading import buy_and_sell_in_5_days


# lstm_train('stock_data/0066.HK.csv',"2010-01-01","2020-06-30",num_epochs=200)

#####利用训练好的lstm模型预测
mse_train,mse_test,testPredictPlot,trainPredictPlot,data,whole_predict= predict('lstm',datapath = 'stock_data/002948.SZ.csv',startdate = "2020-01-11",enddate = "2024-4-29")
###交易回测
buy_and_sell_in_one_day(whole_predict,'lstm_log_HSI_1.txt')
#buy_and_select_when_sell(whole_predict,'log/lstm_log_BQD._2.txt') 
#buy_and_sell_in_5_days(whole_predict,'log/lstm_log_B5D.txt') 

plt.figure(figsize=(12,8))

# plt.plot(pd.to_datetime(data['Date']),data['Close'],label = 'data')
# plt.plot(pd.to_datetime(testPredictPlot['Date']),testPredictPlot['Close'],label = 'test_pred')
# plt.plot(pd.to_datetime(trainPredictPlot['Date']),trainPredictPlot['Close'],label = 'train_pred')
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