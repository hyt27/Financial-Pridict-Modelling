from predict_v2 import predict
import pylab as plt
import pandas as pd
import joblib
import matplotlib.dates as mdates
from models.linear_regression import linearregression_train
import tushare as ts
from trading_strategy.machinelearning_trading import buy_and_sell_in_one_day
from trading_strategy.machinelearning_trading import buy_and_select_when_sell
from trading_strategy.machinelearning_trading import buy_and_sell_with_ma
from trading_strategy.technical_trading import double_moving_ave
from trading_strategy.technical_trading import single_moving_ave
from trading_strategy.technical_trading import Bollinger_Bands
import sys
import json


def meachinlearning(task_id = '1',stock_id = '000001.SZ',start = '2001-1-1',end = '2024-1-1',modelname = 'linearregression',strategy = 'buy_and_select_when_sell',balance = 100000):
    # 初始化 tushare，替换 'your_token_here' 为你的 Tushare Token
    ts.set_token('a8a8aff322d10b48165a40f49b8fb2b8a899495fd652e40dbbcb241b')
    pro = ts.pro_api()


    # 下载股票数据
    df = pro.daily(ts_code=stock_id, start_date=start, end_date=end)
    df.rename(columns={'close': 'Close'}, inplace=True)
    df.rename(columns={'trade_date': 'Date'}, inplace=True)
    df.rename(columns={'open': 'Open'}, inplace=True)
    df.rename(columns={'high': 'High'}, inplace=True)
    df.rename(columns={'low': 'Low'}, inplace=True)
    df.rename(columns={'amount': 'Volume'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    df.set_index('Date',inplace=True)
    df.sort_index(inplace=True)
    df.to_csv('stock_data.csv')

    datapath = 'stock_data.csv'
    # data = pd.read_csv(datapath)
    # data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
    # 将日期从字符串转换为 datetime 对象

    # 格式化日期显示格式
    # data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
    # data.set_index('Date',inplace=True)
    # data.sort_index(inplace=True)
 
    # data = data.loc[start:end]
    ####model
    if modelname ==  'lstm':
        mse_train,mse_test,testPredictPlot,trainPredictPlot,data,whole_predict= predict('lstm',datapath = datapath,startdate = start,enddate = end)
    elif modelname ==  'linearregression':
        linearregression_train(datapath = datapath,startdate = start,enddate =end)
        mse,actual,pred,whole_predict = predict('linearregression',datapath = datapath,startdate = start,enddate = end)
    
    plt.ioff()
    #####作prediction图并保存
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
    # plt.show()
    plt.savefig('prediction_'+task_id+'.jpg')
    plt.close()


    log_path = 'log_'+task_id+'.txt'
    ####strategy
    if strategy == 'buy_and_select_when_sell':
        yield_curve,final_balance,yield_rate,benchmark_yield = buy_and_select_when_sell(whole_predict,log_path,balance=balance) 

    elif strategy == 'buy_and_sell_in_one_day':
        yield_curve,final_balance,yield_rate,benchmark_yield = buy_and_sell_in_one_day(whole_predict,log_path,balance=balance)

    elif strategy == 'buy_and_sell_with_ma':
        yield_curve,final_balance,yield_rate,benchmark_yield = buy_and_sell_in_5_days(whole_predict,log_path,balance=balance)

    # plt.ioff()
    #####作收益曲线图并保存


    
    # plt.ylim(-100,100)
    plt.figure(figsize=(12,8))
    plt.plot(yield_curve.index,yield_curve['return']*100,label = 'return')
    plt.plot(yield_curve.index,yield_curve['benchmark']*100,label = 'benchmark')

    plt.xlabel('Date')
    plt.ylabel('year yield rate %')
    

    plt.gcf().autofmt_xdate()
    plt.title('return  vs benchmark on '+task_id)
    plt.legend()

    plt.savefig('return_rate_'+task_id+'.jpg')
    plt.close()
    result = {"预测图片":'prediction_'+task_id+'.jpg',
               "收益率图片": 'return_rate_'+task_id+'.jpg',
               "日志信息":'log/log_'+task_id+'.txt',
               "收益率":yield_rate,
               "基准收益率":benchmark_yield,
               "账户价值":final_balance}
    return  json.dumps(result, ensure_ascii=False)

def technical(task_id = '1',stock_id = '000001.SZ',start = '2001-1-1',end = '2024-1-1',strategy = '单均线策略',balance = 100000):
    ####data
    # 初始化 tushare，替换 'your_token_here' 为你的 Tushare Token
    ts.set_token('a8a8aff322d10b48165a40f49b8fb2b8a899495fd652e40dbbcb241b')
    pro = ts.pro_api()


    # 下载股票数据
    df = pro.daily(ts_code=stock_id, start_date=start, end_date=end)
    df.rename(columns={'close': 'Close'}, inplace=True)
    df.rename(columns={'trade_date': 'Date'}, inplace=True)
    df.rename(columns={'open': 'Open'}, inplace=True)
    df.rename(columns={'high': 'High'}, inplace=True)
    df.rename(columns={'low': 'Low'}, inplace=True)
    df.rename(columns={'amount': 'Volume'}, inplace=True)
    
    df.to_csv('stock_data.csv')


    datapath = 'stock_data.csv'
    data = pd.read_csv(datapath)
    data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
    # 将日期从字符串转换为 datetime 对象

    # 格式化日期显示格式
    # data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
    data.set_index('Date',inplace=True)
    data.sort_index(inplace=True)
 
    data = data.loc[start:end]
    log_path = 'log_'+task_id+'.txt'
    ####strategy
    if strategy == 'single_moving_average_strategy':
        yield_curve,final_balance,yield_rate,benchmark_yield = single_moving_ave(data,logpath = log_path,balance=balance) 

    elif strategy == 'double_moving_average_strategy':
        yield_curve,final_balance,yield_rate,benchmark_yield = double_moving_ave(data,logpath = log_path,balance=balance)

    elif strategy == 'bollinger_bands_strategy':
        yield_curve,final_balance,yield_rate,benchmark_yield = Bollinger_Bands(data,logpath = log_path,balance=balance)

    # plt.ioff()
    #####作收益曲线图并保存


    
    # plt.ylim(-100,100)
    plt.figure(figsize=(12,8))
    plt.plot(yield_curve.index,yield_curve['return']*100,label = 'return')
    plt.plot(yield_curve.index,yield_curve['benchmark']*100,label = 'benchmark')

    plt.xlabel('Date')
    plt.ylabel('year yield rate %')
    

    plt.gcf().autofmt_xdate()
    plt.title('return  vs benchmark on '+task_id)
    plt.legend()

    plt.savefig('return_rate_'+task_id+'.jpg')
    plt.close()


    result = {"收益率图片": 'return_rate_'+task_id+'.jpg',
               "日志信息":'log/log_'+task_id+'.txt',
               "收益率":yield_rate,
               "基准收益率":benchmark_yield,
               "账户价值":final_balance}
    return  json.dumps(result, ensure_ascii=False)





# result = meachinlearning(task_id = 'lstm测试',stock_id = '000004.SZ',start = '2020-1-1',end = '2024-1-1',modelname = 'lstm',strategy = 'buy_and_sell_in_one_day',balance = 100000)
# # result = technical(task_id = '布林带测试1',stock_id = '600000.SH',start = '2020-1-1',end = '2024-1-1',strategy = '布林带策略',balance = 100000)
# print(result)

if __name__ == '__main__':
    task_id = sys.argv[1]
    category = sys.argv[2]
    stock_id = sys.argv[3]
    start = sys.argv[4]
    end = sys.argv[5]
    modelname = sys.argv[6]
    strategy = sys.argv[7]
    initial_balance = int(sys.argv[8])
    

    ####机器学习
    if category == '1':
       
        print(meachinlearning(task_id,stock_id,start,end,modelname,strategy,initial_balance))
        
        
    ####技术面策略
    elif category == '2':
        print(technical(task_id,stock_id,start,end,strategy,initial_balance))
        # return result
