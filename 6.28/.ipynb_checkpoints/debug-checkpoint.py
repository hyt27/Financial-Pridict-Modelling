######布林带策略
from predict_v2 import predict
from lstm_train import lstm_train
import pylab as plt
import pandas as pd
import joblib
import matplotlib.dates as mdates
from trading_strategy.technical_trading import momentum_strategy
import tushare as ts
# lstm_train('stock_data/0066.HK.csv',"2010-01-01","2020-06-30",num_epochs=200)


# 假设 df 是你的 DataFrame
# 示例数据
ts.set_token('a8a8aff322d10b48165a40f49b8fb2b8a899495fd652e40dbbcb241b')
pro = ts.pro_api()

    # 设置股票代码和时间范围
stock_code = '000001.SZ'  # 例如，'000001.SZ' 代表平安银行


    # 下载股票数据
df = pro.daily(ts_code=stock_code, start_date='2020-1-1', end_date='2024-1-1')
df.rename(columns={'close': 'Close'}, inplace=True)
df.rename(columns={'trade_date': 'Date'}, inplace=True)
df.rename(columns={'open': 'Open'}, inplace=True)
df.rename(columns={'high': 'High'}, inplace=True)
df.rename(columns={'low': 'Low'}, inplace=True)
    # 显示数据


    # 可以选择保存数据到 CSV 文件
df.to_csv('testdata.csv')



    # 可以选择保存数据到 CSV 文件
df.to_csv('stock_data.csv')
datapath = 'stock_data.csv'
data = pd.read_csv(datapath)
data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
    # 将日期从字符串转换为 datetime 对象

    # 格式化日期显示格式
# data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
data.set_index('Date',inplace=True)

data = data.loc['2021-1-1':'2023-12-12']
print(data)