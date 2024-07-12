import yfinance as yf
import pandas as pd

# 股票列表
stocks = ["AAPL", "GOOGL", "MSFT", "AMZN","TSLA", "BRK-A", "V", "JNJ", "WMT"]

# 数据下载和保存
for stock in stocks:
    # 下载股票数据
    data = yf.download(stock, start="2000-01-01", end="2024-04-30")
    print(data)
    # 保存到CSV
    # data.to_csv('stock_data/'+f"{stock}.csv")