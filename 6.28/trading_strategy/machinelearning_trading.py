import pandas as pd
import pylab as plt
import numpy as np

######开盘买收盘卖
def buy_and_sell_in_one_day(whole_predict,logpath,balance = 100000,fees = 0):
    ###trading 在开盘时，如果预测今天的close比开盘价高则买入

    yield_curve = pd.DataFrame(index=whole_predict.index, columns=['balance', 'return','benchmark'])
    origin_balance = balance      ###原始本金
    origin_price =  whole_predict['act_close'].iloc[0]
    yesterday = whole_predict['act_close'].iloc[0]
    start = yesterday

    for i,row in whole_predict.iterrows():
        today = row['pred_close']
        if today > yesterday:
        # if today > row['open']:   ##如果预测收盘价结果大于开盘价

            ####开盘价买入
            buy_num = balance // row['open']    ##要买多少只
            
            balance = balance - buy_num * row['open'] - buy_num * row['open'] * fees  ###买完之后账户余额


            ###收盘价卖出
            balance = balance + buy_num * row['act_close']   ###卖出后余额

            today_yield = (row['act_close']-row['open']) * buy_num  ###当日收益


            with open(logpath, "a") as file:
                
        
                file.write('DATE:'+ str(row['Date'])[:-9]+' BUY '+ str(int(buy_num)) + ' at '+ str(round(row['open'],2)) + '  SELL at ' + str(round(row['act_close'],2)) + '  predicted close price=' + str(round(row['pred_close'],2)) + '  todayield:' + str(round(today_yield,2)) + '  balance:' + str(round(balance,2)) + "\n")  # 写入日志消息并添加换行符

        yesterday = row['act_close']


        yield_curve.loc[i, 'balance'] = balance
        date_difference = (row['Date'] - whole_predict.index.min()).days + 1

        
        yield_curve.loc[i, 'return'] = ((balance/origin_balance - 1)/date_difference) * 365

        yield_curve.loc[i, 'benchmark'] = ((row['act_close']/origin_price - 1)/date_difference) * 365


    yield_rate = balance/origin_balance-1

    start_date = whole_predict.index.min()
    end_date = whole_predict.index.max()

    # 计算日期差
    date_difference = (end_date - start_date).days
    year_rate = (yield_rate / date_difference) * 365

    benchmark_yield = ((row['act_close']/origin_price - 1)/date_difference) * 365     
    # print('balance:' + str(balance) + ' yield rate:' + str(year_rate*100) + '%')
    # print('benchimark yield:'+ str(benchmark_yield*100) + '%')

    # plt.figure(figsize=(24,8))

    
    # plt.ylim(-100,100)

    # plt.plot(yield_curve.index,yield_curve['return']*100,label = 'return')
    # plt.plot(yield_curve.index,yield_curve['benchmark']*100,label = 'benchmark')

    # plt.xlabel('Date')
    # plt.ylabel('year yield rate %')
    
    
    # plt.gcf().autofmt_xdate()
    # plt.title('return  vs benchmark on BankofQingdao')
    # plt.legend()
    # plt.show()
    return yield_curve,balance,year_rate * 100,benchmark_yield * 100


######开盘买开盘买（通过预测决定哪天卖）
def buy_and_select_when_sell(whole_predict,logpath,balance = 100000,tax = 0.001,fees = 0.0003):
    ###trading 在开盘时，如果预测今天的close比开盘价高则买入
    # print(whole_predict)

    yield_curve = pd.DataFrame(index=whole_predict.index, columns=['balance', 'return','benchmark'])
    origin_balance = balance      ###原始本金
    origin_price =  whole_predict['act_close'].iloc[0]
    ####初始化
    today_sell = False
    today_buy = False
    num_stock = 0       ##持有股票数
    # today_buy = False
    hadstock = False  ###当前是否持仓

    for i,row in whole_predict.iterrows():

        today_sell = False
        today_buy = False

        ########盘前决策######################################
        if hadstock == True:
            if nextday_sell == True:     ####开盘卖出
                ###今日决策
                today_sell = True     #####今天决定要卖 
                today_buy = False   
            
            elif whole_predict.index.get_loc(i) == len(whole_predict)-1:   ###如果今天是最后一天则卖出
                today_sell = True
                today_buy = False 

            ####反之继续持仓，跳到收盘决策步


        elif hadstock == False:    ####当前未持仓，要决定是否买入

            if whole_predict.index.get_loc(i)-1 > 0:
                yesterday_row = whole_predict.iloc[whole_predict.index.get_loc(i)-1]
            else:
                yesterday_row  = whole_predict.iloc[0]
            if whole_predict.index.get_loc(i) == len(whole_predict)-1:
                today_sell = False
                today_buy = False 
            elif row['pred_close'] > yesterday_row['act_close']:
                ###今日决策
                today_sell = False
                today_buy = True     ###今天决定要买
            elif whole_predict.index.get_loc(i) == len(whole_predict)-1:   ###如果今天是最后一天则卖出
                today_sell = False
                today_buy = False
    
        ########盘前决策结束##################################
            


        ########日内交易##################################

        if today_sell == True:    ####执行卖出
            balance = balance + num_stock * row['open'] - buy_num * row['open'] * (tax+fees)  ###卖出后余额
            hadstock = False
            
            with open(logpath, "a") as file:
                file.write('DATE:'+ str(row['Date'])[:-9]+' SELL '+ str(int(num_stock)) + ' stocks at '+ str(round(row['open'],2)) + '  balance:' + str(round(balance,2)) + "\n")  # 写入日志消息并添加换行符
            
            num_stock = 0
        if today_buy == True:     #####执行买入
            ####开盘价买入
            buy_num = balance // (row['open'] * (1+fees))    ##要买多少只
            balance = balance - buy_num * row['open'] - buy_num * row['open'] * fees  ###买完之后账户余额
            num_stock = buy_num
            hadstock = True

            with open(logpath, "a") as file:
                file.write('DATE:'+ str(row['Date'])[:-9]+' BUY '+ str(int(buy_num)) + ' stocks at '+ str(round(row['open'],2)) + '  balance:' + str(round(balance,2)) + "\n")  # 写入日志消息并添加换行符

        ########日内交易结束##################################





        #########盘后决策######################################

        if hadstock == True:
            if i + pd.Timedelta(days=1) < whole_predict.index.max():
                next_row = whole_predict.iloc[whole_predict.index.get_loc(i)+1]

                if next_row['pred_close'] < row['act_close']:    #####如果预测明天会跌则决定明天开盘卖出
                    nextday_sell = True
                else:                                           ####反之明天不卖继续持仓
                    nextday_sell = False

        #########盘后决策结束######################################

        ####记录
        # with open(logpath, "a") as file:
        #     file.write('DATE:'+ str(row['Date'])+' buy '+ str(buy_num) + ' at '+ str(row['open']) + ' balance:' + str(balance) + "\n")  # 写入日志消息并添加换行符


        #####记录每天余额和收益
        yield_curve.loc[i, 'balance'] = balance 
        date_difference = (row['Date'] - whole_predict.index.min()).days + 1

        
        yield_curve.loc[i, 'return'] = (((balance+ num_stock * row['act_close'])/origin_balance - 1)/date_difference) * 365

        yield_curve.loc[i, 'benchmark'] = ((row['act_close']/origin_price - 1)/date_difference) * 365


    ##########循环结束#################

    yield_rate = balance/origin_balance-1


    start_date = whole_predict.index.min()
    end_date = whole_predict.index.max()

    # 计算日期差
    date_difference = (end_date - start_date).days
    year_rate = (yield_rate / date_difference) * 365

    benchmark_yield = ((row['act_close']/origin_price - 1)/date_difference) * 365     
    # print('balance:' + str(balance) + ' yield rate:' + str(year_rate*100) + '%')
    # print('benchimark yield:'+ str(benchmark_yield*100) + '%')

    # plt.figure(figsize=(24,8))

    # plt.ylim(-100,200)

    # plt.plot(yield_curve.index,yield_curve['return']*100,label = 'return')
    # plt.plot(yield_curve.index,yield_curve['benchmark']*100,label = 'benchmark')

    # plt.xlabel('Date')
    # plt.ylabel('year yield rate %')
    
    
    # plt.gcf().autofmt_xdate()
    # plt.title('return  vs benchmark on BankofQingdao')
    # plt.legend()
    # plt.show()

    return yield_curve,balance,year_rate * 100,benchmark_yield * 100

def buy_and_sell_in_5_days(whole_predict, logpath, balance=100000, fees=0.0003):
    yield_curve = pd.DataFrame(index=whole_predict.index, columns=['balance', 'return', 'benchmark'])
    origin_balance = balance  # 原始本金
    origin_price = whole_predict['act_close'].iloc[0]
    num_stock = 0  # 持有股票数
    hadstock = False  # 当前是否持仓

    for i, row in whole_predict.iterrows():
        today_sell = False
        today_buy = False

        # 策略执行时间为每个星期一
        #if i.weekday() == 0:
            # 获取最近15个交易日的特征变量
            #features = whole_predict.loc[i - pd.DateOffset(days=15):i - pd.DateOffset(days=1), 'open':'volume']
            # 进行预测
            #prediction = model.predict(features)  # 假设使用名为model的预测模型进行预测，需要根据实际模型进行调整
            #prediction = features['prediction_column']  # 替换为实际预测结果的列名
            #whole_predict['prediction'] = (whole_predict['act_close'] < whole_predict['pred_close'].shift(-5)).astype(int)
            # 获取5个交易日后的预测收盘价
            # pred_close_5_days = whole_predict.loc[i + pd.DateOffset(days=5), 'pred_close']
            # 寻找5个交易日后的预测收盘价
            # 寻找5个交易日后的预测收盘价
        target_date = i + pd.DateOffset(days=5)
        while target_date not in whole_predict.index:
            target_date += pd.DateOffset(days=1)
            if target_date > whole_predict.index[-1]:
                break
        if target_date <= whole_predict.index[-1]:
            pred_close_5_days = whole_predict.loc[target_date, 'pred_close']
            # 根据预测收盘价与当前实际收盘价的比较，确定预测值
            if pred_close_5_days > row['act_close']:
                prediction = 1  # 上涨
            else:
                prediction = 0  # 下跌

            # 如果预测结果为上涨，则进行开仓操作购买标的股票
            if prediction == 1:
                today_buy = True

        # 如果已经持有仓位，在盈利大于10%时止盈
        if hadstock and (row['act_close'] - origin_price) / origin_price > 0.1:
            today_sell = True

        # 当时间为周五并且跌幅大于2%时，平掉所有仓位止损
        if i.weekday() == 4 and (row['act_close'] - origin_price) / origin_price < -0.02:
            today_sell = True

        # 执行买入操作
        if today_buy:
            buy_num = balance // (row['open'] * (1 + fees))  # 要买多少只
            balance = balance - buy_num * row['open'] - buy_num * row['open'] * fees  # 买完之后账户余额
            num_stock = buy_num
            hadstock = True

            with open(logpath, "a") as file:
                file.write('DATE:' + str(row['Date']) + ' buy ' + str(buy_num) + ' at ' + str(row['open']) + ' balance:' + str(balance) + "\n")  # 写入日志消息并添加换行符

        # 执行卖出操作
        if today_sell:
            balance = balance + num_stock * row['open'] - num_stock * row['open'] * fees  # 卖出后余额
            hadstock = False

            with open(logpath, "a") as file:
                file.write('DATE:' + str(row['Date']) + ' sell ' + str(num_stock) + ' at ' + str(row['open']) + ' balance:' + str(balance) + "\n")  # 写入日志消息并添加换行符

            num_stock = 0

        # 记录每天余额和收益
        yield_curve.loc[i, 'balance'] = balance
        date_difference = (row['Date'] - whole_predict.index.min()).days + 1

        yield_curve.loc[i, 'return'] = (((balance + num_stock * row['act_close']) / origin_balance - 1) / date_difference) * 365
        yield_curve.loc[i, 'benchmark'] = ((row['act_close'] / origin_price - 1) / date_difference) * 365

    # 计算收益率
    yield_rate = balance / origin_balance - 1
    start_date = whole_predict.index.min()
    end_date = whole_predict.index.max()
    date_difference = (end_date - start_date).days
    year_rate = (yield_rate / date_difference) * 365
    benchmark_yield = ((row['act_close'] / origin_price - 1) / date_difference) * 365

    # print('balance:' + str(balance) + ' yield rate:' + str(year_rate * 100) + '%')
    # print('benchmark yield:' + str(benchmark_yield * 100) + '%')

    # plt.figure(figsize=(24, 8))
    # plt.plot(yield_curve.index, yield_curve['return'] * 100, label='return')
    # plt.plot(yield_curve.index, yield_curve['benchmark'] * 100, label='benchmark')
    # plt.xlabel('Date')
    # plt.ylabel('year yield rate %')
    # plt.gcf().autofmt_xdate()
    # plt.title('return vs benchmark on BankofQingdao')
    # plt.legend()
    # plt.show()

    return yield_curve,balance,year_rate * 100,benchmark_yield * 100

def buy_and_sell_with_ma(whole_predict, logpath, balance=100000, fees=0, short_window=5, long_window=20):
    
    yield_curve = pd.DataFrame(index=whole_predict.index, columns=['balance', 'return', 'benchmark'])
    origin_balance = balance
    origin_price = whole_predict['act_close'].iloc[0]
    yesterday = whole_predict['act_close'].iloc[0]
    start = yesterday

    prices = whole_predict['act_close'].values
    for i, row in whole_predict.iterrows():
        today = row['pred_close']
        if today > yesterday:
            # 計算短期和長期移動平均線
            #short_ma = np.mean(prices[max(0, i - short_window + 1):i + 1])
            #long_ma = np.mean(prices[max(0, i - long_window + 1):i + 1])
            #short_ma = prices[max(0, i - short_window + 1):i + 1].mean()
            #print(i)
            #start_index = max(0, i - short_window + 1)
            start_index = max(pd.Timestamp('1970-01-01'), i - pd.Timedelta(days=short_window - 1))
            #print(start_index)
            #print(whole_predict['Date'] >= start_index)
            #end_index = i + 1
            end_index = i + pd.Timedelta(days=1)
            #print(end_index)
            #print(prices)
            mask = (whole_predict['Date'] >= start_index) & (whole_predict['Date'] <= end_index)
            subset = whole_predict.loc[mask]
            short_ma = subset['act_close'].mean()
            #print(short_ma)
            #short_ma = prices[start_index:end_index].mean()
            start_index_2 = max(pd.Timestamp('1970-01-01'), i - pd.Timedelta(days=long_window - 1))
            end_index_2 = i + pd.Timedelta(days=1)
            mask_2 = (whole_predict['Date'] >= start_index_2) & (whole_predict['Date'] <= end_index_2)
            subset_2 = whole_predict.loc[mask_2]
            long_ma = subset_2['act_close'].mean()
            #short_ma = np.mean(prices[max(0, i - pd.Timedelta(days=short_window - 1)):i + 1])
            #long_ma = np.mean(prices[max(0, i - pd.Timedelta(days=long_window - 1)):i + 1])
            #short_ma = np.mean(prices[max(0, i - pd.Timedelta(days=short_window - 1)):i + 1].values)
            #long_ma = np.mean(prices[max(0, i - pd.Timedelta(days=long_window - 1)):i + 1].values)
            #short_ma = prices[max(0, i - pd.Timedelta(days=short_window - 1)):i + 1].mean()
            #long_ma = prices[max(0, i - pd.Timedelta(days=long_window - 1)):i + 1].mean()
            if short_ma > long_ma:
                # 開盤價買入
                buy_num = balance // row['open']
                balance = balance - buy_num * row['open'] - buy_num * row['open'] * fees
                # 收盤價賣出
                balance = balance + buy_num * row['act_close']
                today_yield = (row['act_close'] - row['open']) * buy_num
                with open(logpath, "a") as file:
                    file.write('DATE:' + str(row['Date'].strftime("%Y-%m-%d")) + ' BUY ' + str(int(buy_num)) + ' at ' +
                               str(round(row['open'], 2)) + '  SELL at ' + str(round(row['act_close'], 2)) +
                               '  predicted close price=' + str(round(row['pred_close'], 2)) + '  todayield:' +
                               str(round(today_yield, 2)) + '  balance:' + str(round(balance, 2)) + "\n")
        yesterday = row['act_close']
        prices = np.append(prices, row['act_close'])
        yield_curve.loc[i, 'balance'] = balance
        date_difference = (row['Date'] - whole_predict.index.min()).days + 1
        yield_curve.loc[i, 'return'] = ((balance / origin_balance - 1) / date_difference) * 365
        yield_curve.loc[i, 'benchmark'] = ((row['act_close'] / origin_price - 1) / date_difference) * 365
    yield_rate = balance / origin_balance - 1
    start_date = whole_predict.index.min()
    end_date = whole_predict.index.max()
    date_difference = (end_date - start_date).days
    year_rate = (yield_rate / date_difference) * 365
    benchmark_yield = ((row['act_close'] / origin_price - 1) / date_difference) * 365
    return yield_curve, balance, year_rate * 100, benchmark_yield * 100