import pandas as pd
import pylab as plt
import datetime

######单均线策略
def single_moving_ave(data,logpath =  'log/no_name_log.txt',ave_day = 20, balance = 100000,fees = 0.0003,tax = 0.001):
    time_string = datetime.datetime.now()
    time_string = time_string.strftime("%Y-%m-%d_%H-%M-%S")
    filepath = logpath
    data['mov_ave'] = data['Close'].rolling(window = ave_day).mean()


    #####去掉前面没有均线的部分
    data = data.iloc[ave_day-1:]

    ###核心策略：当close上涨高于均线时买入，下跌到均线之下时卖出
    ####初始化
    
    yield_curve = pd.DataFrame(index=data.index, columns=['balance', 'return','benchmark'])
    origin_balance = balance      ###原始本金
    origin_price =  data['Close'].iloc[0]
    today_sell = False
    today_buy = False
    num_stock = 0  ##持仓数量
    origin_balance = balance      ###原始本金
    # today_buy = False
    hadstock = False  ###当前是否持仓

    for i,row in data.iterrows():

        today_sell = False
        today_buy = False

        ########盘前决策######################################
        
        if data.index.get_loc(i) > 0:    #跳过第一天因为没有前一天无法判断金叉

            if hadstock == False:      #未持仓
                if data.iloc[data.index.get_loc(i)-1]['Close'] < data.iloc[data.index.get_loc(i)-1]['mov_ave'] and row['Close'] >= row['mov_ave']:           #判断金叉
                    ###买入
                    today_buy = True
                if data.index.get_loc(i) == len(data)-1:   ###如果今天是最后一天则不买
                    today_buy = False
                
            elif hadstock == True:     #已经持仓
                if data.iloc[data.index.get_loc(i)-1]['Close'] > data.iloc[data.index.get_loc(i)-1]['mov_ave'] and row['Close'] <= row['mov_ave']:           #判断死叉   
                    ###卖出
                    today_sell = True

                if data.index.get_loc(i) == len(data)-1:   ###如果今天是最后一天则卖出
                    today_sell = True
                    
        ########盘前决策结束##################################
            


        ########日内交易##################################

            if today_buy == True:   ###执行买入
                buy_num = balance // (row['Open'] * (1+fees))    ##要买多少只
                balance = balance - buy_num * row['Open'] - buy_num * row['Open'] * fees  ###买完之后账户余额
                num_stock = buy_num
                hadstock = True

                with open(filepath, "a") as file:
                    file.write('DATE:'+ str(i)[:-9]+' BUY '+ str(int(buy_num)) + ' stocks at '+ str(round(row['Open'],2)) + '  balance:' + str(round(balance,2)) + "\n")  # 写入日志消息并添加换行符
            
            if today_sell == True:   ###执行卖出
                balance = balance + num_stock * row['Open'] - buy_num * row['Open'] * (tax+fees)  ###卖出后余额
                hadstock = False
                
                with open(filepath, "a") as file:
                    file.write('DATE:'+ str(i)[:-9]+' SELL '+ str(int(num_stock)) + ' stocks at '+ str(round(row['Open'],2)) + '  balance:' + str(round(balance,2)) + "\n")  # 写入日志消息并添加换行符
                
                num_stock = 0
        ########日内交易结束##################################





        #########盘后决策######################################


        #########盘后决策结束######################################

        #####记录每天余额和收益
        yield_curve.loc[i, 'balance'] = balance 
        date_difference = (i - data.index.min()).days + 1

        
        yield_curve.loc[i, 'return'] = (((balance+ num_stock * row['Close'])/origin_balance - 1)/date_difference) * 365

        yield_curve.loc[i, 'benchmark'] = ((row['Close']/origin_price - 1)/date_difference) * 365




    ##########循环结束#################

    yield_rate = balance/origin_balance-1


    start_date = data.index.min()
    end_date = data.index.max()

    # 计算日期差
    date_difference = (end_date - start_date).days
    year_rate = (yield_rate / date_difference) * 365

    benchmark_yield = ((row['Close']/origin_price - 1)/date_difference) * 365     
    # print('balance:' + str(balance) + ' yield rate:' + str(year_rate*100) + '%')
    # print('benchimark yield:'+ str(benchmark_yield*100) + '%')
    

    # ################画收益曲线
    # plt.figure(figsize=(24,8))

    # # plt.ylim(-100,200)

    # plt.plot(yield_curve.index,yield_curve['return']*100,label = 'return')
    # plt.plot(yield_curve.index,yield_curve['benchmark']*100,label = 'benchmark')

    # plt.xlabel('Date')
    # plt.ylabel('year yield rate %')
    
    
    # plt.gcf().autofmt_xdate()
    # plt.title('return  vs benchmark on BankofQingdao')
    # plt.legend()
    # plt.show()

    # ######画价格走势
    # plt.figure(figsize=(24,8))

    # # plt.ylim(-,200)

    # plt.plot(data.index,data['Close']*100,label = 'Close')
    # plt.plot(data.index,data['mov_ave']*100,label = 'mov_ave')

    # plt.xlabel('Date')
    # plt.ylabel('price')
    
    
    # plt.gcf().autofmt_xdate()
    # plt.title('Close and Mov_ave')
    # plt.legend()
    # plt.show()

    return yield_curve,balance+ num_stock * row['Close'],year_rate * 100,benchmark_yield * 100



######双均线策略
def double_moving_ave(data,logpath =  'log/no_name_log.txt',ave_day1 = 20,ave_day2 = 50,balance = 100000,fees = 0.0003,tax = 0.001):
    time_string = datetime.datetime.now()
    time_string = time_string.strftime("%Y-%m-%d_%H-%M-%S")
    filepath = logpath
    data['mov_ave1'] = data['Close'].rolling(window = ave_day1).mean()
    data['mov_ave2'] = data['Close'].rolling(window = ave_day2).mean()

    #####去掉前面没有均线的部分
    data = data.iloc[ave_day2-1:]

    ###核心策略：当close上涨高于均线时买入，下跌到均线之下时卖出
    ####初始化
    yield_curve = pd.DataFrame(index=data.index, columns=['balance', 'return','benchmark'])
    origin_balance = balance      ###原始本金
    origin_price =  data['Close'].iloc[0]
    today_sell = False
    today_buy = False
    num_stock = 0  ##持仓数量
    origin_balance = balance      ###原始本金
    # today_buy = False
    hadstock = False  ###当前是否持仓

    for i,row in data.iterrows():

        today_sell = False
        today_buy = False

        ########盘前决策######################################
        
        if data.index.get_loc(i) > 0:    #跳过第一天因为没有前一天无法判断金叉

            if hadstock == False:      #未持仓
                if data.iloc[data.index.get_loc(i)-1]['mov_ave1'] < data.iloc[data.index.get_loc(i)-1]['mov_ave2'] and row['mov_ave1'] >= row['mov_ave2']:           #判断金叉
                    ###买入
                    today_buy = True
                if data.index.get_loc(i) == len(data)-1:   ###如果今天是最后一天则不买
                    today_buy = False
                
            elif hadstock == True:     #已经持仓
                if data.iloc[data.index.get_loc(i)-1]['mov_ave1'] > data.iloc[data.index.get_loc(i)-1]['mov_ave2'] and row['mov_ave1'] <= row['mov_ave2']:           #判断死叉   
                    ###卖出
                    today_sell = True

                if data.index.get_loc(i) == len(data)-1:   ###如果今天是最后一天则卖出
                    today_sell = True
                    
        ########盘前决策结束##################################
            


        ########日内交易##################################

            if today_buy == True:   ###执行买入
                buy_num = balance // (row['Open'] * (1+fees))    ##要买多少只
                balance = balance - buy_num * row['Open'] - buy_num * row['Open'] * fees  ###买完之后账户余额
                num_stock = buy_num
                hadstock = True

                with open(filepath, "a") as file:
                    file.write('DATE:'+ str(i)[:-9]+' BUY '+ str(int(buy_num)) + ' stocks at '+ str(round(row['Open'],2)) + '  balance:' + str(round(balance,2)) + "\n")  # 写入日志消息并添加换行符
            
            if today_sell == True:   ###执行卖出
                balance = balance + num_stock * row['Open'] - buy_num * row['Open'] * (tax+fees)  ###卖出后余额
                hadstock = False
                
                with open(filepath, "a") as file:
                    file.write('DATE:'+ str(i)[:-9]+' SELL '+ str(int(num_stock)) + ' stocks at '+ str(round(row['Open'],2)) + '  balance:' + str(round(balance,2)) + "\n")  # 写入日志消息并添加换行符
                
                num_stock = 0
        ########日内交易结束##################################





        #########盘后决策######################################


        #########盘后决策结束######################################

        #####记录每天余额和收益
        yield_curve.loc[i, 'balance'] = balance 
        date_difference = (i - data.index.min()).days + 1

        
        yield_curve.loc[i, 'return'] = (((balance+ num_stock * row['Close'])/origin_balance - 1)/date_difference) * 365

        yield_curve.loc[i, 'benchmark'] = ((row['Close']/origin_price - 1)/date_difference) * 365




    ##########循环结束#################

    yield_rate = balance/origin_balance-1


    start_date = data.index.min()
    end_date = data.index.max()

    # 计算日期差
    date_difference = (end_date - start_date).days
    year_rate = (yield_rate / date_difference) * 365

    benchmark_yield = ((row['Close']/origin_price - 1)/date_difference) * 365     
    # print('balance:' + str(balance) + ' yield rate:' + str(year_rate*100) + '%')
    # print('benchimark yield:'+ str(benchmark_yield*100) + '%')
    

    ################画收益曲线
    # plt.figure(figsize=(24,8))

    # # plt.ylim(-100,200)

    # plt.plot(yield_curve.index,yield_curve['return']*100,label = 'return')
    # plt.plot(yield_curve.index,yield_curve['benchmark']*100,label = 'benchmark')

    # plt.xlabel('Date')
    # plt.ylabel('year yield rate %')
    
    
    # plt.gcf().autofmt_xdate()
    # plt.title('return  vs benchmark on BankofQingdao')
    # plt.legend()
    # plt.show()

    # ######画价格走势
    # plt.figure(figsize=(24,8))

    # # plt.ylim(-,200)

    # plt.plot(data.index,data['Close']*100,label = 'Close')
    # plt.plot(data.index,data['mov_ave1']*100,label = 'mov_ave1')
    # plt.plot(data.index,data['mov_ave2']*100,label = 'mov_ave2')
    # plt.xlabel('Date')
    # plt.ylabel('price')
    
    
    # plt.gcf().autofmt_xdate()
    # plt.title('Close and Mov_ave')
    # plt.legend()
    # plt.show()
    return yield_curve,balance+ num_stock * row['Close'],year_rate * 100,benchmark_yield * 100
#####布林带策略
def Bollinger_Bands(data,logpath =  'log/no_name_log.txt',period = 20,span = 2,balance = 100000,fees = 0.0003,tax = 0.001):
    time_string = datetime.datetime.now()
    time_string = time_string.strftime("%Y-%m-%d_%H-%M-%S")
    filepath = logpath


    data['middle_band'] = data['Close'].rolling(window = period).mean()
    data['STD'] = data['Close'].rolling(window = period).std()
    data['upper_band'] = data['middle_band'] + data['STD'] * span
    data['lower_band'] = data['middle_band'] - data['STD'] * span

    #####去掉前面没有计算的部分
    data = data.iloc[period-1:]

    ###核心策略：当价格冲破上带意味市场超买，是卖出信号，如果下跌突破下带则是买入信号
    ####初始化
    yield_curve = pd.DataFrame(index=data.index, columns=['balance', 'return','benchmark'])
    origin_balance = balance      ###原始本金
    origin_price =  data['Close'].iloc[0]
    today_sell = False
    today_buy = False
    num_stock = 0  ##持仓数量
    origin_balance = balance      ###原始本金
    # today_buy = False
    hadstock = False  ###当前是否持仓

    for i,row in data.iterrows():

        today_sell = False
        today_buy = False

        ########盘前决策######################################
        
        if data.index.get_loc(i) > 0:    #跳过第一天因为没有前一天无法判断金叉

            if hadstock == False:      #未持仓
                if data.iloc[data.index.get_loc(i)-1]['Close'] > data.iloc[data.index.get_loc(i)-1]['lower_band'] and row['Close'] <= row['lower_band']:           #突破下带
                    ###买入
                    today_buy = True
                if data.index.get_loc(i) == len(data)-1:   ###如果今天是最后一天则不买入
                    today_buy = False
                
            elif hadstock == True:     #已经持仓
                if data.iloc[data.index.get_loc(i)-1]['Close'] < data.iloc[data.index.get_loc(i)-1]['upper_band'] and row['Close'] >= row['upper_band']:           #突破上带   
                    ###卖出
                    today_sell = True

                if data.index.get_loc(i) == len(data)-1:   ###如果今天是最后一天则卖出
                    today_sell = True
                    
        ########盘前决策结束##################################
            


        ########日内交易##################################

            if today_buy == True:   ###执行买入
                buy_num = balance // (row['Open'] * (1+fees))    ##要买多少只
                balance = balance - buy_num * row['Open'] - buy_num * row['Open'] * fees  ###买完之后账户余额
                num_stock = buy_num
                hadstock = True

                with open(filepath, "a") as file:
                    file.write('DATE:'+ str(i)[:-9]+' BUY '+ str(buy_num) + ' stocks at '+ str(round(row['Open'],2)) + '  balance:' + str(balance) + "\n")  # 写入日志消息并添加换行符
            
            if today_sell == True:   ###执行卖出
                balance = balance + num_stock * row['Open'] - buy_num * row['Open'] * (tax+fees)  ###卖出后余额
                hadstock = False
                
                with open(filepath, "a") as file:
                    file.write('DATE:'+ str(i)[:-9]+' SELL '+ str(int(num_stock)) + ' stocks at '+ str(round(row['Open'],2)) + '  balance:' + str(round(balance,2)) + "\n")  # 写入日志消息并添加换行符
                
                num_stock = 0
        ########日内交易结束##################################





        #########盘后决策######################################


        #########盘后决策结束######################################

        #####记录每天余额和收益
        yield_curve.loc[i, 'balance'] = balance 
        date_difference = (i - data.index.min()).days + 1

        
        yield_curve.loc[i, 'return'] = (((balance+ num_stock * row['Close'])/origin_balance - 1)/date_difference) * 365

        yield_curve.loc[i, 'benchmark'] = ((row['Close']/origin_price - 1)/date_difference) * 365




    ##########循环结束#################

    yield_rate = balance/origin_balance-1


    start_date = data.index.min()
    end_date = data.index.max()

    # 计算日期差
    date_difference = (end_date - start_date).days
    year_rate = (yield_rate / date_difference) * 365

    benchmark_yield = ((row['Close']/origin_price - 1)/date_difference) * 365     
    # print('balance:' + str(balance) + ' yield rate:' + str(year_rate*100) + '%')
    # print('benchimark yield:'+ str(benchmark_yield*100) + '%')
    

    # ################画收益曲线
    # plt.figure(figsize=(24,8))

    # # plt.ylim(-100,200)

    # plt.plot(yield_curve.index,yield_curve['return']*100,label = 'return')
    # plt.plot(yield_curve.index,yield_curve['benchmark']*100,label = 'benchmark')

    # plt.xlabel('Date')
    # plt.ylabel('year yield rate %')
    
    
    # plt.gcf().autofmt_xdate()
    # plt.title('return  vs benchmark on BankofQingdao')
    # plt.legend()
    # plt.show()

   
   
    # ######画价格走势
    # plt.figure(figsize=(24,8))

    # # plt.ylim(-,200)

    # plt.plot(data.index,data['Close']*100,label = 'Close')
    # plt.plot(data.index,data['middle_band']*100,label = 'middle_band')
    # plt.plot(data.index,data['upper_band']*100,label = 'upper_band')
    # plt.plot(data.index,data['lower_band']*100,label = 'lower_band')

    # plt.xlabel('Date')
    # plt.ylabel('price')
    
    
    # plt.gcf().autofmt_xdate()
    # plt.title('price and band')
    # plt.legend()
    # plt.show()
    return yield_curve,balance+ num_stock * row['Close'],year_rate * 100,benchmark_yield * 100


#####动量选股
def momentum_strategy(stocks,logpath =  'log/no_name_log.txt',period = 20,startdate='2020-05-01',enddate='2024-04-30',balance = 100000,fees = 0.0003,tax = 0.001):
    #######每个period调整仓位，卖出原有股票，买入period内动量最大的两只
    time_string = datetime.datetime.now()
    time_string = time_string.strftime("%Y-%m-%d_%H-%M-%S")
    filepath = logpath
    basket = pd.DataFrame()
    momentum = pd.DataFrame()
    for stock_code in stocks:
        data = pd.read_csv('stock_data/'+stock_code+'.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date',inplace=True)
        data = data.loc[startdate:enddate]
        basket[stock_code+'_Close'] = data['Close']
        basket[stock_code+'_Open'] = data['Open']
        momentum[stock_code] = basket[stock_code+'_Close']/basket[stock_code+'_Close'].shift(20)

    basket = basket.iloc[period:]               #####篮子价格
    momentum = momentum.iloc[period:]             ####篮子动量

    ####初始化
    yield_curve = pd.DataFrame(index=data.index, columns=['balance', 'return','benchmark'])
    origin_balance = balance      ###原始本金
    origin_price =  basket.iloc[0].sum()
    selected_stock = ['null','null']
    num_stock = [0,0]
    hadstock = False
    origin_balance = balance      ###原始本金
    to_buy = []

    for i,row in basket.iterrows():

        today_sell = False
        today_buy = False

        ########盘前决策######################################
        if data.index.get_loc(i) % 20 == 0:
               

            ####决定要买的两只动量最大股票to_buy
            momentum_row= momentum.loc[i]
            to_buy = momentum_row.nlargest(2).index.tolist()
            # print(to_buy)
                    

                    
        ########盘前决策结束##################################
            


        ########日内交易##################################

        
            if hadstock == False:      ######如果当前空仓则直接买to_buy
                with open(filepath, "a") as file:
                    file.write('DATE:'+ str(i))  # 写入日志日期
                money_for_each_stock = balance / 2
                temp = 0
                x = 0
                for id in to_buy:               ##买入两只
                    
                    buy_num = money_for_each_stock // (row[id+'_Open'] * (1+fees))    ##要买多少只
                    remain_money = money_for_each_stock - buy_num * row[id+'_Open'] - buy_num * row[id+'_Open'] * fees  ###买完之后账户余额
                    temp = temp + remain_money
                    num_stock[x] = buy_num
                    selected_stock[x] = id
                    with open(filepath, "a") as file:
                        file.write(' buy '+to_buy[x]+' '+ str(buy_num) + ' at '+ str(row[id+'_Open']))  # 写入日志消息并添加换行符
                    x = x+1
                    

                balance = temp            ####余额
                with open(filepath, "a") as file:
                    file.write(' balance:'+ str(balance)+'\n')  # 写入日志消息并添加换行符
                hadstock = True


            elif hadstock == True:    #####持仓

                ####先卖出  
                with open(filepath, "a") as file:
                    file.write('DATE:'+ str(i))  # 写入日志日期
                x = 0
                for id in selected_stock:
                    balance = balance + num_stock[x] * row[id+'_Open'] - num_stock[x] * row[id+'_Open'] * (tax+fees)  ###卖出后余额
                    with open(filepath, "a") as file:
                        file.write('  sell '+ str(num_stock[x]) +' '+ selected_stock[x] + ' at '+ str(row[id+'_Open']) + ' ')  # 写入日志消息并添加换行符
                    x = x+1
                    
                selected_stock = ['null','null']             ####清空
                num_stock = [0,0]
                #####然后买
                money_for_each_stock = balance / 2
                temp = 0
                x = 0
                for id in to_buy:               ##买入两只
                    
                    buy_num = money_for_each_stock // (row[id+'_Open'] * (1+fees))    ##要买多少只
                    remain_money = money_for_each_stock - buy_num * row[id+'_Open'] - buy_num * row[id+'_Open'] * fees  ###买完之后账户余额
                    temp = temp + remain_money
                    num_stock[x] = buy_num
                    selected_stock[x] = id
                    with open(filepath, "a") as file:
                        file.write(' buy '+ str(buy_num) +' '+to_buy[x]+ ' at '+ str(row[id+'_Open']))  # 写入日志消息并添加换行符
                    x = x+1
                    
                balance = temp            ####余额
                with open(filepath, "a") as file:
                    file.write('balance:'+ str(balance)+'\n')  # 写入日志消息并添加换行符



            
                
                
        ########日内交易结束##################################





        #########盘后决策######################################


        #########盘后决策结束######################################

        #####记录每天余额和收益
        yield_curve.loc[i, 'balance'] = balance 
        date_difference = (i - data.index.min()).days + 1
        asset = balance
        if hadstock == True:
            for x in range(0,2):

                asset = asset + num_stock[x] * row[selected_stock[x]+'_Close']

        yield_curve.loc[i, 'return'] = (((asset)/origin_balance - 1)/date_difference) * 365

        
        yield_curve.loc[i, 'benchmark'] = ((row.sum()/origin_price - 1)/date_difference) * 365

        with open(filepath, "a") as file:
            file.write('DATE:'+ str(i) + ' asset:'+ str(asset)+'\n')  # 写入现有资产

    yield_rate = asset/origin_balance-1


    start_date = basket.index.min()
    end_date = basket.index.max()

    # 计算日期差
    date_difference = (end_date - start_date).days
    year_rate = (yield_rate / date_difference) * 365

    benchmark_yield = ((row.sum()/origin_price - 1)/date_difference) * 365     
    # print('asset:' + str(asset) + ' yield rate:' + str(year_rate*100) + '%')
    # print('benchimark yield:'+ str(benchmark_yield*100) + '%')
    # ################画收益曲线
    # plt.figure(figsize=(24,8))

    # # plt.ylim(-100,200)

    # plt.plot(yield_curve.index,yield_curve['return']*100,label = 'return')
    # plt.plot(yield_curve.index,yield_curve['benchmark']*100,label = 'benchmark')

    # plt.xlabel('Date')
    # plt.ylabel('year yield rate %')
    
    
    # plt.gcf().autofmt_xdate()
    # plt.title('return  vs benchmark on monmentum strategy')
    # plt.legend()
    # plt.show()

