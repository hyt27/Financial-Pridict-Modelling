# Financial-Pridict-Modelling
My final project
## 主要idea
modelling产生predict，通过predict的值和实际的值表现出我们stategy的好处

## main 函数返回的东西
'''

    ########此文件供前后端直接调用，来对模型进行训练和预测
    # lstm_train()
    mse_train, mse_test, y_train, train_pred, y_test, test_pred, mtr = predict('randomforest')
  
    print('!MSE ON TRAIN:'+str(mse_train))
    print('!MSE ON TEST:'+str(mse_test))
  
'''
然后这个运行就是产生这样的图片，就是看你预测的和实际的区别，当然后续也有其他的功能但我还没开始。
![image](https://github.com/hyt27/Financial-Pridict-Modelling/assets/74773913/56dafe64-f50a-424f-acac-e8be44b70d89)
## 参数描述
predict函数不是返回了这么多参数吗，我把这些参数打印出来：

* mse_train: 0.00012435772731710378
  
* mse_test: 5.199904383605103e-05
  
* y_train:
 [[-0.83777453]
 [-0.87666532]
 [-0.868155  ]
 ...
 [-0.29274355]
 [-0.30126253]
 [-0.29158689]]

* train_pred: 
[-0.83458791 -0.86500523 -0.87256013 ... -0.2909271  -0.30271816 -0.29501827]

* y_test: 
[[-0.20591803]
 [-0.19976202]
 [-0.19323285]
 ...
 [-0.29274355]
 [-0.30126253]
 [-0.29158689]]
* test_pred: 
[-0.21340489 -0.20049463 -0.19272365 ... -0.2909271  -0.30271816
 -0.29501827]
* mtr: 
[[-0.79667562]
 [-0.79146965]
 [-0.79724886]
 ...
 [-0.29274355]
 [-0.30126253]
 [-0.29158689]]

### 分别代表
y_train：训练集的实际目标值。它是一个一维数组，包含训练数据的目标变量的真实值。

train_pred：训练集的预测目标值。它是一个一维数组，包含训练数据的目标变量的预测值。

y_test：测试集的实际目标值。它是一个一维数组，包含测试数据的目标变量的真实值。

test_pred：测试集的预测目标值。它是一个一维数组，包含测试数据的目标变量的预测值。

mtr：用于训练和测试的经过缩放和转换的数据。它是一个二维数组，表示输入特征。其中存储了经过缩放和转换后的时间序列数据，可以用于训练机器学习模型。每一行代表一个时间步，每一列代表一个特征。

#### mtr具体代码
df长这样

      Date Close                    
      2013-01-04  2276.991943
      2013-01-07  2285.364014
      2013-01-08  2276.070068
      2013-01-09  2275.340088
      2013-01-10  2283.656982
                      ...
      2022-12-26  3065.562988
      2022-12-27  3095.570068
      2022-12-28  3087.399902
      2022-12-29  3073.699951
      2022-12-30  3089.260010
      
      [2427 rows x 1 columns]

然后mtr经过归一化转换

        df = pd.read_csv(csv_file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        mtr = scaler.fit_transform(df.values)

# 关于strategy
在machinelearning_trading 中写一个策略
like
https://www.myquant.cn/docs/python_strategyies/112
1. 若没有仓位则在每个星期一的时候输入标的股票近15个交易日的特征变量进行预测,并在预测结果为上涨的时候购买标的.
2. 若已经持有仓位则在盈利大于10%的时候止盈,在星期五损失大于2%的时候止损.
3. 特征变量为:1.收盘价/均值2.现量/均量3.最高价/均价4.最低价/均价5.现量6.区间收益率7.区间标准差
然后在jpyter notebook那个文件中调用这个策略，
我不是有三个模型吗，调用我写的这个策略做图；现在是不用全部都改，只改一个就好了
所以实际上只改两个代码文件：一个策略以及在一个模型上的应用
输入：股票数据（字符串代码），starttime，endtime，选择的模型（String），策略(String)，账户初始金（int)
输出：预测图片，收益率图片，收益率，基准收益率，账户价值，日志信息
那个调用接口

### 我的策略
将5个交易日后的收盘价涨跌作为输出变量y，上涨为1，下跌为0。

在每个星期一的时候，使用最近15个交易日的特征变量进行预测。如果预测结果为上涨，则进行开仓操作购买标的股票。

设置止损止盈点：如果已经持有仓位，在盈利大于10%时止盈；
当时间为周五并且跌幅大于2%时,平掉所有仓位止损

### 我要用的model——SVM

    elif which_model == 'svm':
        lookback = 100   
        csv_file_path = 'SSE_Index.csv'  # 替换为你的 CSV 文件路径
        df = pd.read_csv(csv_file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        # 对数据进行标准化
        scaler = MinMaxScaler(feature_range=(-1, 1))
        mtr = scaler.fit_transform(df.values)

        # 构建训练集和测试集
        x_train, y_train, x_test, y_test = [], [], [], []
        for i in range(len(mtr) - lookback):
            x_train.append(mtr[i:i+lookback])
            y_train.append(mtr[i+lookback])
        for j in range(len(mtr) - lookback, len(mtr)):
            x_test.append(mtr[j-lookback:j])
            y_test.append(mtr[j])

        # 转换为 numpy 数组并重塑为二维数组
        x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

        # Load pre-trained SVM model
        model = torch.load('pthfile/svm_model.pth')

        # 进行预测
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
        
        # 计算均方误差
        mse_train = mean_squared_error(y_train, train_pred)
        mse_test = mean_squared_error(y_test, test_pred)
        
        # 打印均方误差
        print('MSE ON TRAIN:', mse_train)
        print('MSE ON TEST:', mse_test)

        # 绘制预测结果
        plt.plot(y_train, label='Actual Train')
        plt.plot(train_pred, label='Predicted Train')
        plt.legend()
        plt.title('SVM Predictions on Train Data')
        plt.show()
        
        plt.plot(y_test, label='Actual Test')
        plt.plot(test_pred, label='Predicted Test')
        plt.legend()
        plt.title('SVM Predictions on Test Data')
        plt.show()

        return mse_train, mse_test, y_train, train_pred, y_test, test_pred, mtr

# 模型
读取数据：从给定的数据路径（默认为 stock_data/0066.HK.csv）读取数据，并将日期列转换为日期时间格式，并设置日期为索引。然后根据指定的起始日期和结束日期筛选出需要预测的数据范围。

LSTM模型预测：如果选择的模型是'LSTM'，则进行LSTM模型的预测。
设置回溯大小（lookback）为50.
对收盘价进行MinMaxScaler标准化处理。
使用split_data函数将数据划分为训练集和测试集。
加载预训练的LSTM模型（保存在pthfile/lstm_model_complete.pth文件中）。
对数据进行逆标准化，以便将预测结果转换回原始范围。
使用模型对测试集进行预测，并将预测结果保存到test_pred中。
使用模型对训练集进行预测，并将预测结果保存到train_pred中。
创建用于绘制预测结果的DataFrame对象：testPredictPlot和trainPredictPlot。
创建包含整体预测序列的DataFrame对象：whole_predict。
计算训练集和测试集的均方误差（MSE）。

线性回归模型预测：如果选择的模型是'linearregression'，则进行线性回归模型的预测。
将数据向前移动一个时间步，以作为特征输入。
获取预测的特征数据（Open、High、Low、Volume和Close）和目标数据（Close）。
加载预训练的线性回归模型（保存在pthfile/linear_regression_model.joblib文件中）。
对特征数据进行预测，并计算预测结果和实际值之间的均方误差（MSE）。
创建包含预测结果和实际值的DataFrame对象：pred和actual。
创建包含整体预测序列的DataFrame对象：whole_predict。

返回结果：返回训练集和测试集的均方误差（mse_train和mse_test）、测试集的预测结果（testPredictPlot和trainPredictPlot）、原始数据（data_close）和整体预测序列（whole_predict）。

### 代码
代码试图访问whole_predict中的日期索引，以获取5个交易日后的预测收盘价。如果这个日期不存在，则向后顺延一位，直到找到日期

whole_predict = pd.DataFrame({'Date':data_to_be_predict.index,'pred_close':whole_predict.flatten(),'act_close':data_to_be_predict['Close'],'open':data_to_be_predict['Open']})
whole_predict = pd.DataFrame({'Date':data_to_be_predict.index,'pred_close':predict_result.flatten(),'open':data_to_be_predict['Open'],'act_close':data_to_be_predict['Close']})







