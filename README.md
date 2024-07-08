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

# 期中總結
我做的工作：
1.寫了三個模型的代碼，通過歷史數據訓練random forest，SVM ，GRU
2.將模型保存到pth文件裏面
3.創建一個接口 predict.py 通過所要用的數據輸入模型中得到預測的結果，并且返回誤差和畫圖表示
4.創造一個策略，利用預測出來的五天後的close值與今天的close值對比，如果
5.利用另一個接口，這個接口接受用戶所選擇的股票名稱，時間段，所要用的策略，以及機器學習預測的模型，然後在這個環境下調用模型，通過策略做出買賣股票的選擇存入log中
##
在期中报告中，我进行了以下工作：

模型开发：
针对股票预测任务，我实现了三种不同的机器学习模型：随机森林(Random Forest)、支持向量机(SVM)和门控循环单元(GRU)。
这些模型分别具有不同的特点和优势，通过对历史数据的训练，它们能够学习到股票价格的趋势和模式。
为了确保模型的可复用性，我将训练好的模型保存到.pth文件中，以便在后续的预测任务中直接加载使用。
创建预测接口：
我编写了一个名为predict.py的接口文件，其中包含了一个函数，可以通过输入数据来调用训练好的模型进行预测。
该接口函数接受股票数据作为输入，然后使用所选的模型对未来的股票价格进行预测。
预测结果不仅包括预测的股票价格，还包括与实际价格的误差评估指标，例如均方误差(MSE)。
为了更直观地展示预测结果，我使用绘图库将预测结果可视化，以便对比预测值和实际值。
策略开发：
为了更好地利用预测结果，我设计了一种简单的策略，基于预测的五天后的收盘价与当天收盘价的对比。
如果预测值较高，意味着股票可能上涨，可以考虑买入股票；如果预测值较低，意味着股票可能下跌，可以考虑卖出股票。
这种策略是基于预测结果的简单规则，可以作为决策的参考，但需要进一步优化和改进以提高准确性。
创建交互接口：
我进一步实现了一个交互接口，该接口可以根据用户的选择调用预测模型和策略。
用户可以选择股票名称、时间段、使用的模型和策略，系统将根据用户的选择进行预测和交易决策。
交易决策的结果将被记录在日志中，以便后续分析和回顾。
通过以上工作，我已经建立了一个基本的股票预测和交易系统，并实现了模型训练、预测接口、策略开发和交互接口等关键功能。在系统的后续开发中，我将进一步优化模型的性能和准确性，改进交易策略的智能化程度，并增加更多的功能和用户体验，以实现一个完整的股票预测和交易平台。



### GRU
这段代码实现了使用 GRU（门控循环单元）模型进行时间序列预测的过程。下面是代码的主要步骤和功能：

数据准备：将输入数据按照给定的滑动窗口大小进行切分，形成训练集和测试集。
加载数据：从 CSV 文件中读取数据，并进行日期处理和索引设置。
数据预处理：使用 MinMaxScaler 对数据进行归一化处理，将数值范围缩放到[-1, 1]之间。
定义超参数：设置模型的输入维度、隐藏层维度、输出维度、层数、滑动窗口大小和训练参数等。
准备训练集和测试集：根据滑动窗口大小和切分后的数据，构建训练集和测试集。
转换为张量：将训练集和测试集转换为 PyTorch 张量。
定义模型：使用 nn.GRU 和 nn.Linear 构建 GRU 模型。
定义损失函数和优化器：使用均方误差（MSE）作为损失函数，使用 Adam 优化器进行模型参数优化。
训练模型：使用训练集进行模型训练，通过反向传播更新模型参数。
模型评估：在训练和测试集上进行模型预测，并计算均方误差。
绘制预测结果：使用 Matplotlib 绘制训练集和测试集的实际值与预测值的对比图。
保存模型：将训练好的模型保存为 .pth 文件。
加载模型：从保存的 .pth 文件加载模型。
使用加载的模型进行预测：使用加载的模型对测试集进行预测。
绘制加载的模型的预测结果：使用 Matplotlib 绘制加载的模型在测试集上的预测结果。
总体而言，这段代码通过 GRU 模型对时间序列数据进行预测，并提供了训练、评估、保存和加载模型的功能。

### predict_v2.py
给定的代码定义了一个名为 predict 的函数，它接受一个参数 which_model 来指定要使用哪个模型进行预测。该函数使用不同的模型（随机森林、支持向量机和GRU）进行时间序列预测。

以下是代码对每个模型的操作摘要：

随机森林：
从Yahoo Finance加载股票数据，获取'0066.HK'股票的数据，时间范围从2010年1月1日到2020年6月30日。
使用MinMaxScaler对数据进行缩放。
准备训练集和测试集。
从文件中加载预训练的随机森林模型。
对训练集和测试集进行预测。
计算训练集和测试集的均方误差（MSE）。
返回训练集和测试集的MSE值、实际值和预测值，以及缩放后的数据。
支持向量机（SVM）：
从CSV文件加载股票数据。
使用MinMaxScaler对数据进行缩放。
准备训练集和测试集。
从文件中加载预训练的SVM模型。
对训练集和测试集进行预测。
计算训练集和测试集的均方误差（MSE）。
绘制训练集和测试集的实际值和预测值。
返回训练集和测试集的MSE值、实际值和预测值，以及缩放后的数据。
GRU（门控循环单元）：
从CSV文件加载股票数据。
使用MinMaxScaler对数据进行缩放。
准备训练集和测试集。
从文件中加载预训练的GRU模型。
对训练集和测试集进行预测。
将预测值逆向缩放回原始比例。
绘制训练集和测试集的实际值和预测值。
计算训练集和测试集的均方误差（MSE）。
返回训练集和测试集的MSE值、实际值和预测值，以及缩放后的数据。
代码使用了多个库，如pandas、numpy、scikit-learn和torch，用于数据处理、模型训练和预测。代码还包括了一些被注释掉的绘制预测值的代码，目前被禁用了。

注意：代码假设预训练的模型保存在特定的文件路径中，并且模型的数据是从CSV文件或Yahoo Finance加载的。



