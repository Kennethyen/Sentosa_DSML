@[toc]
# 一、背景描述
&emsp;&emsp;股票价格是一种不稳定的时间序列,受多种因素的影响。影响股市的外部因素很多,主要有经济因素、政治因素和公司自身因素三个方面的情况。自股票市场出现以来,研究人员采用各种方法研究股票价格的波动。随着数理统计方法和机器学习的广泛应用,越来越多的人将机器学习等预测方法应用于股票预测中,如神经网络预测、决策树预测、支持向量机预测、逻辑回归预测等。
&emsp;&emsp;XGBoost是由TianqiChen在2016年提出来,并证明了其模型的计算复杂度低、运行速度快、准确度高等特点。XGBoost是GBDT的高效实现。在分析时间序列数据时,GBDT虽然能有效提高股票预测结果,但由于检测速率相对较慢,为寻求快速且精确度较高的预测方法,采用XGBoost模型进行股票预测,在提高预测精度同时也提高预测速率。可以利用XGBoost网络模型对股票历史数据的收盘价进行分析预测,将真实值和预测值进行对比,最后通过评估算子来评判XGBoost模型对股价预测的效果。
&emsp;&emsp;数据集通过爬虫获取从2005年开始到2020年的股票（代码为 510050.SH）历史数据，下表展示了股票在多个交易日内的市场表现，主要字段包括：

| 字段       | 含义                           |
|------------|--------------------------------|
| ts_code    | 股票代码                       |
| trade_date | 交易日期                       |
| pre_close  | 前一个交易日的收盘价             |
| open       | 开盘价                         |
| high       | 当日最高价                     |
| low        | 当日最低价                     |
| close      | 当日收盘价                     |
| change     | 收盘价变化值（与前一日相比的差值）|
| pct_chg    | 收盘价变化百分比               |
| vol        | 成交量                         |
| amount     | 成交金额                       |
| label      | 标记某日涨跌情况 |

&emsp;&emsp;这些字段全面记录了股票每天的价格波动和交易情况，用于后续分析和预测股票趋势。
# 二、Python代码和Sentosa_DSML社区版算法实现对比
## (一) 数据读入
1、python代码实现
&emsp;&emsp;导入需要的库
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
```
&emsp;&emsp;数据读入
```python
dataset = pd.read_csv('20_year_FD.csv')
print(dataset.head())
```
2、Sentosa_DSML社区版实现、

&emsp;&emsp;首先，利用文本算子从本地文件读入股票数据集。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5123282ad13a4afa9cec1b9763f5012e.png#pic_center)
## (二) 特征工程
1、python代码实现
```python
def calculate_moving_averages(dataset, windows):
    for window in windows:
        column_name = f'MA{window}'
        dataset[column_name] = dataset['close'].rolling(window=window).mean()
    dataset[['close'] + [f'MA{window}' for window in windows]] = dataset[['close'] + [f'MA{window}' for window in windows]].round(3)
    return dataset

windows = [5, 7, 30]
dataset = calculate_moving_averages(dataset, windows)

print(dataset[['close', 'MA5', 'MA7', 'MA30']].head())

plt.figure(figsize=(14, 7))
plt.plot(dataset['close'], label='Close Price', color='blue')
plt.plot(dataset['MA5'], label='5-Day Moving Average', color='red', linestyle='--')
plt.plot(dataset['MA7'], label='7-Day Moving Average', color='green', linestyle='--')
plt.plot(dataset['MA30'], label='30-Day Moving Average', color='orange', linestyle='--')
plt.title('Close Price and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/07c8170aa2124aa995c37cef0687c6b8.jpeg#pic_center.png =500x300)

得到实际股价与平均股价的差值的绝对值，观察偏离水平。

```python
def calculate_deviation(dataset, ma_column):
    deviation_column = f'deviation_{ma_column}'
    dataset[deviation_column] = abs(dataset['close'] - dataset[ma_column])
    return dataset

dataset = calculate_deviation(dataset, 'MA5')
dataset = calculate_deviation(dataset, 'MA7')
dataset = calculate_deviation(dataset, 'MA30')

plt.figure(figsize=(10, 6))
plt.plot(dataset['deviation_MA5'], label='Deviation from MA5')
plt.plot(dataset['deviation_MA7'], label='Deviation from MA7')
plt.plot(dataset['deviation_MA30'], label='Deviation from MA30')
plt.legend(loc='upper left')
plt.title('Deviation from Moving Averages')
plt.show()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/82534a1f42d743c286c9a01ae797b4a3.jpeg#pic_center.png =500x300)
```python
def calculate_vwap(df, close_col='close', vol_col='vol'):
    if close_col not in df.columns or vol_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{close_col}' and '{vol_col}' columns.")
    try:
        cumulative_price_volume = (df[close_col] * df[vol_col]).cumsum()
        cumulative_volume = df[vol_col].cumsum()
        vwap = np.where(cumulative_volume == 0, np.nan, cumulative_price_volume / cumulative_volume)
    except Exception as e:
        print(f"Error in VWAP calculation: {e}")
        vwap = pd.Series(np.nan, index=df.index)
    return pd.Series(vwap, index=df.index)
dataset['VWAP'] = calculate_vwap(dataset)
```
```python
def generate_signals(df, close_col='close', vwap_col='VWAP'):
    if close_col not in df.columns or vwap_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{close_col}' and '{vwap_col}' columns.")

    signals = pd.Series(0, index=df.index)

    signals[(df[close_col] > df[vwap_col]) & (df[close_col].shift(1) <= df[vwap_col].shift(1))] = 1  # 买入信号
    signals[(df[close_col] < df[vwap_col]) & (df[close_col].shift(1) >= df[vwap_col].shift(1))] = -1  # 卖出信号
    return signals

dataset['signal'] = generate_signals(dataset)
print(dataset[['close', 'VWAP', 'signal']].head())
```
2、Sentosa_DSML社区版实现
&emsp;&emsp;移动平均线是一种常用的技术指标，通过计算移动平均来分析股票的价格走势，帮助识别市场趋势，并为交易决策提供参考。根据不同的窗口大小（5天、7天、30天）来计算股票的收盘价的移动平均线，移动平均线可以平滑股价的短期波动，从而更好地识别股票的长期趋势。短期的 5 日、7 日移动平均线通常用来捕捉股票的短期趋势，帮助交易者快速做出买入或卖出的决策。30 日移动平均线则代表中长期趋势，帮助识别更广泛的市场方向。通过绘制图表，可以直观地看到收盘价格及其对应的移动平均线，方便观察价格变化和趋势。
&emsp;&emsp;利用生成列算子，通过设定的生成列表达式计算的新列的值，并设置列名，这里生成列分别为 moving_avg_5d、 moving_avg_7d、 moving_avg_30d，分别表示不同周期（5天、7天、30天）的移动平均线。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/263a0e001cd74945970a0503ab3d2de2.png#pic_center)
&emsp;&emsp;表达式为SQL窗口函数，
```sql
AVG(`close`) OVER ( ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)
AVG(`close`) OVER ( ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)
AVG(`close`) OVER ( ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b4de513452d34df1b17b1d151d6d266b.jpeg#pic_center)
&emsp;&emsp;连接折线图算子，选择收盘价实际值和移动平均线，进行图表展示。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0f974e84c71544b9ae58933e167799f4.png#pic_center)
&emsp;&emsp;得到结果如下，可以直观地看到收盘价格及其对应的移动平均线，方便观察价格变化和趋势。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ffa25f03d4284c458908ce45cc30ad5d.jpeg#pic_center)
&emsp;&emsp;再利用生成列算子，计算股票价格与不同周期的移动平均线的偏差的绝对值，得出当前价格偏离移动平均线的程度，观察偏离水平。偏差值越大，意味着价格波动越剧烈，可能处于较强的上涨或下跌趋势中。偏差值越小，意味着价格与均值靠近，波动较小，市场可能处于震荡或横盘阶段。
&emsp;&emsp;如果偏差持续扩大，说明价格远离均值，可能面临较大的回调风险或即将突破某个方向。
&emsp;&emsp;如果偏差开始收窄，说明价格回归均值，可能表明市场趋势趋于稳定或发生反转。
&emsp;&emsp;这里设置生成列列名分别为deviation_MA5、 deviation_MA7、deviation_MA30，分别表示不同周期得偏差。
&emsp;&emsp;生成列值得表达式如下：
```sql
abs(`close`-` moving_avg_5d`)
abs(`close`-` moving_avg_7d`)
abs(`close`-` moving_avg_29d`)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/eec43fd9b3904d888072d471c71d5725.png#pic_center)
&emsp;&emsp;右键生成列算子预览可以得到数据展示。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/24d018cb889f469fa21003745a7c16c2.png#pic_center)
&emsp;&emsp;或者利用图表算子对偏差值进行可视化图表展示，通过对偏差值进行可视化展示，绘制偏差曲线，可以直观呈现实际收盘价格与移动平均线之间的偏离趋势，不仅有助于揭示市场波动的幅度，还能为识别潜在的价格反转或趋势变化提供重要依据，能够更精准地判断市场的动向，从而优化决策流程并降低交易风险。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/dab9705a558f405186fed9d526badd48.jpeg#pic_center)
&emsp;&emsp;然后，基于交易量计算加权平均价格，反映特定时间段内股票的平均成交价格，考虑成交量的影响。计算公式是用股票的收盘价（close）乘以交易量（vol），然后计算加权收盘价的累积和，除以交易量的累积和。
&emsp;&emsp;利用生成列算子设置列名，并构造生成列表达式计算成交量加权平均值。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b09fd5bfe19945648eab1203a7801008.png#pic_center)
&emsp;&emsp;当股票的收盘价（close）大于成交量加权平均值时，signal 设置为 1，表示一个买入信号，股票价格处于强势。
&emsp;&emsp;当股票的收盘价小于等于成交量加权平均值时，signal 为 0，表示弱势，可以用于做空或保持观望。这个信号可以作为简单的策略来指导交易决策。
&emsp;&emsp;利用选择算子，对数据按照表达式`trade_date`;`close`>`成交量加权平均`对数据进行选择。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f1a17bcd086046b6b910db4a102dc0c4.png#pic_center)
&emsp;&emsp;并连接删除和重命名算子将进行条件判断后得列修改列名为signal，表示交易决策的指导信号。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/54032f4baf14431fab7b8a2c1a7271ee.png#pic_center)
&emsp;&emsp;再连接合并算子，将数据利用关键字trade_date将特征列进行合并。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7bcecbe8b1db41f88a52528d8c98a179.png#pic_center)
&emsp;&emsp;右键预览，可观察合并后的数据情况，也可以连接表格算子对数据进行表格输出。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/99477e4728b54c22909c0d5f70744b96.png#pic_center)
## (三) 样本分区
1、python代码实现
&emsp;&emsp;对数据进行预处理和顺序分区。
```python
def preprocess_data(dataset, columns_to_exclude, label_column):
    if label_column not in dataset.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset.")
    dataset[columns_to_exclude] = None

    for column in columns_to_convert:
        if column in dataset.columns:
            dataset[column] = pd.to_numeric(dataset[column], errors='coerce')
        else:
            print(f"Warning: Column '{column}' not found in dataset.")
    dataset.fillna(0, inplace=True)
    return dataset
```
```python
def split_data(dataset, label_column, train_ratio=0.8):
    dataset.sort_values(by='trade_date', ascending=True, inplace=True)
    split_index = int(len(dataset) * train_ratio)

    train_set = dataset.iloc[:split_index]
    test_set = dataset.iloc[split_index:]

    return train_set, test_set
```
```python
def prepare_dmatrix(train_set, test_set, label_column):
    if label_column not in train_set.columns or label_column not in test_set.columns:
        raise ValueError(f"Label column '{label_column}' must be in both training and testing sets.")

    dtrain = xgb.DMatrix(train_set.drop(columns=[label_column]), label=train_set[label_column])
    dtest = xgb.DMatrix(test_set.drop(columns=[label_column]), label=test_set[label_column])

    return dtrain, dtest
```

```python
columns_to_exclude = [
    'trade_date', 'ts_code', 'label', 'VWAP', 'signal',
    'MA5', 'MA7', 'deviation_MA5', 'deviation_MA7'
]
columns_to_convert = [
    'close', 'MA5', 'MA7', 'deviation_MA5',
    'deviation_MA7', 'MA30', 'deviation_MA30',
    'VWAP', 'signal'
]

label_column = 'close'
dataset = preprocess_data(dataset, columns_to_exclude, label_column)
train_set, test_set = split_data(dataset, label_column)
dtrain, dtest = prepare_dmatrix(train_set, test_set, label_column)
```
2、Sentosa_DSML社区版实现
&emsp;&emsp;在处理数据时，将trade_date列从int类型转换为datetime 类型，可以连接两个格式算子完成，首先将int类型的日期转换为字符串，然后再将字符串转换为datetime类型。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7246865a70794b2aaba8001fd91b0387.png#pic_center)![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/51aa815b07e64c61b274c51dac4e1f48.png#pic_center)
&emsp;&emsp;对数据输出类型进行格式化后，连接类型算子，设置数据的测量类型和模型类型。这里修改模型类型，设置建模算子输入数据需要的标签列和特征列等属性。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9a083f77badf4f5b9a5ecc90e5f7bf8f.png#pic_center)
&emsp;&emsp;然后，连接样本分区算子，利用时间序列对数据进行分区，训练集和测试集比例为8：2。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3abcb798f26940379eef872a80090c08.png#pic_center)
## (四) 模型训练和评估
1、python代码实现

```python
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 1,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 1,
    'colsample_bytree': 0.8,
    'lambda': 1,
    'alpha': 0
}
model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, 'test')])
y_train_pred = model.predict(dtrain)
y_test_pred = model.predict(dtest)
```

```python
def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
    mse = mean_squared_error(y_true, y_pred)
    return {
        'R2': r2,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'SMAPE': smape,
        'MSE': mse
    }
train_metrics = calculate_metrics(train_set[label_column], y_train_pred)
test_metrics = calculate_metrics(test_set[label_column], y_test_pred)
print("训练集评估结果:")
print(train_metrics)
print("测试集评估结果:")
print(test_metrics)
```

2、Sentosa_DSML社区版实现

&emsp;&emsp;首先，选择XGBoost回归算子，并设置了相关参数用于模型训练，使用均方根误差（RMSE）作为评估模型表现的指标。构建了一个XGBoost预测模型，并将其应用于股票收盘价预测。也可以连接其他回归模型进行训练，将XGBoost模型的预测结果与其他模型的预测结果进行比较，并通过模型评价指标（如R²、MAE、RMSE等）对各个模型的表现进行验证和评估。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cd5ce0a1f8c74f5fa5669069486e61e5.png#pic_center)
&emsp;&emsp;执行后可以得到训练完成的XGBoost回归模型，右键可进行查看模型信息和预览结果等操作。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/018be43378264e52a4b9bd3aceee0c5a.jpeg#pic_center)
&emsp;&emsp;连接评估算子对XGBoost模型进行评估。股票预测模型的预测性能评价指标采用R²、MAE、RMSE、MAPE、SMAPE和MSE，分别用于评估模型的拟合优度、预测误差的平均绝对值、均方根误差、绝对百分比误差、对称百分比误差和均方误差，用于衡量预测的准确性和稳定性。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/64a100ecc2b340ec80c9a57bb269025c.png#pic_center)
&emsp;&emsp;得到训练集和测试集的评估结果如下所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0e7cc3b8a74a40bca3a96d65dd4f3595.jpeg#pic_center)![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0bcd9e14d56245a8b2a30a0b67a7452c.jpeg#pic_center)
&emsp;&emsp;该XGBoost股票预测模型在训练集上表现优异，误差较小，表明模型能够很好地拟合训练数据。在测试集上的评估结果也较为理想，MAE为0.054，RMSE为0.093，MAPE和SMAPE分别为1.8%和1.7%，说明模型在测试集上的预测误差较小，具有良好的泛化能力，能够较为准确地预测股票收盘价，该模型在平衡训练集拟合和测试集泛化上表现稳定。
## (五) 模型可视化
1、python代码实现

```python
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['SimHei']

train_residuals = train_set[label_column] - y_train_pred

plt.figure(figsize=(10, 6))
xgb.plot_importance(model, importance_type='weight', title='特征重要性图', xlabel='重要性', ylabel='特征')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(train_residuals, bins=30, kde=True, color='blue')
plt.title('残差分布', fontsize=16)
plt.xlabel('残差', fontsize=14)
plt.ylabel('频率', fontsize=14)
plt.axvline(x=0, color='red', linestyle='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

```

```python
`if '预测值' in test_set.columns:
    test_data = pd.DataFrame(test_set.drop(columns=[label_column, '预测值']))
else:
    test_data = pd.DataFrame(test_set.drop(columns=[label_column]))

test_data['实际值'] = test_set[label_column].values
test_data['预测值'] = y_test_pred
test_data_subset = test_data.head(400)

original_values = test_data_subset['实际值'].values
predicted_values = test_data_subset['预测值'].values
x_axis = range(1, 401)

plt.figure(figsize=(12, 6))
plt.plot(x_axis, original_values, label='实际值', color='orange')
plt.plot(x_axis, predicted_values, label='预测值', color='green')
plt.title('实际值与预测值比较', fontsize=16)
plt.xlabel('样本编号', fontsize=14)
plt.ylabel('收盘价', fontsize=14)
plt.legend()
plt.grid()
plt.show()`
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/36a0a907f6e446278ff839fc81776972.jpeg#pic_center.png =500x300)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/db199c44526743baa72c56d5afabe7b5.jpeg#pic_center.png =500x300)

2、Sentosa_DSML社区版实现

&emsp;&emsp;右键模型信息可以查看特征重要性图、残差直方图等信息。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d35cbead872541009f3c3a2b4b1202c0.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/df33766c31d448e7beaed05b8fc35d49.jpeg#pic_center)
&emsp;&emsp;连接时序图算子，用于将XGBoost模型预测的股票收盘价与实际收盘价进行可视化对比，将每个序列单独显示，生成时序对比曲线图，通过这种方式可以直观地看到模型预测与实际数据的差异，从而评估模型的性能和可靠性。这在数据预测中非常重要，因为它有助于识别模型是否能够准确捕捉市场趋势。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d63fd99f949c4ad4a822e61b3bbc1f23.png#pic_center)
&emsp;&emsp;得到时序图算子的执行结果如下所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c8426e56e4c8489398b0695c8f4c6014.jpeg#pic_center)
&emsp;&emsp;这张图包含两条时间序列曲线，分别展示了模型预测值（Predicted_close）和实际值（close）在一段时间内的走势对比，显示的是模型预测的股票收盘价随时间变化的趋势。两条曲线的整体趋势相似，尤其是在大的波动区域（如2008年左右的高峰期和之后的下降期），表明模型的预测效果与实际值接近。这张图直观地展示了模型预测值与实际值的时间序列对比，帮助评估模型的表现是否符合实际市场走势。
# 三、总结
&emsp;&emsp;相比传统代码方式，利用Sentosa_DSML社区版完成机器学习算法的流程更加高效和自动化，传统方式需要手动编写大量代码来处理数据清洗、特征工程、模型训练与评估，而在Sentosa_DSML社区版中，这些步骤可以通过可视化界面、预构建模块和自动化流程来简化，有效的降低了技术门槛，非专业开发者也能通过拖拽和配置的方式开发应用，减少了对专业开发人员的依赖。
&emsp;&emsp;Sentosa_DSML社区版提供了易于配置的算子流，减少了编写和调试代码的时间，并提升了模型开发和部署的效率，由于应用的结构更清晰，维护和更新变得更加容易，且平台通常会提供版本控制和更新功能，使得应用的持续改进更为便捷。

&emsp;&emsp;Sentosa数据科学与机器学习平台（Sentosa_DSML）是力维智联完全自主知识产权的一站式人工智能开发部署应用平台，可同时支持零代码“拖拉拽”与notebook交互式开发，旨在通过低代码方式帮助客户实现AI算法模型的开发、评估与部署，结合完善的数据资产化管理模式与开箱即用的简捷部署支持，可赋能企业、城市、高校、科研院所等不同客户群体，实现AI普惠、化繁为简。
&emsp;&emsp;Sentosa_DSML产品由1+3个平台组成，以数据魔方平台（Sentosa_DC）为主管理平台，三大功能平台包括机器学习平台（Sentosa_ML）、深度学习平台（Sentosa_DL）和知识图谱平台（Sentosa_KG）。力维智联凭借本产品入选“全国首批人工智能5A等级企业”，并牵头科技部2030AI项目的重要课题，同时服务于国内多家“双一流”高校及研究院所。
&emsp;&emsp;为了回馈社会，矢志推动全民AI普惠的实现，不遗余力地降低AI实践的门槛，让AI的福祉惠及每一个人，共创智慧未来。为广大师生学者、科研工作者及开发者提供学习、交流及实践机器学习技术，我们推出了一款轻量化安装且完全免费的Sentosa_DSML社区版软件，该软件包含了Sentosa数据科学与机器学习平台（Sentosa_DSML）中机器学习平台（Sentosa_ML）的大部分功能，以轻量化一键安装、永久免费使用、视频教学服务和社区论坛交流为主要特点，同样支持“拖拉拽”开发，旨在通过零代码方式帮助客户解决学习、生产和生活中的实际痛点问题。
&emsp;&emsp;该软件为基于人工智能的数据分析工具，该工具可以进行数理统计与分析、数据处理与清洗、机器学习建模与预测、可视化图表绘制等功能。为各行各业赋能和数字化转型，应用范围非常广泛，例如以下应用领域：
&emsp;&emsp;金融风控：用于信用评分、欺诈检测、风险预警等，降低投资风险；
&emsp;&emsp;股票分析：预测股票价格走势，提供投资决策支持；
&emsp;&emsp;医疗诊断：辅助医生进行疾病诊断，如癌症检测、疾病预测等；
&emsp;&emsp;药物研发：进行分子结构的分析和药物效果预测，帮助加速药物研发过程；
&emsp;&emsp;质量控制：检测产品缺陷，提高产品质量；
&emsp;&emsp;故障预测：预测设备故障，减少停机时间；
&emsp;&emsp;设备维护：通过分析机器的传感器数据，检测设备的异常行为；
&emsp;&emsp;环境保护：用于气象预测、大气污染监测、农作物病虫害防止等；
&emsp;&emsp;客户服务：通过智能分析用户行为数据，实现个性化客户服务，提升用户体验；
&emsp;&emsp;销售分析：基于历史数据分析销量和价格，提供辅助决策；
&emsp;&emsp;能源预测：预测电力、天然气等能源的消耗情况，帮助优化能源分配和使用；
&emsp;&emsp;智能制造：优化生产流程、预测性维护、智能质量控制等手段，提高生产效率。

&emsp;&emsp;欢迎访问Sentosa_DSML社区版的官网https://sentosa.znv.com/，免费下载体验。同时，我们在B站、CSDN、知乎、博客园等平台有技术讨论博客和应用案例分享，欢迎广大数据分析爱好者前往交流讨论。

&emsp;&emsp;Sentosa_DSML社区版，重塑数据分析新纪元，以可视化拖拽方式指尖轻触解锁数据深层价值，让数据挖掘与分析跃升至艺术境界，释放思维潜能，专注洞察未来。
社区版官网下载地址：https://sentosa.znv.com/
社区版官方论坛地址：http://sentosaml.znv.com/
B站地址：https://space.bilibili.com/3546633820179281
CSDN地址：https://blog.csdn.net/qq_45586013?spm=1000.2115.3001.5343
知乎地址：https://www.zhihu.com/people/kennethfeng-che/posts
博客园地址：https://www.cnblogs.com/KennethYuen

</center>
<center class="half">
<a href ="https://sentosa.znv.com/"><img src="https://i-blog.csdnimg.cn/direct/5ad97144846d4bb5a9ea5dd3d4667e54.jpeg"></a>
</center>
