@[toc]
# 一、Holt-Winters算法原理
什么是Holt-Winters预测算法？
&emsp;&emsp;Holt-Winters算法是一种时间序列预测方法。时间序列预测方法用于提取和分析数据和统计数据并表征结果，以便根据历史数据更准确地预测未来。Holt-Winters 预测算法允许用户平滑时间序列并使用该数据预测感兴趣的领域。指数平滑法会根据历史数据分配指数递减的权重和值，以降低较旧数据的权重值。换句话说，在预测中，较新的历史数据比较旧的结果具有更大的权重。
&emsp;&emsp;Holt-Winters中使用的指数平滑方法有三种：
&emsp;&emsp;单指数平滑——适用于预测没有趋势或季节性模式的数据，其中数据水平可能随时间而变化。
&emsp;&emsp;双重指数平滑法——用于预测存在趋势的数据。
&emsp;&emsp;三重指数平滑法——用于预测具有趋势和/或季节性的数据。
&emsp;&emsp;Holt-Winters包括预测方程和三个平滑方程，分别用于处理水平 $\ell_{t},$ 趋势 $b_{t}$ 和季节性成分 $s t$ ，对应的平滑参数分别是 $\alpha, \ \beta^{*}$ 和 $\gamma$ 。通常用 $m$ 表示季节性的周期，比如季度数据 $m=4$ ,月度数据 $m=1 2$ ,
&emsp;&emsp;Holt-Winters方法有两种变体，主要区别在于季节性成分的处理方式:
&emsp;&emsp;1. 加法模型：当季节性变化较为稳定时使用加法模型。
&emsp;&emsp;2. .乘法模型：当季节性变化与数据水平成比例变化时，适用乘法模型。
## (一) 加法模型
&emsp;&emsp;在加法模型中，季节性成分用绝对值来表示，并在水平方程中通过减去季节性成分来对数据进行季节性调整。每年内，季节性成分的和大约为零。加法模型的分量形式为：
$$\hat{y}_{t+h | t}=\ell_{t}+h b_{t}+s_{t+h-m ( k+1 )} $$
&emsp;&emsp;包含三个平滑方程，其中，水平方程是一个加权平均，包含季节性调整后的观察值 $( y_{t}-s_{t-m} )$ 和非季节性预测值$( \ell_{t-1}+b_{t-1} )$
$$\ell_{t}=\alpha( y_{t}-s_{t-m} )+( 1-\alpha) ( \ell_{t-1}+b_{t-1} ) $$
&emsp;&emsp;趋势方程与Holt的线性方法相同。
$$b_{t}=\beta^{*} ( \ell_{t}-\ell_{t-1} )+( 1-\beta^{*} ) b_{t-1} $$
&emsp;&emsp;季节性方程通过当前的季节性指数 $( y_{t}-\ell_{t-1}-b_{t-1} )$ 和上一年同一季节的季节性指数 $s_{t-m}$ 来平滑季节性成分。
$$s_{t}=\gamma( y_{t}-\ell_{t-1}-b_{t-1} )+( 1-\gamma) s_{t-m} $$
## (二) 乘法模型
&emsp;&emsp;在乘法模型中，季节性成分以相对值（百分比）表示，并通过将时间序列除以季节性成分来进行季节性调整。每年内，季节性成分的和约为 $m_{\circ}$ ，乘法模型的分量形式为：
$$\hat{y}_{t+h | t}=( \ell_{t}+h b_{t} ) s_{t+h-m ( k+1 )} $$
$$\ell_{t}=\alpha{\frac{y_{t}} {s_{t-m}}}+( 1-\alpha) ( \ell_{t-1}+b_{t-1} ) $$
$$b_{t}=\beta^{*} ( \ell_{t}-\ell_{t-1} )+( 1-\beta^{*} ) b_{t-1} $$
$$s_{t}=\gamma{\frac{y_{t}} {( \ell_{t-1}+b_{t-1} )}}+( 1-\gamma) s_{t-m} $$
## (三) 阻尼趋势
&emsp;&emsp;Holt-Winters 可以在加法和乘法季节性模型中引入阻尼（Damping）趋势。阻尼趋势能够使模型在预测未来趋势时更加稳健，避免趋势无限延伸，适用于那些趋势可能逐渐趋于稳定的时间序列数据，该方法结合了季节性和趋势的平滑，并通过阻尼因子 𝜙（0<𝜙<1） 控制趋势的持续性，将 𝜙 引入到趋势分量中，使得未来的趋势贡献逐渐减小。这样，随着预测期的增加，趋势的影响力会逐渐减弱，从而避免过度延伸。
&emsp;&emsp;结合了阻尼趋势的乘法季节性的预测方程为：
$$\hat{y}_{t+h | t}=\left[ \ell_{t}+( \phi+\phi^{2}+\cdots+\phi^{h} ) b_{t} \right] s_{t+h-m ( k+1 )} $$
$$\ell_{t}=\alpha\left( \frac{y_{t}} {s_{t-m}} \right)+\left( 1-\alpha\right) \left( \ell_{t-1}+\phi b_{t-1} \right) $$
$$b_{t}=\beta^{*} \left( \ell_{t}-\ell_{t-1} \right)+( 1-\beta^{*} ) \phi b_{t-1} $$
$$s_{t}=\gamma\left( {\frac{y_{t}} {\ell_{t-1}+\phi b_{t-1}}} \right)+( 1-\gamma) s_{t-m} $$
# 二、Holt Winters算法优缺点
## 优点
&emsp;&emsp;1、Holt-Winters 方法能够有效捕捉和建模时间序列中的季节性变化，适用于具有周期性波动的数据。
&emsp;&emsp;2、通过平滑参数的设置，Holt-Winters 方法能够动态调整对趋势和季节性的估&emsp;&emsp;计，适应时间序列数据的变化。
&emsp;&emsp;3、模型中包含的参数（水平、趋势、季节性）易于解释，便于理解时间序列的组成部分。
&emsp;&emsp;4、在短期预测方面，Holt-Winters 方法通常能提供较高的准确性。
## 缺点
&emsp;&emsp;平滑参数的选择对模型性能有很大影响，通常需要通过经验或交叉验证来优化这些参数，增加了模型设置的复杂性。
&emsp;&emsp;在长周期时间序列的预测中，Holt-Winters 方法可能会产生不切实际的趋势，特别是没有阻尼的情况下，可能导致长期预测的结果不稳定。

# 三、Python代码和Sentosa_DSML社区版算法实现对比
## (一) 数据读入和统计分析
1、python代码实现
```python
#导入需要的库
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import rcParams
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


file_path = r'.\每月香槟销量.csv'#文件路径
df = pd.read_csv(file_path, header=0)
print("原始数据前5行:")
print(df.head())

>>原始数据前5行:
     Month  Perrin Freres monthly champagne sales millions ?64-?72
0  1964-01                                             2815.0     
1  1964-02                                             2672.0     
2  1964-03                                             2755.0     
3  1964-04                                             2721.0     
4  1964-05                                             2946.0     

df = df.rename(columns={
    'Month': '月份',
    'Perrin Freres monthly champagne sales millions ?64-?72': '香槟销量'
})


print("\n修改列名后的数据前5行:")
print(df.head())

>>修改列名后的数据前5行:
        月份    香槟销量
0  1964-01  2815.0
1  1964-02  2672.0
2  1964-03  2755.0
3  1964-04  2721.0
4  1964-05  2946.0
```

&emsp;&emsp;完成数据读入后，对数据进行统计分析，统计数据分布图，计算每一列数据的极值、异常值等结果。代码如下：

```python
stats_df = pd.DataFrame(columns=[
    '列名', '数据类型', '最大值', '最小值', '平均值', '非空值数量', '空值数量',
    '众数', 'True数量', 'False数量', '标准差', '方差', '中位数', '峰度', '偏度',
    '极值数量', '异常值数量'
])

def detect_extremes_and_outliers(column, extreme_factor=3, outlier_factor=5):
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)
    iqr = q3 - q1
    
    lower_extreme = q1 - extreme_factor * iqr
    upper_extreme = q3 + extreme_factor * iqr
    lower_outlier = q1 - outlier_factor * iqr
    upper_outlier = q3 + outlier_factor * iqr
    
    extremes = column[(column < lower_extreme) | (column > upper_extreme)]
    outliers = column[(column < lower_outlier) | (column > upper_outlier)]
    
    return len(extremes), len(outliers)

for col in df.columns:
    col_data = df[col]

    dtype = col_data.dtype

    max_value = col_data.max() if np.issubdtype(dtype, np.number) else None
    min_value = col_data.min() if np.issubdtype(dtype, np.number) else None
    mean_value = col_data.mean() if np.issubdtype(dtype, np.number) else None
    non_null_count = col_data.count()
    null_count = col_data.isna().sum()
    mode_value = col_data.mode().iloc[0] if not col_data.mode().empty else None
    true_count = col_data[col_data == True].count() if dtype == 'bool' else None
    false_count = col_data[col_data == False].count() if dtype == 'bool' else None
    std_value = col_data.std() if np.issubdtype(dtype, np.number) else None
    var_value = col_data.var() if np.issubdtype(dtype, np.number) else None
    median_value = col_data.median() if np.issubdtype(dtype, np.number) else None
    kurtosis_value = col_data.kurt() if np.issubdtype(dtype, np.number) else None
    skew_value = col_data.skew() if np.issubdtype(dtype, np.number) else None

    extreme_count, outlier_count = detect_extremes_and_outliers(col_data) if np.issubdtype(dtype, np.number) else (None, None)

    new_row = pd.DataFrame({
        '列名': [col],
        '数据类型': [dtype],
        '最大值': [max_value],
        '最小值': [min_value],
        '平均值': [mean_value],
        '非空值数量': [non_null_count],
        '空值数量': [null_count],
        '众数': [mode_value],
        'True数量': [true_count],
        'False数量': [false_count],
        '标准差': [std_value],
        '方差': [var_value],
        '中位数': [median_value],
        '峰度': [kurtosis_value],
        '偏度': [skew_value],
        '极值数量': [extreme_count],
        '异常值数量': [outlier_count]
    })

    stats_df = pd.concat([stats_df, new_row], ignore_index=True)

print(stats_df)

>>     列名     数据类型      最大值     最小值  ...        峰度        偏度  极值数量 异常值数量
0    月份   object      NaN     NaN  ...       NaN       NaN  None  None
1  香槟销量  float64  13916.0  1413.0  ...  2.702889  1.639003     3     0


rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['SimHei']

output_dir = r'.\holtwinters'#选择路径

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for col in df.columns:
    plt.figure(figsize=(10, 6))
    df[col].dropna().hist(bins=30)
    plt.title(f"{col} - 数据分布图")
    plt.ylabel("频率")

    file_name = f"{col}_数据分布图.png"
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path)
    plt.close()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7dbfb3790b834694842cab789b16653b.png#pic_center)
2、Sentosa_DSML社区版实现

&emsp;&emsp;首先，进行数据读入，利用文本算子直接对数据进行读取，选择数据所在路径，
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/48ef9f2879384a9397da144d967f4068.png#pic_center)
&emsp;&emsp;同时，可以在文本算子的删除和重命名配置中修改列名或者删除列，这里将列明分别修改为'月份'和 '香槟销量'。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f52d87d9f15e446d99f1584a76e5d005.jpeg#pic_center)
&emsp;&emsp;点击应用，右击预览可以查看数据。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/71eebacc815045eba19c3cd7af2532e2.png)
&emsp;&emsp;接着，利用描述算子即可对数据进行统计分析，得到每一列数据的数据分布图、极值、异常值等结果。连接描述算子，右侧设置极值倍数为3，异常值倍数为5。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/96c5d6557f96448196231abffa40a0ec.jpeg#pic_center)
&emsp;&emsp;右击执行，可以得到结果如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/af6858694155409f82116dbcec067741.jpeg#pic_center)
## (二) 数据预处理
1、python代码实现

```python
#数据预处理
for col in df.columns:
    print(f"列名: {col}, 数据类型: {df[col].dtype}")

>>列名: 月份, 数据类型: object
列名: 香槟销量, 数据类型: float64

df = df.dropna()
df['月份'] = pd.to_datetime(df['月份'], format='%Y-%m', errors='coerce')  

df['香槟销量'] = pd.to_numeric(df['香槟销量'], errors='coerce') 
df = df.dropna(subset=['香槟销量'])
df['香槟销量'] = df['香槟销量'].astype(int)

for col in df.columns:
    print(f"列名: {col}, 数据类型: {df[col].dtype}")
    
print(df)
>>列名: 月份, 数据类型: datetime64[ns]
列名: 香槟销量, 数据类型: int32


filtered_df1 = df[df['月份'] <= '1971-09']
print(filtered_df1)
>>            月份  香槟销量
0   1964-01-01  2815
1   1964-02-01  2672
2   1964-03-01  2755
3   1964-04-01  2721
4   1964-05-01  2946

filtered_df2 = df[df['月份'] > '1971-09']
print(filtered_df2)

>>    月份   香槟销量
93  1971-10-01   6981
94  1971-11-01   9851
95  1971-12-01  12670
96  1972-01-01   4348
97  1972-02-01   3564

filtered_df1.set_index('月份', inplace=True)
resampled_df1 = filtered_df1['香槟销量'].resample('MS').bfill()

print(resampled_df1)

>>     月份   香槟销量
1964-01-01    2815
1964-02-01    2672
1964-03-01    2755
1964-04-01    2721
1964-05-01    2946
              ... 
1971-05-01    5010
1971-06-01    4874
1971-07-01    4633
1971-08-01    1659
1971-09-01    5951
```
2、Sentosa_DSML社区版实现

&emsp;&emsp;首先，连接格式算子对数据进行格式修改，将月份数据格式由String类型修改为Data类型。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a3dff209fb5845f1980d3627d91260ec.jpeg#pic_center)
&emsp;&emsp;其次，对数据进行过滤，将小于等于1971-09的数据作为训练和验证数据集，条件为大于1971-09的数据用于与时序预测数据做对比。可以利用两个过滤算子实现，算子右侧表格中属性“表达式”为spark sql表达式。
&emsp;&emsp;第一个过滤算子，条件为`月份`<='1971-09'，
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3d532ac68d94425380ff9242ab8c81f1.jpeg#pic_center)
&emsp;&emsp;第二个过滤算子条件为`月份`>'1971-09'，右击预览即可查看过滤数据。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cfe80c630614476ca4ce961254eacbed.jpeg#pic_center)
&emsp;&emsp;连接时序数据清洗算子，对用于模型训练的数据进行预处理，设置时间列为月份（时间列必须为Data/DataTime类型数据），选择采样频率使时间列数据时间相隔为1月，对香槟销量列以线性方式进行数据填充。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/63da8f55ff874fd3be22ec43bac0eb7e.jpeg#pic_center)
## (三) 模型训练和模型评估
1、python代码实现

```python
#模型定义
model = ExponentialSmoothing(
    resampled_df1, trend='add', seasonal='mul', seasonal_periods=12,damped_trend=True)
fit = model.fit(damping_slope=0.05)

#预测
forecast = fit.predict(
    start=len(resampled_df1), end=len(resampled_df1) + 11
)

residuals = resampled_df1 - fit.fittedvalues
residual_std = np.std(residuals)
upper_bound = forecast + 1.96 * residual_std
lower_bound = forecast - 1.96 * residual_std

results_df = pd.DataFrame({
    '预测值': forecast,
    '上限': upper_bound,
    '下限': lower_bound
})
print(results_df)
>> 月份            预测值            上限            下限
1971-10-01   7143.862498   8341.179324   5946.545672
1971-11-01  10834.141889  12031.458716   9636.825063
1971-12-01  13831.428845  15028.745671  12634.112019
1972-01-01   4054.821228   5252.138054   2857.504402
1972-02-01   3673.653407   4870.970233   2476.336580

#模型评估
y_true = resampled_df1.values
y_pred = fit.fittedvalues.values

def evaluate_model(y_true, y_pred, model_name="Holt-Winters"):
    r_squared = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    print(f"模型评估结果 ({model_name}):")
    print(f"{'-' * 40}")
    print(f"R² (决定系数): {r_squared:.4f}")
    print(f"MAE (平均绝对误差): {mae:.4f}")
    print(f"MSE (均方误差): {mse:.4f}")
    print(f"RMSE (均方根误差): {rmse:.4f}")
    print(f"{'-' * 40}\n")

    return {
        "R²": r_squared,
        "MAE": mae,
        "RMSE": rmse,
        "MSE": mse
    }

evaluation_results = evaluate_model(y_true, y_pred, model_name="Holt-Winters")

>>模型评估结果 (Holt-Winters):
----------------------------------------
R² (决定系数): 0.9342
MAE (平均绝对误差): 451.4248
MSE (均方误差): 402168.8567
RMSE (均方根误差): 634.1678
```
2、Sentosa_DSML社区版实现
&emsp;&emsp;在时序数据清洗算子后，连接HoltWinters算子，HoltWinters算子根据现有的时间序列对应的数据，预测未来时间的数据。算子的输入数据支持多种key键，但必须是满足相同key键下时间列间隔为固定数值，且数值列非空的时序数据，建议是时序数据清洗算子处理后的数据。
&emsp;&emsp;这里将时间列设为月份列，数据列设为香槟销量列，预测数量和周期性参数设置为12，分析频率为month，模型类型为Multiplicative，显著性水平alpha设置为0.05。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/07f49ff2da6c4152b723dfb7693c38ab.jpeg#pic_center)
&emsp;&emsp;模型连接时间序列模型评估算子，右击执行，可以查看评估结果。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4e49dcc60da84246bed4b58e757e59cb.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7381dc64820e40af9a6ae1cc98f0db9b.jpeg#pic_center)
## (四) 模型可视化
1、python代码实现

```python
#可视化
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['SimHei'] 

plt.figure(figsize=(12, 6))
plt.plot(resampled_df1, label='实际销量', color='blue')
plt.plot(fit.fittedvalues, label='拟合值', color='orange')
plt.plot(forecast, label='预测销量', color='green')
plt.title('Holt-Winters 方法预测香槟销量')
plt.xlabel('时间')
plt.ylabel('香槟销量')
plt.axvline(x=resampled_df1.index[-1], color='red', linestyle='--', label='预测起始点')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(resampled_df1.index, resampled_df1, label='实际值', color='blue')
plt.plot(results_df.index, results_df['预测值'], label='预测值', color='orange')
plt.fill_between(results_df.index, results_df['下限'], results_df['上限'], color='lightgray', alpha=0.5, label='95% 置信区间')
plt.title('Holt-Winters 预测与置信区间')
plt.xlabel('时间')
plt.ylabel('香槟销量')
plt.legend()
plt.show()

filtered_forecast_df = results_df[results_df.index > pd.Timestamp('1971-09-01')]
print(filtered_forecast_df)
>> 月份        预测值            上限            下限
1971-10-01   7143.862498   8341.179324   5946.545672
1971-11-01  10834.141889  12031.458716   9636.825063
1971-12-01  13831.428845  15028.745671  12634.112019
1972-01-01   4054.821228   5252.138054   2857.504402
1972-02-01   3673.653407   4870.970233   2476.336580


results_df = results_df.drop(columns=['上限', '下限'])
print(results_df)
>> 月份         预测值
1971-10-01   7143.862498
1971-11-01  10834.141889
1971-12-01  13831.428845
1972-01-01   4054.821228
1972-02-01   3673.653407
1972-03-01   4531.419772
1972-04-01   4821.096141

results_df.index.name = '月份'
merged_df = pd.merge(filtered_df2, results_df, left_on='月份', right_index=True, how='left')

print(merged_df)
>>         月份   香槟销量           预测值
93  1971-10-01   6981   7143.862498
94  1971-11-01   9851  10834.141889
95  1971-12-01  12670  13831.428845
96  1972-01-01   4348   4054.821228
97  1972-02-01   3564   3673.653407


scaler = StandardScaler()
merged_df[['香槟销量', '预测值']] = scaler.fit_transform(merged_df[['香槟销量', '预测值']])

plt.figure(figsize=(12, 6))
plt.plot(merged_df['月份'], merged_df['香槟销量'], label='香槟销量', color='blue')
plt.plot(merged_df['月份'], merged_df['预测值'], label='香槟预测销量', color='orange')
plt.title('时序图')
plt.xlabel('时间')
plt.ylabel('香槟销量')
plt.legend()
plt.show()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/708c4c76ce3a4d2dab15046819d1d5cd.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/508eb4e2576f4f9c84d025bf1a1b706b.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/285f05fbebfa43af9619dfe5235eb2ef.png)

2、Sentosa_DSML社区版实现

&emsp;&emsp;为了对比原始数据和预测数据，首先，利用过滤算子对HoltWinters模型预测数据进行过滤，过滤条件为`月份`>'1971-09'。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ab43aaf3489d4998b3761a4174f62df6.jpeg#pic_center)
&emsp;&emsp;右击预览可以查看数据过滤结果。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3ac94d06eaeb4577800d0e8216363be1.jpeg#pic_center)
&emsp;&emsp;其次，连接删除和重命名算子，将需要的时间列和预测结果列保留，其余列删除。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7bf593cae7b748ee9cfc827beb8d24e3.jpeg#pic_center)
&emsp;&emsp;应用完成后右击即可查看处理结果。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4b757129d981476e9997561b69accfc5.jpeg#pic_center)
&emsp;&emsp;然后，连接合并算子，将原始数据和预测数据进行合并，分为关键字合并和顺序合并两种，这里使用关键字合并，用于合并的关键字为月份列，合并方式选择左连接。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b91757f505e34dc9a64d151122635414.jpeg#pic_center)
&emsp;&emsp;右击预览可以得到合并算子的处理结果。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/713263ee3268471e91ef9ecb42623285.jpeg#pic_center)
&emsp;&emsp;再连接图表分析中的时序图算子，“序列”可以选择多列，当序列为多列时需要配置“每个序列是否单独显示”，
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a33bde0046ed41e59a2d399eb6eddb17.png#pic_center)
&emsp;&emsp;右击执行后可以得到可视化结果，右上方可以进行下载等操作，鼠标移动可以查看当前位置的数据信息，下方可以滑动调整数据的时序区间。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/872d9969e732432a9086b577586bc400.jpeg#pic_center)
&emsp;&emsp;对于HoltWinters模型的预测结果，直接连接时序图算子进行图表分析，采用序列模式，对香槟销量实际值和预测值进行对比。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/13b09e4337dd4573b3edf104e322288c.png#pic_center)
&emsp;&emsp;右击执行得到结果如下所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d5a3308332c54294b8abb6e889c7638b.jpeg#pic_center)
&emsp;&emsp;采用时间序列模型模式对于HoltWinters模型的预测结果进行图表分析，属性设置如右侧所示。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3c90daa9c5dd497bb30092470d0dc09a.png#pic_center)
&emsp;&emsp;右击执行得到结果，其中，实心点数据表示原始真实值，实线表示对原始数据的拟合数据，空心虚线表示预测数据，阴影边界的上下虚线分别表示置信区间的预测上限和下限。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e4894951d8a643e3b64085dbcf469c94.jpeg#pic_center)
# 四、总结
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
