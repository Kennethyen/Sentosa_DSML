
@[toc]
# 一、算法和背景介绍
&emsp;&emsp;关于XGBoost的算法原理，已经进行了介绍与总结，相关内容可参考[【机器学习(一)】分类和回归任务-XGBoost算法-Sentosa_DSML社区版](https://blog.csdn.net/qq_45586013/article/details/142068169?spm=1001.2014.3001.5502)一文。本文以预测二手车的交易价格为目标，通过Python代码和Sentosa_DSML社区版分别实现构建XGBoost回归预测模型，并对模型进行评估，包括评估指标的选择与分析。最后得出实验结论，确保模型在二手汽车价格回归预测中的有效性和准确性。

**数据集介绍**
&emsp;&emsp;以预测二手车的交易价格为任务，数据来自某交易平台的二手车交易记录，总数据量超过40w，包含31列变量信息，其中15列为匿名变量。数据集概况介绍：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d0a608e1b48045149301d90d6507d6d1.png)
# 二、Python代码和Sentosa_DSML社区版算法实现对比
## (一) 数据读入与统计分析
1、python代码实现

```python
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import rcParams
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```
&emsp;&emsp;数据读入

```python
file_path = r'.\二手汽车价格.csv'
output_dir = r'.\xgb'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"文件未找到: {file_path}")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
df = pd.read_csv(file_path)

print(df.isnull().sum())
print(df.head())
>>   SaleID    name   regDate  model  ...      v_11      v_12      v_13      v_14
0       0     736  20040402   30.0  ...  2.804097 -2.420821  0.795292  0.914763
1       1    2262  20030301   40.0  ...  2.096338 -1.030483 -1.722674  0.245522
2       2   14874  20040403  115.0  ...  1.803559  1.565330 -0.832687 -0.229963
3       3   71865  19960908  109.0  ...  1.285940 -0.501868 -2.438353 -0.478699
4       4  111080  20120103  110.0  ...  0.910783  0.931110  2.834518  1.923482
```
&emsp;&emsp;统计分析

```python
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['SimHei']

stats_df = pd.DataFrame(columns=[
    '列名', '数据类型', '最大值', '最小值', '平均值', '非空值数量', '空值数量',
    '众数', 'True数量', 'False数量', '标准差', '方差', '中位数', '峰度', '偏度',
    '极值数量', '异常值数量'
])

def detect_extremes_and_outliers(column, extreme_factor=3, outlier_factor=5):
    if not np.issubdtype(column.dtype, np.number):
        return None, None
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
    if np.issubdtype(dtype, np.number):
        max_value = col_data.max()
        min_value = col_data.min()
        mean_value = col_data.mean()
        std_value = col_data.std()
        var_value = col_data.var()
        median_value = col_data.median()
        kurtosis_value = col_data.kurt()
        skew_value = col_data.skew()
        extreme_count, outlier_count = detect_extremes_and_outliers(col_data)
    else:
        max_value = min_value = mean_value = std_value = var_value = median_value = kurtosis_value = skew_value = None
        extreme_count = outlier_count = None

    non_null_count = col_data.count()
    null_count = col_data.isna().sum()
    mode_value = col_data.mode().iloc[0] if not col_data.mode().empty else None
    true_count = col_data[col_data == True].count() if dtype == 'bool' else None
    false_count = col_data[col_data == False].count() if dtype == 'bool' else None

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

def save_plot(filename):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = os.path.join(output_dir, f"{filename}_{timestamp}.png")
    plt.savefig(file_path)
    plt.close()

for col in df.columns:
    plt.figure(figsize=(10, 6))
    df[col].dropna().hist(bins=30)
    plt.title(f"{col} - 数据分布图")
    plt.ylabel("频率")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f"{col}_数据分布图_{timestamp}.png"
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path)
    plt.close()

if 'car_brand' in df.columns and 'price' in df.columns:
    grouped_data = df.groupby('car_brand')['price'].count()
    plt.figure(figsize=(8, 8))
    plt.pie(grouped_data, labels=grouped_data.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    plt.title("品牌和价格分布饼状图", fontsize=16)
    plt.axis('equal')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f"car_brand_price_distribution_{timestamp}.png"
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path)
    plt.close()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4d6bde2810d84979aee7c232ca50fd6f.jpeg#pic_center.png =500x300)
2、Sentosa_DSML社区版实现
&emsp;&emsp;首先，进行数据读入，利用文本算子直接对数据进行读取，右侧进行读取属性配置
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b56080930e014261aa20b8d84cca3621.png#pic_center)
&emsp;&emsp;接着，利用描述算子即可对数据进行统计分析，得到每一列数据的数据分布图、极值、异常值等结果。连接描述算子，右侧设置极值倍数为3，异常值倍数为5。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a058fdd311c540639090122aff8163e3.png#pic_center)
&emsp;&emsp;右击执行，得到数据统计分析结果，可以对数据每一列的数据分布图、存储类型，最大值、最小值、平均值、非空值数量、空值数量、众数、中位数、极值和异常值数量等进行计算并展示，结果如下所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bdddf7f8687f4abead3c2a8a33207573.jpeg#pic_center)
&emsp;&emsp;描述算子执行结果有助于我们对于数据的理解和后续分析。
## (二) 数据处理

1、python代码实现
&emsp;&emsp;进行数据处理操作
```python
def handle_power(power, threshold=600, fill_value=600):
    return fill_value if power > threshold else power

def handle_not_repaired_damage(damage, missing_value='-', fill_value=0.0):
    return fill_value if damage == missing_value else damage

def extract_date_parts(date, part):
    if part == 'year':
        return str(date)[:4]
    elif part == 'month':
        return str(date)[4:6]
    elif part == 'day':
        return str(date)[6:8]

def fix_invalid_month(month, invalid_value='00', default='01'):
    return default if month == invalid_value else month

columns_to_fill_with_mode = ['model', 'bodyType', 'fuelType', 'gearbox']

for col in columns_to_fill_with_mode:
    mode_value = df[col].mode().iloc[0]
    df[col].fillna(mode_value, inplace=True)

df = (
    df.fillna({
        'model': df['model'].mode()[0],
        'bodyType': df['bodyType'].mode()[0],
        'fuelType': df['fuelType'].mode()[0],
        'gearbox': df['gearbox'].mode()[0]
    })
    .assign(power=lambda x: x['power'].apply(handle_power).fillna(600))
    .assign(notRepairedDamage=lambda x: x['notRepairedDamage'].apply(handle_not_repaired_damage).astype(float))
    .assign(
        regDate_year=lambda x: x['regDate'].apply(lambda y: str(extract_date_parts(y, 'year'))),
        regDate_month=lambda x: x['regDate'].apply(lambda y: str(extract_date_parts(y, 'month'))).apply(
            fix_invalid_month),
        regDate_day=lambda x: x['regDate'].apply(lambda y: str(extract_date_parts(y, 'day')))
    )

    .assign(
        regDate=lambda x: pd.to_datetime(x['regDate_year'] + x['regDate_month'] + x['regDate_day'],
                                         format='%Y%m%d', errors='coerce'),
        creatDate=lambda x: pd.to_datetime(x['creatDate'].astype(str), format='%Y%m%d', errors='coerce')
    )
    .assign(
        car_day=lambda x: (x['creatDate'] - x['regDate']).dt.days,
        car_year=lambda x: (x['car_day'] / 365).round(2) 
    )
    .assign(log1p_price=lambda x: np.log1p(x['price']))
)
print(df.head())
>>   SaleID    name    regDate  model  ...  regDate_day  car_day  car_year  log1p_price
0       0     736 2004-04-02   30.0  ...           02     4385     12.01     7.523481
1       1    2262 2003-03-01   40.0  ...           01     4757     13.03     8.188967
2       2   14874 2004-04-03  115.0  ...           03     4382     12.01     8.736007
3       3   71865 1996-09-08  109.0  ...           08     7125     19.52     7.783641
4       4  111080 2012-01-03  110.0  ...           03     1531      4.19     8.556606

print(df.dtypes)
>>SaleID                        int64
name                          int64
regDate              datetime64[ns]
model                       float64
brand                         int64
bodyType                    float64
fuelType                    float64
gearbox                     float64
power                         int64
kilometer                   float64
notRepairedDamage           float64
regionCode                    int64
seller                        int64
offerType                     int64
creatDate            datetime64[ns]
price                         int64
v_0                         float64
v_1                         float64
...
```

2、Sentosa_DSML社区版实现
&emsp;&emsp;通过描述算子的执行结果可以观察到，"model"，"bodyType"，"fuelType","gearbox","power","notRepairedDamage"列需要进行缺失值和异常值处理，首先，连接异常值缺失值填充算子，点击配置列选择，选择需要进行异常值缺失值处理的列。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f617987e30324a0989788d56b19006e1.png#pic_center)
&emsp;&emsp;然后，对配置列异常值缺失值填充方式进行选择。"model"，"bodyType"，"fuelType","gearbox"列选择保留异常值，利用众数填充缺失值，
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9cc89bc5996e4eb89af0fd1bc7f5237b.jpeg#pic_center)
&emsp;&emsp;"power"列选择输入规则处理异常值，指定异常值的检测规则为‘`power`>600’，选择按照缺失值方法进行填充，使用固定值600对缺失值进行填充。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/be8a829af8764c748136408194106495.jpeg#pic_center)&emsp;&emsp;"notRepairedDamage"列选择输入规则处理异常值，指定异常值的检测规则为`notRepairedDamage`== '-'，选择按照缺失值方法进行填充，使用固定值0.0对缺失值进行填充。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3a28f5b32ef64161b4f58a05075615c7.jpeg#pic_center)
&emsp;&emsp;然后，利用生成列算子分别提取年，月和日信息，并生成对应的列，生成年，月和日列的表达式分别为：substr(`regDate`,1,4)、substr(`regDate`,5,2)、substr(`regDate`,7,2)。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/50f402b9459d4464adbe432948a56f26.png#pic_center)
&emsp;&emsp;生成列算子处理完成后的结果如下所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1be060b0a9f74e86ac92cf274f25b59a.jpeg#pic_center)
&emsp;&emsp;为了处理无效或缺失的月份信息，利用填充算子对月份列`regDate_月`进行处理，填充条件为`regDate_月 == '00'` ，使用填充值 '01'对 `regDate` 列进行填充。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/06b43df282564f7f8b48295a62a49837.png#pic_center)
&emsp;&emsp;对于有效的regDate列数据，利用concat(regDate_年, regDate_月, regDate_日) 来填充regDate列。通过从有效的年、月、日列填充一个新的regDate，可以修复原始数据中某些部分不完整或异常的情况。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/055400ad68304a6488970da0237a7e0f.png#pic_center)
&emsp;&emsp;通过格式算子将‘regDate’和‘creatData’列修改为String类型（Intege不能直接修改为Date类型），将‘notRepairedDamage’列修改为Double类型。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/870bbc22d34b44e9aa85c1aa3baa5f1f.png#pic_center)
&emsp;&emsp;将‘regDate’和‘creatData’列修改为Date类型（格式：yyyyMMdd）。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2e9679eeba6844659014a670035a1e15.png#pic_center)
&emsp;&emsp;格式修改完成后，利用生成列算子。
&emsp;&emsp;1、生成'car day' 列，表达式为DATEDIFF(`creatDate`, `regDate`)，计算汽车注册时间 (regDate) 与上线时间 (creatDate) 之间的日期差。
&emsp;&emsp;2、生成'car year’列表示汽车使用的年数。表达式为DATEDIFF(`creatDate`, `regDate`) / 365，计算使用的年数。
&emsp;&emsp;3、生成'log1p_price'列，表达式为log1p(`price`)，通过计算 price 的自然对数，避免价格为 0 时的错误。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/99361abc49d74651980f2bb30b537831.png#pic_center)
&emsp;&emsp;生成列执行结果如下所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/49e45ca32c0f43c0823b7c07528a13e1.jpeg#pic_center)
&emsp;&emsp;这些步骤为后续的建模和分析奠定了坚实的数据基础，减少了数据异常的影响，增强了模型对数据的理解能力。
## (三) 特征选择与相关性分析

1、python代码实现
&emsp;&emsp;直方图和皮尔森相关性系数计算
```python
def plot_log1p_price_distribution(df, column='log1p_price', bins=20,output_dir=None):
    """
    绘制指定列的分布直方图及正态分布曲线
    参数:
    df: pd.DataFrame - 输入数据框
    column: str - 要绘制的列名
    bins: int - 直方图的组数
    output_dir: str - 保存图片的路径
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], bins=bins, kde=True, stat='density',
                   color='orange', edgecolor='black', alpha=0.6)

    mean = df[column].mean()
    std_dev = df[column].std()

    x = np.linspace(df[column].min(), df[column].max(), 100)
    p = np.exp(-0.5 * ((x - mean) / std_dev) ** 2) / (std_dev * np.sqrt(2 * np.pi))

    plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution Curve')
    plt.title('Distribution of log1p_price with Normal Distribution Curve', fontsize=16)
    plt.xlabel('log1p_price', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend()
    plt.tight_layout()
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_path = os.path.join(output_dir, 'log1p_price_distribution.png')
        plt.savefig(save_path, dpi=300)
    plt.show()

plot_log1p_price_distribution(df,output_dir=output_dir)

numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
log1p_price_corr = correlation_matrix['log1p_price'].drop('log1p_price')

print(log1p_price_corr)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/08b08f4fcfbe436ea795595cfe8a4562.png#pic_center.png =500x300)
&emsp;&emsp;删除'SaleID', 'name', 'regDate', 'model', 'brand', 'regionCode', 'seller', 'offerType', 'creatDate', 'price', 'v_4', 'v_7', 'v_13', 'regDate_year', 'regDate_month', 'regDate_day'列并进行流式归一化：

```python
columns_to_drop = ['SaleID', 'name', 'regDate', 'model', 'brand', 'regionCode', 'seller', 'offerType',
                   'creatDate', 'price', 'v_4', 'v_7', 'v_13', 'regDate_year', 'regDate_month', 'regDate_day']

df = df.drop(columns=columns_to_drop)
print(df.head())
>>   bodyType  fuelType  gearbox  power  ...      v_14  car_day  car_year  log1p_price
0       1.0       0.0      0.0     60  ...  0.914763     4385     12.01     7.523481
1       2.0       0.0      0.0      0  ...  0.245522     4757     13.03     8.188967
2       1.0       0.0      0.0    163  ... -0.229963     4382     12.01     8.736007
3       0.0       0.0      1.0    193  ... -0.478699     7125     19.52     7.783641
4       1.0       0.0      0.0     68  ...  1.923482     1531      4.19     8.556606

columns_to_normalize = df.columns.drop('log1p_price')
scaler = MaxAbsScaler()
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
df = df.round(3)
print(df.head())
>>   bodyType  fuelType  gearbox  power  ...   v_14  car_day  car_year  log1p_price
0     0.143       0.0      0.0  0.100  ...  0.106    0.475     0.475        7.523
1     0.286       0.0      0.0  0.000  ...  0.028    0.516     0.516        8.189
2     0.143       0.0      0.0  0.272  ... -0.027    0.475     0.475        8.736
3     0.000       0.0      1.0  0.322  ... -0.055    0.772     0.772        7.784
4     0.143       0.0      0.0  0.113  ...  0.222    0.166     0.166        8.557
```

2、Sentosa_DSML社区版实现
&emsp;&emsp;选择直方图图表分析算子，设定 log1p_price 列作为统计列，分组方式选择组数为20，然后启用显示正态分布的选项，用于展示 log1p_price列的值在不同区间的分布情况。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/14b82e014e004a1d87e5f00e6651697d.png#pic_center)
&emsp;&emsp;得到直方图结果如下所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/623c10a4c0924654a82c8be5ba32b3c2.jpeg#pic_center)
&emsp;&emsp;连接皮尔森相关系数算子并计算每一列之间的相关性，右侧设置需要计算皮尔森相关性系数的列，目的是为了分析特征之间的关系，以便为数据建模和特征选择提供依据。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9768bef459c34a63a37e5f9af482c38c.png#pic_center)
&emsp;&emsp;计算得到结果如下所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/372c654fd99d4a0283b0d337d56fc443.png)
&emsp;&emsp;通过进一步处理，删除冗余特征，连接删除和重命名算子，'SalelD','name','regDate','model','brand','regionCode','seller','offerType','creatDate','price','v_4','v_7','v_13','regDate_年','regDate_月','regDate_日'列进行删除。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/16d6d5eaa44e432d95488a3b2c62abbf.png#pic_center)
&emsp;&emsp;连接流式归一化算子进行特征处理，右侧选择归一化列，算法类型选择绝对值标准化。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/966573d43abd4d6f9f3f2eb47717bb4c.png#pic_center)
&emsp;&emsp;可以右击算子预览特征处理后的结果。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/49ce09c93f1d4e7a80ffd8cfcc8abe7e.jpeg#pic_center)
## (四) 样本分区与模型训练

1、python代码实现

```python
X = df.drop(columns=['log1p_price'])
y = df['log1p_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(
    n_estimators=100,             
    learning_rate=1,               
    max_depth=6,                   
    min_child_weight=1,            
    subsample=1,                   
    colsample_bytree=0.8,          
    objective='reg:squarederror',  
    eval_metric='rmse',            
    reg_alpha=0,                  
    reg_lambda=1,                  
    scale_pos_weight=1,                    
)

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
```

2、Sentosa_DSML社区版实现

&emsp;&emsp;连接样本分区算子，对数据处理和特征工程完成后的数据划分训练集和测试集，以用于实现后续的模型训练和验证流程。训练集和测试集样本比例为8：2。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/148fc88452a64f9aad53520ecff3be7d.png#pic_center)
&emsp;&emsp;然后，连接类型算子，展示数据的存储类型，测量类型和模型类型，将'log1p_price'列的模型类型设置为Label。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7b69fc7410bf44299be8d906d21a298c.png#pic_center)
&emsp;&emsp;样本分区完成后，连接XGBoost回归算子，可再右侧配置算子属性，评估指标即算法的损失函数，有均方根误差、均方根对数误差、平均绝对误差、伽马回归偏差四种；学习率，树的最大深度，最小叶子节点样本权重和，子采样率，最小分裂损失，每棵树随机采样的列数占比，L1正则化项和L2正则化项都是用来防止算法过拟合。当子节点样本权重和不大于所设的最小叶子节点样本权重和时不对该节点进行进一步划分。添加节点方式、最大箱数、是否单精度，这三个参数是当树构造方法是为hist的时候，才生效。最小分裂损失指定了节点分裂所需的最小损失函数下降值。
&emsp;&emsp;在本案例中，回归模型中的属性配置为，迭代次数：100，学习率：1，最小分裂损失：0，数的最大深度：6，最小叶子节点样本权重和：1，子采样率：1，每棵树随机采样的列数占比：0.8，树构造算法:auto，正负样本不均衡调节权重：1，L2正则化项：1，L1正则化项：0，学习目标为reg:squarederror，评估指标为均方根误差，初始预测分数为0.5，并计算特征重要性和残差直方图。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/868c678b1c154b39aa1109737ee300cd.png#pic_center)
&emsp;&emsp;右击执行可以得到XGBoost回归模型。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/732653551fda43c8a365630cb3e2f023.jpeg#pic_center)
## (五) 模型评估和模型可视化
1、python代码实现

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

train_metrics = calculate_metrics(y_train, y_train_pred)
test_metrics = calculate_metrics(y_test, y_test_pred)

print("训练集评估结果:")
print(train_metrics)
>>训练集评估结果:
{'R2': 0.9762793552927467, 'MAE': 0.12232836257076264, 'RMSE': 0.18761878906931295, 'MAPE': 1.5998275558939563, 'SMAPE': 1.5950003598874698, 'MSE': 0.035200810011835344}

print("\n测试集评估结果:")
print(test_metrics)
>>测试集评估结果:
{'R2': 0.9465739577985525, 'MAE': 0.16364796002127327, 'RMSE': 0.2815951292200689, 'MAPE': 2.176241755969303, 'SMAPE': 2.1652435034262068, 'MSE': 0.0792958168004673}
```
&emsp;&emsp;画出特征重要行图和残差直方图
```python
plt.figure(figsize=(10, 6))
plot_importance(model, importance_type='weight', max_num_features=10, color='orange')  
plt.title('特征重要性图', fontsize=16)
plt.xlabel('重要性', fontsize=14)
plt.ylabel('特征', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(train_residuals, bins=30, kde=True, color='blue') 
plt.title('残差分布', fontsize=16)
plt.xlabel('残差', fontsize=14)
plt.ylabel('技术', fontsize=14)
plt.axvline(x=0, color='red', linestyle='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/231c3b2651ab4bcb81618cefe923a3e8.jpeg#pic_center.png =400x300)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4dd45f343a9d47ccb15dfd0e87e7755b.jpeg#pic_center.png =500x300)

```python
test_data = pd.DataFrame(X_test)
test_data['log1p_price'] = y_test
test_data['predicted_log1p_price'] = y_test_pred

test_data_subset = test_data.head(200)

original_values = y_test[:200]
predicted_values = y_test_pred[:200]

x_axis = range(1, 201)

plt.figure(figsize=(12, 6))
plt.plot(x_axis, original_values.values, label='Original Values', color='orange')
plt.plot(x_axis, predicted_values, label='Predicted Values', color='green')

plt.title('Comparison of Original and Predicted Values')
plt.ylabel('log1p_price')
plt.legend()
plt.grid()

plt.show()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/464bc17cf9384e3db7426cb0701591f4.jpeg#pic_center.png =600x300)

2、Sentosa_DSML社区版实现

&emsp;&emsp;连接评估算子对模型进行评估。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/41574353c9cb44ee8cf8c3cb7f6f64e1.png#pic_center)
&emsp;&emsp;得到训练集和测试集的评估结果如下所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1c2e1332325a48b380b7ecdb7cebafad.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/74aa506e85594691823ea7f229067bf4.jpeg#pic_center)&emsp;&emsp;连接过滤算子，对测试集数据进行过滤，表达式为`Partition_Column`=='Testing'，
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/97c2987ceca0433db7df5bc5d59c9407.png#pic_center)
&emsp;&emsp;再接折线图图表分析算子，选择Lable列预测值列和原始值列，利用折线图进行对比分析。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/689dd3bf93304c7388a47eaa6fb2d68a.png#pic_center)
&emsp;&emsp;右击执行可得到测试集预测结果和原始值的对比图。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d875193d336245bfb38c4393ca8a5113.jpeg#pic_center)
&emsp;&emsp;右击模型可以查看特征重要性图、残差直方图等模型信息。结果如下所示：![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9962e5c7d24b4cc5b6f3dc5f85678570.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/22ad16821b2c4a3f84fa28c9e6af44e2.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cd17146053b64921b90a225d7940f39c.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5e164ab0d1da4d12ad43be5047cd0c17.jpeg#pic_center)&emsp;&emsp;通过连接各类算子并配置相关参数，完成了基于 XGBoost 回归算法的二手汽车价格预测建模。从数据的导入和清洗，到特征工程、模型训练，再到最终的预测、性能评估及可视化分析完成全流程建模，通过计算模型的多项评估指标，如均方误差 (MSE) 和 R² 值，全面衡量了模型的性能。结合可视化分析，实际值与预测值的对比展示了模型在二手汽车价格预测任务中的优异表现。通过这一系列的操作，完整体现了 XGBoost 在复杂数据集中的强大表现，为二手汽车价格预测提供了准确而可靠的解决方案。
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
