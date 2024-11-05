# 一、算法概念
什么是KNN？
&emsp;&emsp;K-近邻 (KNN) 是一种监督算法。KNN 背后的基本思想是在训练空间中找到距离新数据点最近的 K 个数据点，然后根据 k 个最近数据点中多数类别对新数据点进行分类，类似于“物以类聚”的思想，将一个样本的类别归于它的邻近样本。K-近邻算法是一种惰性学习模型(lazy learning)，也称为基于实例学习模型，这与勤奋学习模型(eager learning)不一样。
&emsp;&emsp;勤奋学习模型在训练模型的时候会很耗资源，它会根据训练数据生成一个模型，在预测阶段直接带入数据就可以生成预测的数据，所以在预测阶段几乎不消耗资源。
&emsp;&emsp;惰性学习模型在训练模型的时候不会估计由模型生成的参数，他可以即刻预测，但是会消耗较多资源，例如KNN模型，要预测一个实例，需要求出与所有实例之间的距离。
K-近邻算法是一种非参数模型，参数模型使用固定的数量的参数或者系数去定义模型，非参数模型并不意味着不需要参数，而是参数的数量不确定，它可能会随着训练实例数量的增加而增加，当数据量大的时候，看不出解释变量和响应变量之间的关系的时候，使用非参数模型就会有很大的优势，而如果数据量少，可以观察到两者之间的关系的，使用相应的模型就会有很大的优势。
&emsp;&emsp;存在一个样本集，也就是训练集，每一个数据都有标签，也就是我们知道样本中每个数据与所属分类的关系，输入没有标签的新数据后，新数据的每个特征会和样本集中的所有数据对应的特征进行比较，算出新数据与样本集其他数据的欧几里得距离，这里需要给出K值，这里会选择与新数据距离最近的K个数据，其中出现次数最多的分类就是新数据的分类，一般k不会大于20。
&emsp;&emsp;KNN在做回归和分类的主要区别，在于最后做预测时候的决策不同。在分类预测时，一般采用多数表决法。在做回归预测时，一般使用平均值法。
多数表决法：分类时，哪些样本离我的目标样本比较近，即目标样本离哪个分类的样本更接近。
&emsp;&emsp;平均值法： 预测一个样本的平均身高，观察目标样本周围的其他样本的平均身高，我们认为平均身高是目标样本的身高。
&emsp;&emsp;这里就运用了KNN的思想。KNN方法既可以做分类，也可以做回归，这点和决策树等算法相同。
![image](https://github.com/user-attachments/assets/5204bcd6-11c4-4574-b019-f4152d48628f)
&emsp;&emsp;由上图KNN分类算法可以发现，数据分为蓝色和绿色两个类别，当有一个新的数据点（红色）出现，并且 K = 5时，可以看到红色点有 3 个绿色近邻样本和 2个蓝色近邻样本，这说明蓝点将被归类为绿色类，因为多数投票为 3。同样，当 K 值变化时，圆内的近邻样本数量会增加，新的数据点被归类到其对应的多数投票类中。

![image](https://github.com/user-attachments/assets/df07d83e-d378-458e-a16a-eddc59e42472)
&emsp;&emsp;在 KNN 回归中，因变量是连续的，分布在整个特征空间中。当有新的数据点红色点出现时，会使用某种距离度量（如欧几里得距离）找到最接近的新数据点的 K 个近邻样本。找到这些邻居后，新数据点的预测值通过计算这些邻居的因变量值的平均值来确定。
例如，假设我们想预测学生的考试成绩，而已知的特征是学习时长。我们已经有了许多学生的学习时长和对应的考试成绩数据。现在，针对一个新来的学生，我们知道他的学习时长，通过 KNN 回归，我们可以找到学习时长最接近的 K 个学生，然后将这 K 个学生的考试成绩取平均值，作为这个新学生的成绩预测。
# 二、算法原理
## （一）K值选择

&emsp;&emsp; $K$值的选择与样本分布有关，一般选择一个较小的 $K$值，可以通过交叉验证来选择一个比较优的 $K$值，默认值是5。如果数据是三维以下的，如果数据是三维或者三维以下的，可以通过可视化观察来调参。
&emsp;&emsp;当$k=1$时的 $k$近邻算法称为最近邻算法，此时将点$X$分配给特征空间中其最近邻的点的类。即：$C_{n}^{1nn}(X)=Y_{(1)}$
&emsp;&emsp; $K$值的选择会对 $k$近邻法的结果产生重大影响。若 $K$值较小，则相当于用较小的邻域中的训练样本进行预测，"学习"的偏差减小。
&emsp;&emsp;只有与输入样本较近的训练样本才会对预测起作用，预测结果会对近邻的样本点非常敏感。
&emsp;&emsp;若 $k$近邻的训练样本点刚好是噪声，则预测会出错。即：  值的减小意味着模型整体变复杂，易发生过拟合。
&emsp;&emsp;**优点**：减少"学习"的偏差。
&emsp;&emsp;**缺点**：增大"学习"的方差（即波动较大）。
&emsp;&emsp;若 $K$值较大，则相当于用较大的邻域中的训练样本进行预测。
&emsp;&emsp;这时输入样本较远的训练样本也会对预测起作用，使预测偏离预期的结果。  $K$值增大意味着模型整体变简单。
&emsp;&emsp;**优点**：减少"学习"的方差（即波动较小）。
&emsp;&emsp;**缺点**：增大"学习"的偏差。
&emsp;&emsp;应用中， $K$值一般取一个较小的数值。通常采用交叉验证法来选取最优的 $K$值, $K$值的选择取决于数据集和问题。较小的 $K$值可能导致过度拟合，而较大的 $K$值可能导致欠拟合，可以尝试不同的 $K$值，以找到特定数据集的最佳值。
## （二）距离度量
&emsp;&emsp;距离度量是 KNN 算法中用来计算数据点之间相似性的重要组成部分，以下是几种常见的距离度量类型：
### 1、欧式距离
&emsp;&emsp;欧式距离是 KNN 中最广泛使用的距离度量，表示两个数据点在欧几里得空间中的直线距离。
![image](https://github.com/user-attachments/assets/4f785e72-5e45-4089-8849-bc349843435f)

&emsp;&emsp;计算公式如下：
$$d(x,y)=\sqrt{\sum_{i=1}^n\left(x_i-y_i\right)^2}$$
### 2、曼哈顿距离
&emsp;&emsp;曼哈顿距离（Manhattan Distance）是一种衡量两点之间距离的方式，两个点之间的距离是沿着网格（即水平和垂直方向）的路径总和，而不是像欧几里得距离那样的直线距离。
![image](https://github.com/user-attachments/assets/cfac6aff-7056-4424-8d3e-bd7a4864274e)
&emsp;&emsp;计算公式如下：

$$d(x,y)=\sum_{i=1}^n|x_i-y_i|$$
### 3、闵可夫斯基距离
&emsp;&emsp;闵可夫斯基距离是一个广义的距离度量，可以根据参数 𝑝的不同，生成多种常见的距离度量，如曼哈顿距离和欧几里得距离。计算方法是绝对差和的 p 次方根.例如：
&emsp;&emsp;如果 p = 1，简化为曼哈顿距离，即绝对差的和。
&emsp;&emsp;如果 p = 2，简化为欧几里得距离，即平方差的和的平方根。
&emsp;&emsp;距离度量必须满足一些条件：
&emsp;&emsp;&emsp;&emsp;非负性：任意两点之间的距离不能为负。
&emsp;&emsp;&emsp;&emsp;同一性：点与自身的距离为零。
&emsp;&emsp;&emsp;&emsp;对称性：两点之间的距离相同。
&emsp;&emsp;&emsp;&emsp;三角不等式：两点之间的距离应小于或等于通过第三点的路径之和。
&emsp;&emsp;计算公式如下所示：
$$d(x,y)=\left(\sum_{i=1}^n|x_i-y_i|^p\right)^{\frac1p}$$
## （三）决策规则
### 1、分类决策规则
&emsp;&emsp;KNN算法的分类决策通常基于多数表决原则，即在新样本的 $K$个最近邻中，哪个类别的样本数最多，则预测新样本属于该类别。此外，也可以根据样本与新数据点之间的距离远近进行加权投票，即离新样本越近的邻居样本权重越大，影响力也越大。分类的决策规则可以用经验风险最小化来解释，假设我们要最小化分类中的错误，即最小化0-1 损失函数。这个损失函数可以表示为：
$$L=\dfrac{1}{K}\sum_{{{x}}_i\in\mathcal{N}_K({{X}})}I({y}_i\neq c_m)=1-\dfrac{1}{K}\sum_{{{x}}_i\in\mathcal{N}_K({{X}})}I({y}_i=c_m)$$
&emsp;&emsp;其中：
&emsp;&emsp;$\bullet$ ${{X} }$是新样本点。
&emsp;&emsp;$\bullet$ $\mathcal{N} _k( {{x} } )$是新样本$\bullet$ ${{X}}$的 K 个最近邻的集合。
&emsp;&emsp;$\bullet$ ${y} _i$是近邻 ${{x}}_i$ 的实际标签。
&emsp;&emsp;$\bullet$ $c_m$是预测的类别，即我们希望找到的多数类别。
&emsp;&emsp;$\bullet$ $I( {y} _i\neq c_m)$是指示函数，表示近邻${{x}}_i$的实际类别与$c_m$是否相同。
&emsp;&emsp;通过这个公式可以看到，损失函数$L$表示的是分类错误的比例。为了最小化损失函数，我们需要找到一个类别$c_{m}$使得近邻中属于该类别的样本数量最大。因此，损失函数最小化等价于多数表决：
$$c_m=\arg\max_{c_m}\sum_{{{x}}_i\in\mathcal{N}_k({{X}})}I({y}_i=c_m)$$
### 2、回归决策规则
&emsp;&emsp;对于 KNN 回归，决策规则与分类类似，但在回归问题中，输出是连续值，而不是类别。因此，KNN 回归通常采用均值回归，即新样本的预测值是 K 个最近邻样本的目标值的均值。同样，我们也可以基于样本与新样本点之间的距离进行加权投票，距离越近的邻居对预测的影响力越大。
&emsp;&emsp;在回归问题中，我们依然可以使用经验风险最小化的思想，回归中的损失函数一般是均方误差(MSE)。回归问题的损失函数表示为：
$$L=\dfrac{1}{K}\sum_{{{x}}_i\in\mathcal{N}_K({{X}})}({y}_i-\hat{y})^2$$
&emsp;&emsp;其中：
&emsp;&emsp;$\bullet$ $\hat{y}$表示新样本的预测值。
&emsp;&emsp;$\bullet$ ${y} _i$是近邻点${{x}}_i$的实际值。
&emsp;&emsp;这个损失函数表示新样本X预测值与近邻点实际值之间平方误差的平均值。为了最小化均方误差，最优的预测值$\hat{y}$应该是近邻样本实际值${y}_i$的平均值，即：
$$\hat{y}=\frac{1}{K}\sum_{{{x}}_i\in\mathcal{N}_K({{X}})}{y}_i$$
&emsp;&emsp;因此，KNN 回归算法的预测值是近邻点的平均值，即均值回归。
&emsp;&emsp;总结来说，KNN 算法的分类和回归决策规则虽然任务不同，但核心思想都是通过找到最近的K个邻居，然后根据这些邻居的属性 (类别或值) 进行多数表决或均值计算，从而做出预测。
# 三、算法优缺点
### 优点
&emsp;&emsp;1、KNN 可用于分类和回归任务，适应性强。
无需训练：算法不需要训练阶段，直接使用数据进行预测，节省计算资源。
&emsp;&emsp;2、由于依赖于邻居的多数投票，KNN 对噪声数据具有较强的抗干扰能力，不易受异常值影响。
&emsp;&emsp;3、KNN 在小数据集上表现良好，不需要大量数据即可进行预测。
&emsp;&emsp;4、不需要对数据做出特定假设，适合多种数据分布。
&emsp;&emsp;5、算法直观易懂，便于实现和应用。
### 缺点
&emsp;&emsp;1、正确选择 K 值很重要，影响模型的性能，但不同数据集的最佳 K 值不同。
&emsp;&emsp;2、在类别不平衡的数据集中，KNN 可能偏向于多数类别。
&emsp;&emsp;3、当特征维度很高时，KNN 的效果会下降，因为距离计算不精确，难以找到有意义的近邻。
# 四、KNN分类任务实现对比
## （一）数据加载和样本分区
### 1、Python代码
```python
# 导入相关包
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.datasets import load_iris

##在线导入	
#data = datasets.load_iris()

# 本地导入
df = pd.read_csv("D:/sentosa_ML/Sentosa_DSML/mlServer/TestData/Iris.csv")
df.head()

df["Species"].unique()
df.shape
encoder = LabelEncoder()
df["Species"] = encoder.fit_transform(df["Species"])

# 将数据集划分为特征和标签
X = df.drop("Species", axis=1)  # 特征
y = df["Species"]  # 标签

# 数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
```
### 2、Sentosa_DSML社区版
&emsp;&emsp;首先，利用文本算子对数据集进行读入，
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/dfa40eb7f1d3487a9ac7231037f71c60.png#pic_center)
&emsp;&emsp;其次，连接样本分区算子，划分测试集和训练集的比例为2：8
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b7c7a4cd4ddf42bb9933a65ad118f792.png#pic_center)
&emsp;&emsp;然后接类型算子，设置Feature列和Label列
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/12c38ae650364ea2bf11799c47751a41.png#pic_center)
## （二）训练模型
### 1、Python代码

```python
# 初始化 KNN 分类器,K值设置为3
knn = KNeighborsClassifier(n_neighbors=3)

# 训练模型
knn.fit(X_train, y_train)

# 对新样本进行预测
X_new = pd.DataFrame([[5, 2.9, 1, 0.2]], columns=X.columns)
prediction = knn.predict(X_new)

# print(f"Predicted target name: {encoder.inverse_transform(prediction)}")

# 对测试集进行预测
y_pred = knn.predict(X_test)
```
### 2、Sentosa_DSML社区版
&emsp;&emsp;连接KNN分类算子，这里我们将K值设置为3，点击应用并执行，
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2ed323cd6ef9478cb5870697ba3447ef.png#pic_center)
&emsp;&emsp;执行结束后得到KNN模型
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/454c1c244c5843908e2a2b6ff249cbc3.jpeg#pic_center)
## （三）模型评估和模型可视化
### 1、Python代码
```python
# 评估模型
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# 打印评估结果
print(f"KNN 模型的准确率: {accuracy:.2f}")
print(f"加权精度 (Weighted Precision): {precision:.2f}")
print(f"加权召回率 (Weighted Recall): {recall:.2f}")
print(f"F1 值 (Weighted F1 Score): {f1:.2f}")

# 生成混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 绘制特征分布图
plt.figure(figsize=(12, 8))
for i, column in enumerate(X.columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=df['Species'], y=X[column])
    plt.title(f'{column} by Species')

plt.tight_layout()
plt.show()
```
&emsp;&emsp;可以绘制Iris数据集中各个特征在不同分类下的分布图，以帮助我们了解不同类别的鸢尾花（Species）在不同特征（如sepal_length, sepal_width等）上的分布情况，生成了4个箱线图，每个箱线图展示了一个特征在不同Species（花种）类别上的分布情况。
![image](https://github.com/user-attachments/assets/ba064e4d-b56c-4481-a06e-ffc7c38f5319)
### 2、Sentosa_DSML社区版
&emsp;&emsp;连接评估算子对模型进行评估
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/06dfc9a201cc4a3b861d48df69391ee4.png#pic_center)
&emsp;&emsp;得到训练集和测试集的各评估指标结果，如下所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/153151702578407da3b582a2161ff006.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/036fe521db8a420194a824795e058948.jpeg#pic_center)
&emsp;&emsp;右击模型可以查看模型的模型信息，可以得到模型混淆矩阵、特征重要性等结果
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e066e139c47e4f97a412d383ef3469aa.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0de38b6ccb3c47dea16c6599d4f5f913.jpeg#pic_center)
&emsp;&emsp;也可以连接二维箱线图，绘制Iris数据集中不同特征（sepal_length, sepal_width、petal_length、petal_width）在不同分类（Species）下的分布图
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c02f60ead4a7440bad3aa72d97a39ffd.png#pic_center)
&emsp;&emsp;执行后可以得到绘制结果，如下图所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d613c5e3f2824153b16dda5d7645d68d.png#pic_center)
# 五、KNN回归任务实现对比
## （一）数据加载和样本分区
### 1、Python代码
```python
# 导入相关库
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns

# 读取 winequality 数据集
df = pd.read_csv("D:/sentosa_ML/Sentosa_DSML/mlServer/TestData/winequality.csv")

# 检查数据集
print(df.head())
print(df.info())

# 将数据集划分为特征和目标变量
X = df.drop("quality", axis=1)  # 特征
y = df["quality"]  # 标签（目标值）

# 数据集拆分为训练集和测试集，训练集和测试集比例为8：2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```
### 2、Sentosa_DSML社区版
&emsp;&emsp;首先，利用文本算子对数据集进行读取
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/71bfd331aaa04202b565ef7b957d3cc8.png#pic_center)
&emsp;&emsp;其次，连接样本样本分区算子划分训练集和测试集，训练集和测试集比例为8：2
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/49c89dd392604bcda2877057aad8bbe7.png#pic_center)
&emsp;&emsp;然后连接类型算子设置Label列和Feature列
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c8fade971a0a4fc998337c82ea698e0f.png#pic_center)
## （二）训练模型
### 1、Python代码

```python
# 初始化 KNN 回归模型
knn_regressor = KNeighborsRegressor(n_neighbors=3)  # K值设置为3
# 训练模型
knn_regressor.fit(X_train, y_train)

# 对测试集进行预测
y_pred = knn_regressor.predict(X_test)
```
### 2、Sentosa_DSML社区版
&emsp;&emsp;连接KNN回归算子，设置模型超参数K值为3
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d2e7f31626e04f1a8e02fdfa01045597.png#pic_center)
&emsp;&emsp;右键执行后得到KNN模型
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/de6b9b54c5c343e3ae23e6b6347b1406.jpeg#pic_center)
## （三）模型评估和模型可视化
### 1、Python代码

```python
# 计算评估指标
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
smape = 100 / len(y_test) * np.sum(2 * np.abs(y_test - y_pred) / (np.abs(y_test) + np.abs(y_pred)))

# 打印评估结果
print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"SMAPE: {smape:.2f}%")

# 绘制实际值与预测值的对比图
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color="blue", alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual Quality')
plt.ylabel('Predicted Quality')
plt.title('Actual vs Predicted Wine Quality')
plt.show()

# 计算残差
residuals = y_test - y_pred

# 使用 Seaborn 并添加核密度估计 (KDE) 曲线
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins=20)
plt.title('Residuals Histogram with KDE')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
```
### 2、Sentosa_DSML社区版
&emsp;&emsp;连接评估算子，对模型进行评估
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1aa8a60657ad403cb33f4db13cbaab9a.png#pic_center)
&emsp;&emsp;得到训练集和测试集评估结果如下图所示：![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f42f2022f5d546188c3ea4d8fcae3ca1.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ab660c09c5e740df8806d70859f0f85e.jpeg#pic_center)
&emsp;&emsp;右击可以查看模型信息，得到模型特征重要性图、实际值—残差值散点图和残差立方图等结果，如下所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f6b5080e6836483aa18aecf9f073b082.jpeg#pic_center)![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b18f82ca77094a9bad2b2eaf1ad2f027.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fbf592a7c6404b19a54d5f73b2b81287.jpeg#pic_center)
# 六、总结

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
