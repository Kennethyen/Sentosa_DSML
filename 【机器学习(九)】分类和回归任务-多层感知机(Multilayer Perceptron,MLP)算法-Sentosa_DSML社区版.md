# 一、算法概念
什么是多层感知机？
&emsp;&emsp;多层感知机 (Multilayer Perceptron，MLP) 是一种人工神经网络，由多层神经元或节点组成，这些神经元或节点以分层结构排列。它是最简单且使用最广泛的神经网络之一，尤其适用于分类和回归等监督学习任务。
&emsp;&emsp;多层感知器运作的核心原理在于反向传播，是用于训练网络的关键算法。在反向传播过程中，网络通过将误差从输出层反向传播到输入层来调整其权重和偏差。这个迭代过程可以微调模型的参数，使其能够随着时间的推移做出更准确的预测。
&emsp;&emsp;MLP 通常包括以下部分：
&emsp;&emsp;输入层：接收输入数据并将其传递到隐藏层。输入层中的神经元数量等于输入特征的数量。
&emsp;&emsp;隐藏层：由一层或多层神经元组成，用于执行计算并转换输入数据。可以调整每层&emsp;&emsp;中的隐藏层和神经元的数量，以优化网络性能。
&emsp;&emsp;激活函数：对隐藏层中每个神经元的输出应用非线性变换。常见的激活函数包括 Sigmoid、ReLU、tanh 等。
&emsp;&emsp;输出层：网络的最终输出，例如分类标签或回归目标。输出层中的神经元数量取决于具体的数据，例如分类问题中的类别数量。
&emsp;&emsp;权重和偏差：可调节参数，决定相邻层神经元之间的连接强度以及每个神经元的偏差。这些参数在训练过程中学习，以尽量减少网络预测与实际目标值之间的差异。
&emsp;&emsp;损失函数：衡量网络预测与实际目标值之间的差异。MLP 的常见损失函数包括回归任务的均方误差和分类任务的交叉熵。
&emsp;&emsp;MLP 使用梯度下降等优化算法反向传播进行训练，根据损失函数的梯度迭代调整权重和偏差。这个过程持续到网络收敛到一组可最小化损失函数的最佳参数。
# 二、算法原理
## （一）感知机
&emsp;&emsp;感知机由两层神经元组成，输入层接收外界信号后传递给输出层，如下图所示，
 ![image](https://github.com/user-attachments/assets/55cabb19-04c8-457b-a334-6bea7391d8b0)
&emsp;&emsp;感知机模型就是尝试找到一条直线，能够把所有的二元类别分离开，给定输入 $\mathbf{x}$ ，权重 $\mathbf{W}$ ，和偏移 $b$ ，感知机输出:
$$o=\sigma\left( \langle\mathbf{w}, \mathbf{x} \rangle+b \right)$$
$$\quad\sigma( x )=\left\{\begin{array} {l l} {{1}} & {{\mathrm{~} x > 0}} \\ {{-1}} & {{\mathrm{~} x\leq0}} \\ \end{array} \right. $$
&emsp;&emsp;初始化权重向量 w 和偏置 b，然后对于分类错误的样本不断更新w和b，直到所有样本都被正确分类。等价于使用批量大小为1的梯度下降，并使用如下的损失函数：
$$\ell( y, {\bf x}, {\bf w} )=\operatorname* {m a x} ( 0,-y \langle{\bf w}, {\bf x} \rangle) $$
&emsp;&emsp;感知机只能产生线性分割面，感知机算法的训练过程如下。
![image](https://github.com/user-attachments/assets/592c98be-d7cd-42f8-bbfa-20ebb200ee7c)
## （二）多层感知机
### 1、隐藏层
&emsp;&emsp;多层感知机则是在单层神经网络的基础上引入一个或多个隐藏层，使神经网络有多个网络层，下图为两个多层感知机示意图，分别为单隐层和双隐层
![image](https://github.com/user-attachments/assets/62860d0d-a6b5-49a8-8fa3-c6bb954cfcd7)
![image](https://github.com/user-attachments/assets/9e3c2565-af59-47f2-8ea0-230e6a5814c5)
&emsp;&emsp;多层感知机中的隐藏层和输出层都是全连接层，输入 $X \in\mathbb{R}^{n \times d}$ ，其中， $n$ 是批量大小， $d$ 是输入特征的数量。输出 $O \in\mathbb{R}^{n \times q}$ ，其中 $q$ 是输出单元的数量。
&emsp;&emsp;设隐藏层有 $h$ 个隐藏单元，隐藏层的输出 $H$ 是通过输入$X$ 与隐藏层的权重 $W_{h} \in\mathbb{R}^{d \times h}$ 和偏置 $b_{h} \in\mathbb{R}^{1 \times h}$ 计算得到的：$$H=X W_{h}+b_{h} $$
&emsp;&emsp;输出层的权重为 $W_{o} \in\mathbb{R}^{h \times q}$ ，偏置为 $b_{o} \in\mathbb{R}^{1 \times q}$ 。因此，输出层的输出 $O$ 为:$$O=H W_{o}+b_{o} $$
&emsp;&emsp;将隐藏层的输出 $H$ 代入到输出层的方程中，得到如下计算过程：
$$O=( X W_{h}+b_{h} ) W_{o}+b_{o}=X W_{h} W_{o}+b_{h} W_{o}+b_{o} $$
&emsp;&emsp;通过联立后的式子可以看出，尽管引入了隐藏层，模型的计算仍然可以视作单层神经网络，其中，权重矩阵等于 $W_{h} W_{o}$，偏置等于 $b_{h} W_{o}+b_{o}$。
&emsp;&emsp;这表示，尽管引入了隐藏层，在不采用非线性激活函数的情况下，这个设计只能等价于单层神经网络。引入隐藏层的真正意义在于通过非线性激活函数（如ReLU、Sigmoid等）来引入复杂的非线性关系，使得模型具备更强的表达能力。
### 2、激活函数
&emsp;&emsp;激活函数是 MLP的关键组成部分。它们将非线性引入网络，使其能够对复杂问题进行建模。如果没有激活函数，无论有多少层，MLP都相当于单层线性模型。
激活函数需要具备以下几点性质:
1. 连续并可导（允许少数点上不可导），便于利用数值优化的方法来学习网络参数
2. 激活函数及其导函数要尽可能的简单，有利于提高网络计算效率
3. 激活函数的导函数的值域要在合适区间内，不能太大也不能太小，否则会影响训练的效率和稳定性
以下列举常用的三个激活函数
#### sigma函数
$$
sigma( z )=\frac{1} {1+\operatorname{e x p} (-z )} 
$$
&emsp;&emsp;sigma函数也称为 $\mathrm{S}$ 型函数，可以将任何实值数映射到 $0$ 到 $1$ 之间的值。呈S形，具有明确定义的非零导数，这使其适合与反向传播算法一起使用。
![image](https://github.com/user-attachments/assets/2af22427-d4bf-4186-bc2a-50f4ebe007aa)
&emsp;&emsp;sigmoid函数的导数表达式为：
$$sigma^{\prime} ( z )=sigma( z ) \times( 1-sigma ( z ) ) $$
&emsp;&emsp;如下所示：
![image](https://github.com/user-attachments/assets/ac5f5274-54b2-4f43-a073-38dc6ca4206a)
#### tanh函数
$$\operatorname{t a n h} ( z )=\frac{1-\operatorname{e x p} (-2z )} {1+\operatorname{e x p} (-2z )} $$
&emsp;&emsp;双曲正切函数与逻辑函数类似，但输出值在-1和 $1$ 之间。这种居中效果有助于加快训练期间的收敛速度。
![image](https://github.com/user-attachments/assets/e46ebde3-3f5c-420d-a5a8-86708418438d)
&emsp;&emsp;tanh导数表达式如下所示：
$$tanh^{\prime} ( z)=1-\operatorname{t a n h}^{2} ( z ) $$
&emsp;&emsp;下面绘制了tanh函数的导数。当输入为0时，tanh函数的导数达到最大值1；当输入越偏离0时，tanh函数的导数越接近0。
![image](https://github.com/user-attachments/assets/b594f506-deb0-43c5-90dd-67ea6799a3ba)
#### ReLU函数
$$
\mathrm{R e L U} ( z )=\operatorname* {m a x} ( 0, z ) 
$$
&emsp;&emsp;ReLU 函数因其简单性和有效性而被广泛应用于深度学习。如果输入值为正，则输出输入值；否则输出零。尽管 ReLU 在零处不可微，并且对于负输入具有零梯度，但它在实践中表现良好，有助于缓解梯度消失问题
![image](https://github.com/user-attachments/assets/fea6fe08-effd-459c-a31b-008159f5c34d)
&emsp;&emsp;当输入为负数时，ReLU函数的导数为0；当输入为正数时，ReLU函数的导数为1，
&emsp;&emsp;ReLU 函数的导数表达式为：
$$R e L U^{\prime} ( z )=\begin{cases} {{1}} & {{\mathrm{i f ~} z > 0}} \\ {{0}} & {{\mathrm{i f ~} z \leq0}} \\ \end{cases} $$
&emsp;&emsp;下面绘制ReLU函数的导数，
![image](https://github.com/user-attachments/assets/e3935408-6bf1-4dbb-90aa-901de10e5529)
### 3、反向传播算法
**1、前向传播**
&emsp;&emsp;前向传播是反向传播的前提。在前向传播过程中，数据从输入层逐步传递至输出层，经过每一层的计算，最终得到预测输出。
&emsp;&emsp;具体步骤如下:
&emsp;&emsp;1、输入数据传递给神经网络的输入层。
&emsp;&emsp;2、输入层经过一系列权重（W）和偏置（b）的线性运算，然后通过激活函数传递到隐藏层。
&emsp;&emsp;3、逐层传递，直至数据到达输出层，输出层生成预测值 $\hat{y}$ 。
&emsp;&emsp;表达式如下：
$$\hat{y}=f ( W_{3} \cdot f ( W_{2} \cdot f ( W_{1} \cdot x+b_{1} )+b_{2} )+b_{3} ) $$
&emsp;&emsp;其中， $W_{1}, W_{2}, W_{3}$ 是权重矩阵， $b_{1}, b_{2}, b_{3}$ 是偏置， $f ( \cdot)$ 是激活函数。
**2、 损失函数**
&emsp;&emsp;在得到输出后，通过损失函数计算预测结果与真实标签之间的误差，常见的损失函数有：
&emsp;&emsp;**MSE（均方误差）**：通常用于回归问题，输出与标签之差的平方的均值。计算公式如下：
$$MSE=\frac{1} {n} \sum_{i=1}^{n} ( y_{i}-\hat{y}_{i} )^{2} $$
&emsp;&emsp;其中， $y_{i}$ 是真实值， $\hat{y}_{i}$ 是预测值， $n$ 是样本数量。
&emsp;&emsp;**CE（交叉熵损失）**：通常用于回归问题。计算公式如下：
$$H(p,q)=-\sum_{i=1}^{n}p(x_{i}) \operatorname{log}q(x_{i}) $$
&emsp;&emsp;其中， $p ( x_{i} )$ 是真实分布， $q ( x_{i} )$ 是预测分布。
**3、反向传播**
&emsp;&emsp;反向传播根据微积分中的链式规则，按相反的顺序从输入层遍历网络。用于权重更新，使网络输出更接近标签。
&emsp;&emsp;假设有两个函数 $y=f ( u )$ 和 $u=g ( x )$ ，根据链式法则， $y$ 对 $x$ 的导数为：
$$\frac{\partial y} {\partial x}=\frac{\partial y} {\partial u} \frac{\partial u} {\partial x}$$
&emsp;&emsp;在神经网络中，损失函数 $L$ 对某一层权重 $W$ 的导数可以通过链式法则分解为：
$$\frac{\partial L} {\partial W}=\frac{\partial L} {\partial y} \cdot\frac{\partial y} {\partial W} $$
**4、梯度下降**
&emsp;&emsp;在反向传播过程中，利用梯度下降算法来更新权重，使得损失函数的值逐渐减小。权重更新的公式为:
$$W^{(h )}=W^{( o  )}-\eta\cdot\frac{\partial L} {\partial W} $$
&emsp;&emsp;其中， $\eta$ 是学习率，决定了每次权重调整的步长大小，$\frac{\partial L} {\partial W}$ 是损失函数相对于权重的梯度。
# 三、算法优缺点
## （一）优点
&emsp;&emsp;可以通过多个隐藏层和非线性激活函数，学习到更复杂的特征表示，从而提高模型的表达能力。
&emsp;&emsp;可以用于分类、回归和聚类等各种机器学习任务，目在许多领域中取得了很好的效果。
&emsp;&emsp;可以诵过并行计算和GPU加速等技术，高效地处理大规模数据集，适用于大规模深度学习应用。
## （二）缺点
&emsp;&emsp;参数较多，容易在训练集上过拟合，需要采取正则化、dropout等方法来缓解过拟合问题。
&emsp;&emsp;通常需要大量的标记数据进行训练，并且在训练过程中需要较高的计算资源，包括内存和计算
能力。
&emsp;&emsp;MLP的性能很大程度上依赖于超参数的选择。
# 四、MLP分类任务实现对比
## （一）数据加载和样本分区
### 1、Python代码

```python
from sklearn.datasets import load_iris

# 加载iris数据集
iris = load_iris()
X, y = iris['data'], iris['target']

# 样本分区
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 2、Sentosa_DSML社区版
&emsp;&emsp;首先，利用数据读入中的文本算子对数据进行读取，
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e2cf7c45e1084e0fb5befe2358213c1d.png#pic_center)
&emsp;&emsp;然后连接样本分区算子划分训练集和测试集，
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e65b1147128742a3998785eeebd23749.png#pic_center)
&emsp;&emsp;再接类型算子，设置Feature列和Label列，
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2e121111c7a047af84452c6b1a2447a4.png#pic_center)
## （二）模型训练
### 1、Python代码
&emsp;&emsp;使用sklearn自动构建MLP模型
```python
from sklearn.neural_network import MLPClassifier

# 定义MLP分类器模型，使用l-bfgs优化算法，隐藏层设置为100, 50，最大迭代次数200，设置tol为0.000001
mlp_clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=200, alpha=1e-4,
                        solver='lbfgs', tol=1e-6, random_state=42)
# 训练模型
mlp_clf.fit(X_train, y_train)

# 预测训练集和测试集
y_train_pred = mlp_clf.predict(X_train)
y_test_pred = mlp_clf.predict(X_test)
```
### 2、Sentosa_DSML社区版
&emsp;&emsp;连接多层感知机分类算子，右击算子，点击运行，可以得到多层感知机分类模型。右侧进行超参数等设置，隐藏层设置为（100, 50），使用l-bfgs优化算法，最大迭代次数200，设置收敛偏差为0.000001。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cdf0a8f04d07483b94dd4db810935609.png#pic_center)
## （三）模型评估和模型可视化
### 1、Python代码

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 计算训练集评估指标
accuracy_train = accuracy_score(y_train, y_train_pred)
precision_train, recall_train, f1_train, _ = precision_recall_fscore_support(y_train, y_train_pred, average='weighted')

# 计算测试集评估指标
accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(y_test, y_test_pred, average='weighted')

# 输出训练集评估指标
print(f"Training Set Metrics:")
print(f"Accuracy: {accuracy_train * 100:.2f}%")
print(f"Weighted Precision: {precision_train:.2f}")
print(f"Weighted Recall: {recall_train:.2f}")
print(f"Weighted F1 Score: {f1_train:.2f}")

# 输出测试集评估指标
print(f"\nTest Set Metrics:")
print(f"Accuracy: {accuracy_test * 100:.2f}%")
print(f"Weighted Precision: {precision_test:.2f}")
print(f"Weighted Recall: {recall_test:.2f}")
print(f"Weighted F1 Score: {f1_test:.2f}")

from sklearn.metrics import confusion_matrix

# 计算测试集的混淆矩阵
conf_matrix = confusion_matrix(y_test, y_test_pred)

import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# 使用 sklearn 提供的permutation_importance方法计算特征重要性
result = permutation_importance(mlp_clf, X_test, y_test, n_repeats=10, random_state=42)

# 可视化特征重要性
plt.figure(figsize=(8, 6))
plt.barh(range(X.shape[1]), result.importances_mean, align='center')
plt.yticks(np.arange(X.shape[1]), iris['feature_names'])
plt.xlabel('Mean Importance Score')
plt.title('Permutation Feature Importance')
plt.show()
```
![image](https://github.com/user-attachments/assets/2a76299d-6f74-4f9d-a248-9b4d4b25fdc6)
### 2、Sentosa_DSML社区版
&emsp;&emsp;模型后可以连接评估算子，对模型的分类结果进行评估。算子流如下图所示，
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f66fb47580f34c5f89853d68146bd7bd.jpeg#pic_center)
&emsp;&emsp;执行完成后可以得到训练集和测试集的评估，评估结果如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/85a04972a18a40b3976928f4b1b2be13.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/19064ad65d754514b7796001342c9b53.jpeg#pic_center)
&emsp;&emsp;右击模型，查看模型的模型信息，如下所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8dc157abc349430ebb108d7d4034cda6.jpeg#pic_center)
# 五、MLP回归任务实现对比
## （一）数据加载和样本分区
### 1、Python代码
```python
# 读入winequality数据集
df = pd.read_csv("D:/sentosa_ML/Sentosa_DSML/mlServer/TestData/winequality.csv")

# 将数据集划分为特征和标签
X = df.drop("quality", axis=1)  # 特征，假设标签是 "quality"
Y = df["quality"]  # 标签

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```
### 2、Sentosa_DSML社区版
&emsp;&emsp;首先通过数据读入算子读取数据，
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/caf3882b54ee4acbb0f122e476ad339b.png#pic_center)
&emsp;&emsp;中间接样本分区算子对训练集和测试集进行划分，
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b61d006d4c474bf3afcc944f479a7f6d.png#pic_center)
&emsp;&emsp;然后接类型算子，设置Feature列和Label列，
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/12f8fd379dfc4410aa6c841625344149.png#pic_center)
## （二）模型训练
### 1、Python代码
使用 scikit-learn 库中的多层感知机回归模型（MLPRegressor）
```python
# 对数据进行标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义MLP回归模型，使用l-bfgs优化算法，隐藏层设置为50，10，最大迭代次数300，设置tol为0.000001
mlp_reg = MLPRegressor(hidden_layer_sizes=(50, 10), solver='lbfgs', max_iter=300, tol=1e-6, random_state=42)

# 训练模型
mlp_reg.fit(X_train_scaled, y_train)
```
### 2、Sentosa_DSML社区版
&emsp;&emsp;连接标准化算子，对数据特征进行标准化计算，并执行得到标准化模型，
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7aafdb70bea7445aa13c726ca35c7245.png#pic_center)
&emsp;&emsp;其次，连接多层感知机回归算子，右击执行得到多层感知机回归模型。模型训练使用l-bfgs优化算法，隐藏层设置为50，10，最大迭代次数300，设置收敛偏差为0.000001，并选择计算特征重要性等。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/81968d6d2ba14ad38900dbac71657109.png#pic_center)
## （三）模型评估和模型可视化
### 1、Python代码
```python
# 训练集上的评估
y_train_pred = mlp_reg.predict(X_train_scaled)

r2_train = r2_score(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
smape_train = 100 / len(y_train) * np.sum(2 * np.abs(y_train - y_train_pred) / (np.abs(y_train) + np.abs(y_train_pred)))

# 测试集上的评估
y_test_pred = mlp_reg.predict(X_test_scaled)

r2_test = r2_score(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
mape_test = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
smape_test = 100 / len(y_test) * np.sum(2 * np.abs(y_test - y_test_pred) / (np.abs(y_test) + np.abs(y_test_pred)))

# 输出训练集评估指标
print(f"Training Set Metrics:")
print(f"R²: {r2_train:.2f}")
print(f"MAE: {mae_train:.2f}")
print(f"MSE: {mse_train:.2f}")
print(f"RMSE: {rmse_train:.2f}")
print(f"MAPE: {mape_train:.2f}%")
print(f"SMAPE: {smape_train:.2f}%")

# 输出测试集评估指标
print(f"\nTest Set Metrics:")
print(f"R²: {r2_test:.2f}")
print(f"MAE: {mae_test:.2f}")
print(f"MSE: {mse_test:.2f}")
print(f"RMSE: {rmse_test:.2f}")
print(f"MAPE: {mape_test:.2f}%")
print(f"SMAPE: {smape_test:.2f}%")

# 计算残差
residuals = y_test - y_test_pred

# 使用 Seaborn 绘制带核密度估计的残差直方图
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins=20)
plt.title('Residuals Histogram with KDE')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
```
![image](https://github.com/user-attachments/assets/3cdbe0ea-1db6-4ddc-8d75-b7fa933c5903)
### 2、Sentosa_DSML社区版
&emsp;&emsp;模型后可接评估算子，对模型的回归结果进行评估。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2e9ed21e05b5464897acfbdc3a8b4c33.png#pic_center)
&emsp;&emsp;训练集和测试集的评估结果如下所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/935e7f606681469c88c6c2c799e32b92.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/aff3b1e0c2784ba18650cf6de71fac7a.jpeg#pic_center)
&emsp;&emsp;右键查看模型信息，可以得到特征重要性等可视化计算结果。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/425666f62e0643e79bc34f7e8ed63b3b.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/39dada18d75a4d19a9f4bf17afc44ea7.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/03e28195131a42139c2a8ed4b498d8e5.jpeg#pic_center)
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
