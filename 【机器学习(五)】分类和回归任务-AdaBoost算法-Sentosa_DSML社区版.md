# 一、算法概念

什么是 AdaBoost？
&emsp;&emsp;AdaBoost 是 Adaptive Boosting 的缩写，是一种集成机器学习算法，可用于各种分类和回归任务。它是一种监督学习算法，用于通过将多个弱学习器或基学习算法（例如决策树）组合成一个强学习器来对数据进行分类。AdaBoost 的工作原理是根据先前分类的准确性对训练数据集中的实例进行加权，也就是说AdaBoost 构建了一个模型，并为所有数据点分配了相同的权重，然后，它将更大的权重应用于错误分类的点。在模型中，所有权重较大的点都会被赋予更大的权重。它将继续训练模型，直到返回较小的错误。所以，
 - Boosting 是一个迭代的训练过程
 - 后续模型更加关注前一个模型中错误分类的样本 
 - 最终预测是所有预测的加权组合

&emsp;&emsp;如下图所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6c1cb4c0957446dc8161679b4c96080e.png#pic_center)
&emsp;&emsp;以下是 AdaBoost 工作原理的步骤：
 1. 弱学习器：AdaBoost从弱分类器开始，弱分类器是一种相对简单的机器学习模型，其准确率仅略高于随机猜测率。弱分类器通常使用一个浅层决策树。
 2. 权重初始化：训练集中的每个实例最初被分配一个相等的权重。这些权重用于在训练期间赋予更困难的实例更多的重要性。
 3. 迭代（Boosting）：AdaBoost执行迭代来训练一组弱分类器。在每次迭代中，模型都会尝试纠正组合模型到目前为止所犯的错误。在每次迭代中，模型都会为前几次迭代中被错误分类的实例分配更高的权重。
 4. 分类器权重计算：训练较弱的分类器的权重根据其产生的加权误差计算。误差较大的分类器获得较低的权重。
 5. 更新实例权重：错误分类的示例获得更高的权重，而正确分类的示例获得更低的权重。这导致模型在后续迭代中专注于更困难的示例。
 6. 加权组合弱分类器：AdaBoost将加权弱分类器组合生成一个强分类器。这个过程的关键在于，每个弱分类器的权重取决于它在训练过程中的表现，表现好的分类器会得到更高的权重，而表现不佳的分类器会得到较低的权重。
 7. 最终输出：最终输出是所有弱分类器组合而成的强分类器，该最终模型比单个弱分类器准确率更高。

&emsp;&emsp;AdaBoost 的重点在于，通过这个迭代过程，模型能够聚焦于难以分类的样本，从而提升系统的整体性能。AdaBoost 具有很强的鲁棒性，能够适应各种弱模型，是一种强大的集成学习工具。算法流程如下所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9e10b71196a6453780cbbfbef29f5cf9.jpeg#pic_center)
# 一、算法原理
## （一）分类算法基本思路
### 1、训练集和权重初始化
&emsp;&emsp;这一节会详细解释AdaBoost 算法中的公式和推导过程，有助于理解每一步的逻辑及其含义。
训练集和权重初始化
&emsp;&emsp;首先，是训练集和权重初始化部分，
&emsp;&emsp;设训练集为：
$T=\left\{\left(x, y_1\right),\left(x_2, y_2\right), \ldots\left(x_m, y_m\right)\right\}$
&emsp;&emsp;其中，每个样本$x_i∈R^n$。
 
&emsp;&emsp;AdaBoost 算法从样本权重初始化开始。最开始，每个样本被赋予相等的权重：
$$D(t)=\left(w_{t 1}, w_{t 2}, \ldots w_{t m}\right) ; \quad w_{1,i}=\frac{1}{m} ; \quad i=1,2 \ldots m$$
&emsp;&emsp;其中，这里，所有权重$w_{1,i}$的总和为1，表示每个样本的初始重要性是相等的。

### 2、弱分类器的加权误差
&emsp;&emsp;这里假设我们是二元分类问题，在第 t 轮的迭代中，输出为 $\{-1 ， 1\} $， 则第t个弱分类器 $G_t(x)$ 在训统集上的加权误差率为
$$\epsilon_t=P\left(G_t\left(x_i\right) \neq y_i\right)=\sum_{i=1}^m w_{t i} I\left(G_t\left(x_i\right) \neq y_i\right)$$
&emsp;&emsp;这里的$\epsilon_t$是第 t个弱分类器的加权误差率。  $w_{t i}$是样本i在第t轮的权重，反映了在这一轮中该样本的重要性。加权误差表示的是分类错误的样本的权重总和，如果 $\epsilon_t$接近 0，说明分类器表现良好，错误率很低；如果$\epsilon_t$接近 0.5，说明分类器的表现几乎是随机猜测。
### 3、弱分类器的权重
&emsp;&emsp;对于每一轮的弱分类器，权重系数的表达公式为：
$$\alpha_t=\frac{1}{2} \log \frac{1-\epsilon_t}{\epsilon_t}$$
&emsp;&emsp;这个公式表明了弱分类器的表现与其权重之间的关系：

&emsp;&emsp;如果$\epsilon_t$很小，表示弱分类器表现好，那么$\alpha_t$会很大，表示这个分类器在最终组合中有较大的权重。
$\epsilon_t$越接近 0.5，表示分类器效果接近随机猜测，那么$\alpha_t$​越接近 0，表示该分类器在最终组合中权重很小。
如果 $\epsilon_t$大于 0.5，理论上这个分类器的效果是反向的，意味着它的分类错误率超过 50%。因此 AdaBoost 会停止训练。
### 4、Adaboost 分类损失函数
&emsp;&emsp;Adaboost是一种基于加法模型和前向分步算法的分类方法。它通过一系列弱分类器的组合来构建一个强分类器，核心思想是根据分类错误率动态调整样本的权重，使得分类器能更好地处理难以分类的样本。
&emsp;&emsp;在Adaboost中，最终的强分类器 $H_t(x)$ 是若干个弱分类器的加权组合：
&emsp;&emsp;$H(x) = \text{sign}\left(\sum_{t=1}^T \alpha_t G_t(x)\right)$
&emsp;&emsp;$\alpha_i$ 表示第 $i$ 个弱分类器的权重。其中，$G_i(x)$ 是第 $i$ 个弱分类器对输入 $x$ 的预测结果。
&emsp;&emsp;通过前向分步学习方法，强分类器逐步构建为：
$$H_t(x)=H_{t-1}(x)+\alpha_t G_t(x) $$
&emsp;&emsp;Adaboost的损失函数定义为指数损失函数，其公式为：
$$\arg\min_{\alpha,T}\sum_{i=1}^m\exp\left(-y_ih_t(x_i)\right)$$
&emsp;&emsp;其中：
&emsp;&emsp;$\bullet$ $y_i$为样本$i$的真实标签，取值$\{-1,1\}$。
&emsp;&emsp;$\bullet$ $h_t( x_i)$为分类器$h_t$对样本$x_i$的预测结果。
&emsp;&emsp;利用前向分步学习的递推公式$h_t(x)=h_{t-1}(x)+\alpha_tG_t(x)$,损失函数可以改写为：
$$(\alpha_t,G_t(x))=\arg\min_{\alpha,G}\sum_{i=1}^m\exp\left[-y_i\left(h_{t-1}(x_i)+\alpha G(x_i)\right)\right]$$
&emsp;&emsp;定义样本权重$w_{t,i}^{\prime}$为：
$$w_{t,i}'=\exp{(-y_ih_{t-1}(x_i))}$$
&emsp;&emsp;它的值不依赖于$\alpha$和$G$,只与$h_{t-1}(x)$相关。因此，损失函数可以改写为：
$$(\alpha_t,G_t(x))=\arg\min_{\alpha,G}\sum_{i=1}^mw_{t,i}'\exp\left[-y_i\alpha G(x_i)\right]$$
&emsp;&emsp;为了找到最优的弱分类器$G_t(x)$,可以将损失函数展开为：
$$\sum_{i=1}^mw_{t,i}'\exp\left(-y_i\alpha G(x_i)\right)=\sum_{y_i=G_t(x_i)}w_{t,i}'e^{-\alpha}+\sum_{y_i\neq G_t(x_i)}w_{t,i}'e^{\alpha}$$
&emsp;&emsp;由此，可以得到最优弱分类器$G_t(x)$的选择：
$$G_t(x)=\arg\min_G\sum_{i=1}^mw_{t,i}'I(y_i\neq G(x_i))$$
&emsp;&emsp;将$G_t(x)$ 带入损失函数后，对$\alpha$ 求导并令其等于0，可以得到：
$$\alpha_t=\dfrac{1}{2}\log\dfrac{1-e_t}{e_t}$$
&emsp;&emsp;其中，$e_t$为第$t$轮的加权分类误差率：
$$e_t=\frac{\sum_{i=1}^mw_{t,i}'I(y_i\neq G(x_i))}{\sum_{i=1}^mw_{t,i}'}=\sum_{i=1}^mw_{t,i}I(y_i\neq G(x_i))$$
&emsp;&emsp;在第t+1轮中，样本权重会根据弱分类器的表现进行更新。对于分类错误的样本，其权重会增大，从而在下一轮中对这些样本给予更多的关注。通过以上推导，可以得到Adaboost的弱分类器样本权重更新公式。
### 5、样本权重更新
&emsp;&emsp;接下来，计算AdaBoost 更新样本的权重，以便在下一轮训练中更加关注那些被当前弱分类器错误分类的样本。利用$h_t(x)=h_{t-1}(x)+\alpha_tG_t(x)$和$w_{t,i}^{\prime}=\exp(-y_ih_{t-1}(x))$,可以得到样本权重的更新公式为：
$$w_{t+1, i}=\frac{w_{t i}}{Z_T} \exp \left(-\alpha_t y_i G_t\left(x_i\right)\right)$$
&emsp;&emsp;其中，$\alpha_t$ 是第 $t$ 个弱分类器的权重， $y_i$ 是样本 $i$ 的真实标签，$G_t\left(x_i\right)$ 是第 $t$ 个弱分类器对样本 $x_i$ 的预测结果。
&emsp;&emsp;这个公式的作用是通过调整权重来强化难以分类的样本：
- 如果分类器$G_t\left(x_i\right)$对样本$x_i$分类错误，即 $y_i G_t\left(x_i\right)<0$ 会导致 $w_{t+1, i}$ 增大，表示这个样本在下一轮中会被赋予更大的权重，模型会更关注它。
- 如果分类器$G_t\left(x_i\right)$ 对样本 $x_i$ 分类正确，即 $y_i G_t\left(x_i\right)>0$ 会导致 $w_{t+1, i}$ 减小，表示模型认为这个样本已经很好分类了，下轮可以降低它的重要性。

### 6、AdaBoost 的强分类器

&emsp;&emsp;这里 $Z_t$ 是规范化因子，保证更新后的权重仍然是一个概率分布。其计算公式为：
$$Z_t=\sum_{i=1}^m w_{t i} \exp \left(-\alpha_t y_i G_t\left(x_i\right)\right)$$
&emsp;&emsp;通过这个规范化因子，所有的权重从 $w_{t+1, i}$ 被重新调整，使得它们的总和依然为 1。从样本权重更新公式可以看出，分类错误的样本会得到更高的权重，这让下一轮的弱分类器更加关注这些难以分类的样本。这种机制逐步强化了对弱分类器表现不好的部分样本的关注，最终通过多次迭代形成一个强分类器：

$$H(x) = \text{sign}\left(\sum_{t=1}^T \alpha_t G_t(x)\right)$$

## （二）回归算法基本思路
&emsp;&emsp;我们先从回归问题中的误差率定义开始解释，逐步详细分析每个公式和步骤。
### 1、最大误差的计算

&emsp;&emsp;给定第t个弱学习器 $G_t(x)$，其在训练集上的最大误差定义为：
$$E_t=\max \left|y_i-G_t\left(x_i\right)\right| i=1,2 \ldots m$$
&emsp;&emsp;通过计算每个样本上预测值和真实值之间的绝对差值，找到这个差值的最大值。这个最大误差为后续计算每个样本的相对误差提供了一个标准化的尺度，使得每个样本的误差相对该最大误差进行比较。

### 2、相对误差计算

&emsp;&emsp;然后计算毎个样本i的相对误差
$$e_{t i}=\frac{\left|y_i-G_t\left(x_i\right)\right|}{E_t}$$
&emsp;&emsp;通过相对误差，我们可以统一衡量所有样本的误差，而不受特定样本的绝对误差影响。

### 3、误差损失调整

&emsp;&emsp;误差损失可以根据不同的度量方式进行调整。
&emsp;&emsp;如果是线性误差的情况，即直接比较绝对误差：
$$e_{t i}=\frac{\left(y_i-G_t\left(x_i\right)\right)^2}{E_t}$$
&emsp;&emsp;如果使用平方误差，则相对误差为：
$$e_{t i}=\frac{\left(y_i-G_t\left(x_i\right)\right)^2}{E_t^2}$$
&emsp;&emsp;如果我们用的是指数误差，则
$$e_{t i}=1-\exp \left(\frac{\left.-\left|y_i-G_t\left(x_i\right)\right|\right)}{E_t}\right)$$
&emsp;&emsp;指数误差对较大的误差进行了压缩，使其影响变得非线性。
&emsp;&emsp;最终得到第t个弱学习器的误差率
$$e_t=\sum_{i=1}^m w_{t i} e_{t i}$$
&emsp;&emsp;反映了第t个弱学习器在整个训练集上的整体表现
### 4、权重系数计算

&emsp;&emsp;对于第t个弱学习器，权重系数$\alpha_t$的计算公式为：

$$\alpha_t=\frac{e_t}{1-e_t}$$
&emsp;&emsp;这里，权重 $\alpha_t$反映了第 t个弱学习器的重要性。如果误差率 $e_t$小，则$\alpha_t$会较大，表明该弱学习器的重要性较高；反之，误差率大的弱学习器权重较小，这种权重系数分配方法确保了表现更好的弱学习器在组合中获得更大的影响力。
### 5、更新样本权重
&emsp;&emsp;对于更新样本权重D，第t+1个弱学习器的样本集权重系数为：
$$w_{t+1, i}=\frac{w_{t i}}{Z_t} \alpha_t^{1-e_{t i}}$$
&emsp;&emsp;样本权重更新的核心思想是，将更多的关注放在那些难以分类的样本上，以便在后续的训练中重点处理这些样本。
### 6、规范化因子
&emsp;&emsp;这里 $Z_t$ 是规范化因子，规范化因子的计算公式为：

$$Z_t=\sum_{i=1}^m w_{t i} \alpha_t^{1-e_{t i}}$$
&emsp;&emsp;通过这个规范化步骤，保持了样本权重的标准化，使得权重在每一轮迭代中不会无穷增大或减小。
### 7、强学习器
&emsp;&emsp;回归问题与分类问题略有不同，最终的强回归器 f(x)不是简单的加权和，而是通过选择若干弱学习器中的一个，最终的强回归器为：
$$f(x)=G_{t^*}(x)$$
&emsp;&emsp;其中， $G_{t^*}(x)$ 是所有 $\ln \frac{1}{\alpha_t}, t=1,2, \ldots T$ 的中位数值对应序号 $t^*$ 对应的弱学习器。这种方法能够在一定程度上避免极端弱学习器的影响，从而更稳定地进行回归预测。
# 三、算法的优缺点
### 1、优点
&emsp;&emsp;Adaboost算法的主要优点有：　
 - 分类精度高：作为分类器时，Adaboost可以显著提高分类精度。
 - 灵活性强：Adaboost框架下可以使用各种回归或分类模型作为弱学习器，应用广泛。
 - 简单易理解：尤其是用于二元分类时，算法构造简单且结果容易解释。
 -  不易过拟合：相较于其他算法，Adaboost不容易过拟合。
### 2、缺点
&emsp;&emsp;Adaboost的主要缺点有：
 - 对异常样本敏感：在迭代过程中，异常样本可能获得过高的权重，影响最终模型的预测效果。

&emsp;&emsp;此外，虽然理论上任何学习器都可以作为弱学习器，但实践中最常用的弱学习器是决策树和神经网络。在分类任务中，Adaboost通常使用CART分类树；在回归任务中，则使用CART回归树。

# 四、Adaboost分类任务实现对比
&emsp;&emsp;主要根据模型搭建的流程，对比传统代码方式和利用Sentosa_DSML社区版完成机器学习算法时的区别。
## （一）数据加载 
### 1、Python代码
```python
#导入库
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

#加载数据集
iris = load_iris()
X = iris.data
y = iris.target
```
&emsp;&emsp;在此步骤中，我们导入必要的库。AdaBoostClassifier 是用于实现 AdaBoost 的 scikit-learn 类，DecisionTreeClassifier 是基本的弱分类器（在本例中为浅层决策树），其他库用于数据管理和性能评估。
### 2、Sentosa_DSML社区版
&emsp;&emsp;利用文本算子读入数据。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a0125de1069943ed83e178e50cedb241.png#pic_center)
## （二）样本分区
&emsp;&emsp;此步骤将数据集分为训练集和测试集。20％的数据用作测试集，而80％用于训练模型。
### 1、Python代码
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
### 2、Sentosa_DSML社区版
&emsp;&emsp;利用样本分区算子划分训练集和测试集
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/307e0cfe1a3740f3936db0bc77b41a13.png#pic_center)
&emsp;&emsp;利用类型算子设置数据的标签列和特征列
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/72cb7c3df99c4b55bbbf2d949a6b183f.png#pic_center)
## （三）模型训练
### 1、Python代码

```python
#配置弱分类器和AdaBoost分类器。n_estimators指定要训练的迭代次数（弱分类器）

weak_classifier = DecisionTreeClassifier(max_depth=5,   
                                  	     max_leaf_nodes=None,  
                                         random_state=42)

# 设置 AdaBoostClassifier 的参数
n_estimators = 50
learning_rate = 1.0

# 创建 AdaBoost 分类器
adaboost_classifier = AdaBoostClassifier(estimator=weak_classifier,
                                         n_estimators=n_estimators,
                                         learning_rate=learning_rate,
                                         random_state=42)
#训练 AdaBoost 模型
adaboost_classifier.fit(X_train, y_train)
```
### 2、Sentosa_DSML社区版
&emsp;&emsp;使用AdaBoost分类算子进行模型训练。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/245339ba7aeb47ad9b83193444555898.png#pic_center)
&emsp;&emsp;执行完成后得到训练结果
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a5ade4fc3d5044fd87230c51fbea6530.jpeg#pic_center)
## （四）模型评估
### 1、Python代码

```python
#对测试集进行预测，并通过将预测与真实标签进行比较来计算模型的准确性等指标
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
# 评估模型
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')
# 打印评估结果
print(f"AdaBoost 模型的准确率: {accuracy:.2f}")
print(f"加权精度 (Weighted Precision): {precision:.2f}")
print(f"加权召回率 (Weighted Recall): {recall:.2f}")
print(f"F1 值 (Weighted F1 Score): {f1:.2f}")

# 计算混淆矩阵

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
```
### 2、Sentosa_DSML社区版
&emsp;&emsp;利用评估算子对模型进行评估
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/43588c0a416f493194489eb81f145068.jpeg#pic_center)
&emsp;&emsp;评估完成后得到模型训练集和验证集上的的评估结果。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a039addb45c8450b9765e67812e5f23b.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fe84992edfe64e71a2739c45b8bf071b.jpeg#pic_center)
&emsp;&emsp;利用混淆矩阵算子可以计算模型在训练集和测试集上的混淆矩阵
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8993878bf3c24452ba8d9756f6b5bcaa.jpeg#pic_center)
&emsp;&emsp;结果如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/97bba5e0939146eab9999dbd2d581059.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d09a963b5e2e4bd48a53b4b8c8996381.jpeg#pic_center)
## （四）模型可视化
### 1、Python代码
```python
# 计算特征重要性并进行排序
importances = adaboost_classifier.feature_importances_
indices = np.argsort(importances)[::-1]

# 绘制特征重要性柱状图
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [iris.feature_names[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/71ac4493d93a4211bd734fdfb47dd1b2.png)
### 2、Sentosa_DSML社区版
&emsp;&emsp;右键查看模型信息即可得到特征重要性排序图和模型可视化结果等信息，模型信息如下图所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2df65333ec154585abd4c18d04157cc6.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1731fd8e9cb1402885dd4b4641eb7de0.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e38365e982084e4d98de81098d047759.jpeg#pic_center)
# 四、Adaboost回归任务实现对比
&emsp;&emsp;主要根据回归模型搭建的流程，对比传统代码方式和利用Sentosa_DSML社区版完成机器学习算法时的区别。
## （一）数据加载、样本分区和特征标准化
### 1、Python代码
```python
#导入库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 读取数据集
df = pd.read_csv("D:/sentosa_ML/Sentosa_DSML/mlServer/TestData/winequality.csv")

# 将数据集划分为特征和标签
X = df.drop("quality", axis=1)  # 假设标签是 "quality"
Y = df["quality"]

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 标准化特征
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

```
### 2、Sentosa_DSML社区版
&emsp;&emsp;与上一节数据处理和样本分区操作类似，首先，利用文本算子读入数据，然后，连接样本分区算子划分训练集和测试集，其次，使用类型算子设置数据的标签列和特征列。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/506eddd50b1b4ddabc07fce2302a7d93.jpeg#pic_center)
&emsp;&emsp;接下来,连接标准化算子对数据集进行标准化处理
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/16b317d2eb6b44fb91433225bf4025f4.png#pic_center)
&emsp;&emsp;执行得到标准化模型
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c3609996e23942688d614cbd4762e7e5.jpeg#pic_center)
# （二）模型训练
### 1、Python代码

```python
base_regressor = DecisionTreeRegressor(
    max_depth=8,               # 树的最大深度
    min_samples_split=1,        # 最小实例数
    min_impurity_decrease=0.0,  # 最小信息增益
    max_features=None,          # 不限制最大分桶数
    random_state=42
)

# 设置 AdaBoost 回归器的参数
adaboost_regressor  = AdaBoostRegressor(
    estimator=base_regressor,
    n_estimators=100,          # 最大迭代次数为100
    learning_rate=0.1,         # 步长为0.1
    loss='square',             # 最小化的损失函数为 squared error
    random_state=42
)

# 训练模型
adaboost_regressor.fit(X_train, Y_train)

# 预测测试集上的标签
y_pred = adaboost_regressor.predict(X_test)
```
### 2、Sentosa_DSML社区版
&emsp;&emsp;连接AdaBoost回归算子
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d8098e44e6214fc9972491f2497a0fa6.png#pic_center)
&emsp;&emsp;执行得到回归模型
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1b31eb4e54c0416d992e55137f7024e6.jpeg#pic_center)
# （三）模型评估
### 1、Python代码
```python
# 计算评估指标
r2 = r2_score(Y_test, y_pred)
mae = mean_absolute_error(Y_test, y_pred)
mse = mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((Y_test - y_pred) / Y_test)) * 100
smape = 100 / len(Y_test) * np.sum(2 * np.abs(Y_test - y_pred) / (np.abs(Y_test) + np.abs(y_pred)))

# 打印评估结果
print(f"R²: {r2}")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}%")
print(f"SMAPE: {smape}%")
```
### 2、Sentosa_DSML社区版
&emsp;&emsp;连接评估算子对模型进行评估
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c91cd129111b41e3939dad589ff9e9cb.png#pic_center)
&emsp;&emsp;训练集和测试集的评估结果如下图所示:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/62ae15845b3f49f493e337f7a4879fc1.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2fdf791e58b246e2afdb15b925385e05.jpeg#pic_center)
# （四）模型可视化
### 1、Python代码

```python
# 可视化特征重要性
importances = adaboost_regressor.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=45)
plt.tight_layout()
plt.show()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/916c8f0926854b7d9b394ec88c68f204.png)
### 2、Sentosa_DSML社区版
&emsp;&emsp;右键即可查看模型信息,下图为模型特征重要性和残差直方图:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a702e07cb0814f8abdc6b52324177bc0.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/43e09a5eb0c745eebd0086196918870d.jpeg#pic_center)
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

&emsp;&emsp;欢迎访问Sentosa_DSML社区版的官网https://sentosa.znv.com/
免费下载体验。同时，我们在B站、CSDN、知乎、博客园等平台有技术讨论博客和应用案例分享，欢迎广大数据分析爱好者前往交流讨论。

&emsp;&emsp;Sentosa_DSML社区版，重塑数据分析新纪元，以可视化拖拽方式指尖轻触解锁数据深层价值，让数据挖掘与分析跃升至艺术境界，释放思维潜能，专注洞察未来。
社区版官网下载地址：https://sentosa.znv.com/
社区版官方论坛地址：http://sentosaml.znv.com/
B站地址：https://space.bilibili.com/3546633820179281
CSDN地址：https://blog.csdn.net/qq_45586013?spm=1000.2115.3001.5343
知乎地址：https://www.zhihu.com/people/kennethfeng-che/posts
博客园地址：https://www.cnblogs.com/KennethYuen
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5ad97144846d4bb5a9ea5dd3d4667e54.jpeg#pic_center)

