@[toc]
# 一、算法概念

&emsp;&emsp;首先，我们需要了解，什么是决策树？
&emsp;&emsp;开发一个简单的算法来对三个元素A、B和C进行排序，需要把这个问题拆解成更小的子问题。首先，考虑A是否小于B。接下来，如果B小于C，那么事情会变得有趣：如果A<B且B<C，那么显然A<B<C。但如果B不小于C，第三个问题就变得关键：A是否小于C？这个问题也许用图形的方式来解决问题会更好。
&emsp;&emsp;于是，我们可以为每个问题画一个节点，为每个答案画一条边，所有的叶子节点都代表一种可能的正确排序。下图展示了我们绘制的决策树图：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f688e54cacae496db0a3b089b0746dde.png#pic_center)

&emsp;&emsp;这是我们创建的第一个决策树。为了做出正确的决定，需要两到三个if语句，因此，算法流程如下所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/04e6dd0b0de64d0b8d2bbe114a8354db.jpeg#pic_center)
&emsp;&emsp;排序代码：
```python
def sort2(A, B, C):
if (A < B) and (B < C) : return [A, B, C]
if (A < B) and not (B < C) and (A < C): return [A, C, B]
if (A < B) and not (B < C) and not (A < C): return [C, A, B]
if not (A < B) and (B < C) and (A < C): return [B, A, C]
if not (A < B) and (B < C) and not (A < C): return [B, C, A]
if not (A < B) and not (B < C) : return [C, B, A]
```
&emsp;&emsp;正如预期的那样，排序问题给我们提供了6条规则，对应于6种不同的排列组合，最终结果与决策树算法一致。这个简单的例子展示了决策树的四个重要特点：
&emsp;&emsp;1、非常适合用于分类和数据挖掘。
&emsp;&emsp;2、直观且易于理解。
&emsp;&emsp;3、实现起来相对简单。
&emsp;&emsp;4、适用于不同人群。
&emsp;&emsp;这进一步证明了决策树在处理简单排序或分类问题时的高效性和易用性。
	树（tree）是一种具有一个根节点的有向连通图。每个其他节点都有一个前任节点（父节点），没有或多个后继节点（子节点）。没有后继的节点称为叶子。所有节点都通过边连接。节点的深度是到根的路径上的边数。整棵树的高度是从根到任意叶子的最长路径上的边数。
决策树是一种树形结构，其元素的对应关系如下：
|树形结构  | 决策树中的对应元素 |
|--|--|
| 根节点 | 初始决策节点 |
| 节点 | 用于测试属性的内部决策节点 |
| 边 | 需要遵循的规则 |
| 叶子节点 | 表示结果分类的终端节点 |

&emsp;&emsp;机器学习算法的输入由一组实例（例如行、示例或观察）组成。每个实例都由固定数量的属性（即列）和一个称为类的标签（在分类任务的情况下）来描述，这些属性被假定为名义或数字。所有实例的集合称为数据集。
&emsp;&emsp;根据这个定义，我们得到一个包含数据集的表：每个决策都成为一个属性（所有二元关系），所有叶子都是类，而每一行代表数据集的一个实例。如下图所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/58b9e44e23d64d9a9f20d6bf38e62c8a.jpeg#pic_center)
&emsp;&emsp;数据以表格形式（例如数据库）收集，并且必须生成决策树。现在有八个而不是六个类的原因很简单：对于实例1、2和7、8来说，无论A<B与否，都没有关系，结果是同一个。这种删除树上不相关分支的效果称为剪枝。
# 二、算法原理
## （一）树的构造

&emsp;&emsp;基于树的分类方法的核心思想源自于一种概念学习系统。接下来将介绍的一系列算法，都基于一个简单但非常强大的算法，叫做TDIDT，代表“自上而下归纳决策树”。该算法的框架包含两种主要方法：决策树的生长和修剪。这两种方法将会在接下来的两个伪代码中详细展示，它们遵循分而治之的思想。
&emsp;&emsp;通过递归分裂特征空间，创建复杂的决策树，使其在训练数据上取得较高的拟合度。其目的是最大化在训练数据上的准确性。决策树的构建过程：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/902c17bea9924c58828b710b7f6d7b2b.jpeg#pic_center)
&emsp;&emsp;通过剪掉不必要的节点来简化树结构，防止过拟合，目的是提高模型对新数据的泛化能力。决策树的修建过程：![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/19ef404fed12453d851ca115f8a66c5f.jpeg#pic_center)

## （二）划分选择
&emsp;&emsp;在构建决策树的过程中，划分选择（也叫做分裂选择）是决定如何根据数据的特征来划分节点的关键步骤。这个过程涉及选择一个特征以及相应的分割阈值或类别，使得数据集在该节点被分割为尽可能纯净的子集。不同的划分选择方法使用不同的准则来衡量数据分割的质量。设N个训练样本为：
![image](https://github.com/user-attachments/assets/e21349b5-cd15-471c-a34e-9382e2be27ae)
### 1、信息增益
&emsp;&emsp;熵和信息增益信息增益的思想基于 Claude Elwood Shannon于1948年提出的信息论。信息增益基于熵的概念。熵是衡量数据集不确定性或混乱度的一种度量。信息增益的目标是通过划分数据减少这种不确定性。
&emsp;&emsp;熵的定义如下：
![image](https://github.com/user-attachments/assets/3b4d7a21-1369-459e-bf4f-774561d8e67a)
&emsp;&emsp;信息增益是划分前后的熵的减少量：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/497ba33dc87b4bb792ca0de2aa65753f.jpeg#pic_center)
&emsp;&emsp;其中，第一项是熵，第二项是子节点的加权熵。因此，该差异反映了熵的减少或从使用属性获得的信息。
### 2、基尼指数
&emsp;&emsp;基尼指数衡量目标属性值的概率分布之间的差异，定义如下：
![image](https://github.com/user-attachments/assets/e39403af-c1c3-4c07-a82b-6260e45109a7)
&emsp;&emsp;目标是找到一个最“纯粹”的节点，即具有单个类的实例。与基于信息增益的标准中使用的熵和增益信息的减少类似，所选择的属性是杂质减少最多的属性。
### 3、卡方检验
&emsp;&emsp;卡方统计 (χ2) 标准基于将由于分裂而获得的类别频率值与该类别的先验频率进行比较。χ2 值的计算公式为：
![image](https://github.com/user-attachments/assets/45e48a61-acbb-4a51-86ad-75bb717fd4ea)
&emsp;&emsp;其中，![image](https://github.com/user-attachments/assets/197e5579-9375-424c-a391-10a506fca032)
&emsp;&emsp;是k中样本N的先验频率。χ2值越大，表明分割越均匀，即来自某个类的实例的频率更高。通过χ2的最大值选择属性。
&emsp;&emsp;划分选择的好坏直接影响决策树的性能。选择恰当的划分标准有助于生成更优的决策树，提高模型的预测准确性，减少过拟合的风险。在实际应用中，信息增益和基尼指数是最常见的选择，具体使用哪种方法往往取决于任务的类型（分类或回归）和数据的性质。
## （三）停止标准
&emsp;&emsp;在构建决策树的过程中，停止标准（Stopping Criteria）决定了什么时候应该停止继续分裂节点。合理的停止标准可以防止决策树过度拟合数据，生成过于复杂的树结构，同时也有助于提升模型的泛化能力。以下是决策树常用的几种停止标准：

 - 训练集中的所有实例都属于单个y值。 
 - 已达到最大树深度。 
 - 终端节点中的事例数小于父节点的最小事例数。
 
 - 如果节点被拆分，则一个或多个子节点中的事例数将小于子节点的最小事例数。     
 - 最佳分割标准不大于某个阈值。

## （四）剪枝处理

&emsp;&emsp;使用严格的停止标准往往会创建小型且不合适的决策树。另一方面，使用松散停止标准往往会生成过度拟合训练集的大型决策树。为了避免这两个极端，人们提出了剪枝的想法：使用宽松的停止标准，在生长阶段之后，通过删除对泛化精度没有贡献的子分支，将过度拟合的树剪回较小的树。
### 1、预剪枝
&emsp;&emsp;预剪枝是在决策树构建过程中，提前停止分裂节点的过程。当某些停止条件满足时，不再进一步分裂节点，这种方法防止了树的过度生长。常见的预剪枝策略包括：

 1. 最大深度限制：预设树的最大深度，一旦树达到该深度，即停止继续分裂。
 2. 最小样本数限制：每个叶节点中必须包含的最小样本数量，当节点中的样本数量低于该阈值时停止分裂。
 3.  最小信息增益或基尼增益限制：如果分裂带来的信息增益或基尼增益低于某个阈值，则停止分裂。
 4.  叶节点的纯度阈值：如果节点中某一类别占比超过某个预设的阈值，停止分裂，并将节点作为叶节点。

&emsp;&emsp;预剪枝的优点：
&emsp;&emsp;计算效率高：由于是在决策树生成过程中进行剪枝，可以减少不必要的分裂，从而加速模型训练。
&emsp;&emsp;减少过拟合：通过提前停止分裂，预剪枝可以有效防止决策树过度拟合训练数据。

&emsp;&emsp;预剪枝的缺点：
&emsp;&emsp;欠拟合风险：预剪枝可能会过早停止分裂，导致决策树没有充分学习数据中的模式，进而出现欠拟合问题。
### 2、后剪枝
&emsp;&emsp;后剪枝是指先构建出一棵完整的决策树，然后再对树进行简化。后剪枝的过程是从最底层的叶节点开始，逐层向上评估是否可以删除某些子树或分支，并将其替换为叶节点。常见的后剪枝方法包括：

 1. 子树替换（Subtree Replacement）：当发现某一子树的分支对模型的预测贡献不大时，将该子树替换为叶节点，即直接使用该子树中的多数类作为分类结果。
 2. 子树升高（Subtree Raising）：通过将子树的一部分提升到其父节点来简化树结构。具体来说，若子树中某一节点的子节点可以代替该节点的父节点，则将其提升。
 3. 成本复杂度剪枝（Cost Complexity Pruning）：该方法通过增加一个正则化参数，权衡树的复杂性和模型的预测误差。在实际操作中，会对剪枝后的决策树进行交叉验证，选择使得预测误差最小的子树。

&emsp;&emsp;后剪枝的优点：
&emsp;&emsp;更有效的泛化能力：通过后剪枝，可以显著减少过拟合，使模型在新数据上的预测能力更强。
&emsp;&emsp;保留更多的信息：由于后剪枝是在生成完整决策树后进行的，可以确保模型尽可能地利用数据中的信息，减少过早终止分裂的风险。
&emsp;&emsp;后剪枝的缺点：
&emsp;&emsp;计算成本高：后剪枝通常需要先生成一棵完全生长的决策树，这会增加计算成本，特别是当数据集较大时。
&emsp;&emsp;复杂性较高：后剪枝的剪枝策略和评估准则通常比预剪枝复杂，可能需要更多的调参和计算。
&emsp;&emsp;剪枝的具体算法

 1. 贪心剪枝：根据子树的预测误差决定是否剪枝，比较剪枝前后的性能，保留更优的结构。
 2. CART 剪枝算法：基于成本复杂度的剪枝算法，结合树的复杂度和误差表现。通过交叉验证找出最优的剪枝深度。 
 3. Reduced Error Pruning（减少错误剪枝）：从完全生长的树开始，测试每个节点的剪枝效果，如果剪枝不会增加误差率，则保留剪枝后的结构。
# 三、决策树的优缺点

**优点：**

 1. 直观易懂，可以转换为一组规则，便于快速解释、理解和实现：从根节点到叶节点的路径即为分类的解释，叶节点代表最终的分类结果。
 2. 能够处理标称（分类）和数值型输入数据。
 3.  可以处理含有缺失值的数据集。
 4.  作为一种非参数方法，它不依赖于数据的空间分布或分类器的结构假设。

**缺点：**

 1. 决策树对训练集中的噪声、不相关的属性过于敏感。 
 2. 要求目标属性由离散值组成。
 3. 当属性之间高度相关时，决策树通常表现良好，因为它采用分而治之的方法；但当存在许多复杂的相互作用时，决策树的表现就不如预期那么理想。

# 四、决策树分类任务实现对比
&emsp;&emsp;主要根据模型搭建的流程，对比传统代码方式和利用Sentosa_DSML社区版完成机器学习算法时的区别。

## （一）数据加载

### 1、Python代码

```python
import pandas as pd

#读取样本数据
data = pd.read_csv('./TestData/iris.csv')
```


```python
from sklearn import datasets

# 读取iris数据集
iris = datasets.load_iris()
x = iris.data
y = iris.target
```
### 2、Sentosa_DSML社区版
&emsp;&emsp;拖拽数据读入算子，选择数据文件路径，选择读取格式等，点击应用读入数据。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/796eab6f558a44e084230994392fabdb.png#pic_center)

## （二）样本分区

1、Python代码

```python
import pandas as pd from sklearn.model_selection 

# 特征和标签分离 
x = data.drop('species', axis=1) 
y = data['species']

# 分割数据集，测试集比例是 20%，训练集比例是 80% 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) 

# 输出训练集和测试集的样本数 
print("训练集样本数:", len(x_train)) 
print("测试集样本数:", len(x_test))
```


```xml
train_test_split()参数说明

x 表示特征数据
y 表示目标变量

train_test_split()参数说明
test_size 表示测试集的大小，如0.2表示20%的数据用于测试，在这段代码中，train_test_split() 函数将数据集 x 和 y 按照 80% 的训练集和 20% 的测试集比例进行分割。
random_state 表示随机参数种子，用于控制数据集的随机划分过程，保证每次划分的结果都一样，在这段代码中，train_test_split() 函数并没有指定 random_state 参数，因此默认情况下，random_state 是 None。
train_test_split函数的返回值包括：训练集的特征数据x_train、测试集的特征数据x_test、训练集的目标变量y_train、测试集的目标变量y_test
```

### 2、Sentosa_DSML社区版

&emsp;&emsp;连接类型和样本分区算子，划分训练集和测试集数据。
&emsp;&emsp;首先，连接样本分区算子可以选择数据训练集和测试集划分比例。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/247a1a91b7fc458184ce4bfe06ac5f5c.png#pic_center)
&emsp;&emsp;右键预览可以看到数据划分结果。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9759722939534c7c9c4a0fe3ea1b0ae2.png#pic_center)
&emsp;&emsp;其次，连接类型算子将Species列的模型类型设为Label标签列。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/eb485e9b693941a884c46dc86e244400.png#pic_center)

## （三）模型训练

### 1、Python代码

```python
from sklearn.tree import DecisionTreeClassifier

# 实例化决策树分类器，并指定一些参数
clf = DecisionTreeClassifier(
    criterion='entropy',       # 'entropy' 表示使用信息增益来衡量分裂质量，选择信息增益最大的特征进行分裂
    max_depth=5,               # 限制决策树的最大深度为5，以防止过拟合（树不允许深度超过5层）
    min_samples_split=10,      # 内部节点分裂所需的最小样本数，至少需要10个样本才能分裂该节点
    random_state=42            # 固定随机种子，保证每次运行代码时得到的决策树模型相同（可复现）
)

# 使用训练数据拟合模型
clf.fit(x_train, y_train)      # 使用训练数据 x_train（特征） 和 y_train（目标变量） 来训练决策树模型

# 在测试集上进行预测
y_pred = clf.predict(x_test)
```

### 2）Sentosa_DSML社区版
&emsp;&emsp;样本分区完成后，连接决策树分类算子，双击在右侧进行模型属性配置。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5fe6001caf414674bbdaaeb136d040a1.jpeg#pic_center)
&emsp;&emsp;右键执行，训练完成后得到决策树分类模型。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d1b0887ce0984dd8b7af095f36c7ce04.jpeg#pic_center)

## （四）模型评估
### 1、Python代码

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

# 计算并输出模型评估指标
accuracy = accuracy_score(y_test, y_pred)                            # 准确率
precision = precision_score(y_test, y_pred, average='weighted')      # 加权精度
recall = recall_score(y_test, y_pred, average='weighted')            # 加权召回率
f1 = f1_score(y_test, y_pred, average='weighted')                    # 加权F1分数

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# 计算每个类别的 ROC 曲线和 AUC 值
y_score = clf.predict_proba(x_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):  
    fpr[i], tpr[i], _ = roc_curve(y_test, y_score[:, i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])

# 绘制每个类别的 ROC 曲线
plt.figure()
for i in range(3):
    plt.plot(fpr[i], tpr[i], label='ROC curve for class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# 输出评估结果
print(f"Accuracy: {accuracy:.4f}")    # 输出准确率
print(f"Precision: {precision:.4f}")  # 输出加权精度
print(f"Recall: {recall:.4f}")        # 输出加权召回率
print(f"F1 Score: {f1:.4f}")          # 输出加权F1分数
```

### 2、Sentosa_DSML社区版

&emsp;&emsp;利用评估算子对模型进行评估
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/51e8f591908d46a39531130ffb5dd0d6.jpeg#pic_center)
&emsp;&emsp;训练集评估结果
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/960c4dbe63f145ba9d2ccc88e1a1e857.jpeg#pic_center)
&emsp;&emsp;测试集评估结果![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4cbe6a1473f84141b64f3e1dca2ab13b.jpeg#pic_center)
&emsp;&emsp;利用混淆矩阵评估算子对模型进行评估
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/995b1b23891b43ecb2da2491ca79170c.jpeg#pic_center)
&emsp;&emsp;训练集评估结果:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8356b75e3dc5448eae1fe00e7a97c406.jpeg#pic_center)
&emsp;&emsp;测试集评估结果:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/00f9c5120f3a4e44afa365d35ead8acc.jpeg#pic_center)
&emsp;&emsp;利用ROC-AUC算子对模型进行评估
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/912395fa217b44fd8cf34334bd37bc0e.jpeg#pic_center)
&emsp;&emsp;ROC-AUC算子评估结果：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0cd2a339c87f4fda8018339c37169f2c.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c02773dd7fba4eb1a6be0ca78a9f226d.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/beb399b4c0ce4c1484ce8751ce8c00d2.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/52366d7ea6514e149030a8920a17070f.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/96f2bfcd54b94cc9bfca6b9d0d41422d.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d83ff5c593494b8681b79d74fe1204ad.png#pic_center)

## （五）模型可视化
### 1、Python代码

```python
import matplotlib.pyplot as plt

# 显示重要特征
plot_importance(clf)
plt.title('Feature Importance')
plt.show()

```

### 2、Sentosa_DSML社区版
&emsp;&emsp;右键决策树分类模型即可查看模型信息，模型特征重要性、混淆矩阵和决策树可视化如下图所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/69cf3c76a77d46929c7622cd69eff5b8.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7030c40e7e394493be57704ad31aa5ca.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/893223a3de7344fab18b16f7a333bafa.jpeg#pic_center)

# 五、决策树回归任务实现对比
## （一）数据加载和样本分区

&emsp;&emsp;数据加载和样本分区同上

### 1、Python代码

```python
# 读取数据
data = pd.read_csv('./TestData/winequality.csv')
#特征和标签分离
x = data.drop('quality', axis=1)
y = data['quality']
#分割数据集，测试集比例是 20%，训练集比例是 80%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# 输出训练集和测试集的样本数 
print("训练集样本数:", len(x_train)) 
print("测试集样本数:", len(x_test))
```
b）Sentosa_DSML社区版
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/147f66e157774cccb01c7c9aee397cf6.png#pic_center)
## （二）模型训练

### 1、Python代码

```python
from sklearn.tree import DecisionTreeRegressor

# 实例化决策树回归模型
regressor = DecisionTreeRegressor(
    criterion='mse',         # 损失函数使用均方误差（MSE）
    max_depth=5,             # 限制回归树的最大深度为 5，以防止过拟合
    min_samples_split=10,    # 内部节点分裂所需的最小样本数
    random_state=42          # 固定随机种子，保证每次运行代码时得到的回归树模型相同
)

# 使用训练数据拟合模型
regressor.fit(x_train, y_train)

# 在测试集上进行预测 
y_pred = regressor.predict(x_test)
```

### 2、Sentosa_DSML社区版

&emsp;&emsp;样本分区完成后，连接决策树回归算子，进行模型属性配置并执行，得到决策树回归模型。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b5afaff5603c4ae2b94e9c3607a1f7d0.jpeg#pic_center)
## （三）模型评估
### 1、Python代码

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# 预测结果
y_pred = regressor.predict(x_test)
# 计算 R^2
r2 = r2_score(y_test, y_pred)
# 计算 MSE（均方误差）
mse = mean_squared_error(y_test, y_pred)
# 计算 RMSE（均方根误差）
rmse = np.sqrt(mse)
# 计算 MAE（平均绝对误差）
mae = mean_absolute_error(y_test, y_pred)
# 计算 MAPE（平均绝对百分比误差）
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
# 计算 SMAPE（对称平均绝对百分比误差）
smape = np.mean(2 * np.abs(y_pred - y_test) / (np.abs(y_pred) + np.abs(y_test))) * 100

# 输出评估结果
print(f"R²: {r2:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.4f}%")
print(f"SMAPE: {smape:.4f}%")
```

### 2、Sentosa_DSML社区版

&emsp;&emsp;利用评估算子对模型进行评估
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a2346b9c83224431af66814a16dc23ba.jpeg#pic_center)

&emsp;&emsp;训练集评估结果
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/84af7824ec684cf487f0fb3f7c3d2952.jpeg#pic_center)
&emsp;&emsp;测试集评估结果
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8c59a91dfaac4c0f97e1905e576d12fb.jpeg#pic_center)

## （四）模型可视化

### 1、Python代码
```python
#可视化特征重要性
plot_importance(regressor)
plt.title('Feature Importance')
plt.show()
```
### 2、Sentosa_DSML社区版
&emsp;&emsp;右键决策树回归模型即可查看模型信息：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9d5dec01eb644604866c8c24496e197e.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4c92e48f7a84481092e5f37ef9a0119e.jpeg#pic_center)

# 六、总结

&emsp;&emsp;相比传统代码方式，利用Sentosa_DSML社区版完成机器学习算法的流程更加高效和自动化，传统方式需要手动编写大量代码来处理数据清洗、特征工程、模型训练与评估，而在Sentosa_DSML社区版中，这些步骤可以通过可视化界面、预构建模块和自动化流程来简化，有效的降低了技术门槛，非专业开发者也能通过拖拽和配置的方式开发应用，减少了对专业开发人员的依赖。
&emsp;&emsp;Sentosa_DSML社区版提供了易于配置的算子流，减少了编写和调试代码的时间，并提升了模型开发和部署的效率，由于应用的结构更清晰，维护和更新变得更加容易，且平台通常会提供版本控制和更新功能，使得应用的持续改进更为便捷。

&emsp;&emsp;为了回馈社会，实现AI普惠，进一步降低AI实践门槛，为广大师生学者、科研工作者及开发者提供学习、交流及实践机器学习技术，推出了一款**轻量化安装且完全免费的Sentosa_DSML社区版软件**，该软件包含了Sentosa_DSML数据科学与机器学习平台中机器学习平台（Sentosa_ML）的大部分功能，以轻量化一键安装、永久免费使用、视频教学服务和社区论坛交流为主要特点，同样支持“拖拉拽”开发，旨在通过零代码方式帮助客户解决学习、生产和生活中的实际痛点问题。
&emsp;&emsp;该软件为基于通用人工智能的数据分析工具，可以赋能各行各业。应用范围非常广泛，以下是一些主要应用领域：
&emsp;&emsp;**金融风控**：用于信用评分、欺诈检测、风险预警等，降低投资风险；
&emsp;&emsp;**股票分析**：预测股票价格走势，提供投资决策支持；
&emsp;&emsp;**医疗诊断**：辅助医生进行疾病诊断，如癌症检测、疾病预测等；
&emsp;&emsp;**药物研发**：进行分子结构的分析和药物效果预测，帮助加速药物研发过程；
&emsp;&emsp;**质量控制**：检测产品缺陷，提高产品质量；
&emsp;&emsp;**故障预测**：预测设备故障，减少停机时间；
&emsp;&emsp;**设备维护**：通过分析机器的传感器数据，检测设备的异常行为；
&emsp;&emsp;**环境保护**：用于气象预测、大气污染监测等。
&emsp;&emsp;欢迎访问官网https://sentosa.znv.com/
下载体验Sentosa_DSML社区版。同时，我们在B站、CSDN、知乎、博客园等平台也有技术讨论博客和文章，欢迎广大数据分析爱好者前往交流讨论。
&emsp;&emsp;**Sentosa_DSML社区版，重塑数据分析新纪元，以可视化拖拽方式指尖轻触解锁数据深层价值，让数据挖掘与分析跃升至艺术境界，释放思维潜能，专注洞察未来。**

[Sentosa_DSML社区版官网](https://sentosa.znv.com/)

社区版官网下载地址：https://sentosa.znv.com/
B站地址：https://space.bilibili.com/3546633820179281
CSDN地址：https://blog.csdn.net/qq_45586013?spm=1000.2115.3001.5343
知乎地址：https://www.zhihu.com/people/kennethfeng-che/posts
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c4a78ee3e6104732861ca17feb8d269f.png)


