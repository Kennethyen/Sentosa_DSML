@[toc]
# 第二章：Sentosa_DSML社区版产品功能
## 1.公共功能

进入算子流构建页面如图所示，包括**算子控制区、算子流控制区、主题切换、算子集合区和画布**，算子集合区是所有算子分类集合的区域，画布通过从算子集合区拖拽算子到画布上构建和设置算子及算子流属性的区域。

![Sentosa_DSML社区版界面](https://i-blog.csdnimg.cn/direct/b6ec77d7131045359c8eacf8de46fb8e.png#pic_center)

## 2.算子功能
Sentosa_DSML社区版现阶段提供共计120个算子，细分为11类：数据读入、数据写出、行处理、列处理、数据融合、统计分析、特征工程、机器学习、线性规划、图表分析、扩展编程。从在数据分析中的作用也可分为：数据读入，数据处理，机器学习、线性规划、图表分析和数据写出几大类。
（1）数据读入类算子主要是提供从本地文件、文件系统、数据库及流式数据源读入数据，其中文本算子和EXCEL算子还提供了文件上传功能，可以将客户端的文件上传到服务器上或者HDFS上，并在构建算子流中使用。在以下部分读入算子属性配置中，增加了删除重名列配置，在配置读入属性后可通过算子内置删除重命名列功能，对数据列进行修改。

![数据读入](https://i-blog.csdnimg.cn/direct/517863a6b156473287f2424c949d6b5a.png#pic_center)

 	Hive数据库读入(HiveSourceNode)	从Hive数据库读取数据。
	数据库连接(JDBCSourceNode)	支持mysql、oracle、db2、sqlserver、postgresql五种数据库读取数据。
	文本(FileSourceNode)	支持从本地文件和HDFS中读取数据。同时支持上传文件到服务器本地或者hdfs上，支持在目录下创建新的文件夹。
	Exce读入(ExcelSourceNode)	读取Excel数据。同时上传文件到服务器本地或者HDFS上，支持在目录下创建新的文件夹。
	拟合数据生成(FitDataGenerateNode)	用来按照不同分布方式生成不同类型的数据。
	随机数据生成(RandomDataGetNode)	用来随机生成不同类型的数据。
	马尔科夫数据源(MarkovSourceNode)	生成马尔科夫链数据。
	Kafka读入(Kafa2DatasetNode)	从Kafka读取数据。
	Hbase读入(HbaseSourceNode)	从HBase数据库读取数据。
	XML读入(XMLSourceNode)	支持从本地文件和HDFS中读取XML数据。同时支持上传文件到服务器本地或者HDFS上。
	ES读入(ESSourceNode)	从ES数据库读取数据。
	
（2）数据写出类算子主要提供将算子流中处理过的数据写出到文件或者数据库等，文件写出支持多重格式，写出到文件可在文件浏览等界面下载到本地。数据写出类算子的具体功能如下表所示：

![数据写出](https://i-blog.csdnimg.cn/direct/731a2ae302844c1cb7c3582599bbb600.png#pic_center)

	文件输出（FileWriterNode）	支持将结果数据写入到HDFS和本地文件系统。
	数据库输出(JDBCOutputNode)	支持将结果数据写入到JDBC兼容的关系数据库。
	Hive数据库输出算子(HiveWriterNode)	支持以YARN模式启动的时候，将数据写入到集群上的Hive数据库。
	kafka输出算子(KafkaNode)	支持以kafka的方式输出dataset，丰富了算子平台数据的输出方式。
	Hbase输出算子（HbaseOutputNode）	支持写入数据到Hbase数据库。
	Excel输出算子(ExcelOutputNode)	支持将结果数据写入到HDFS和本地文件系统的Excel文件中。
	XML输出算子(XMLOutputNode)	支持将结果数据写入到HDFS和本地文件系统的XML文件中。
	ES数据库输出算子(ESOutputNode)	支持数据写入到ElasticSearch数据库。

（3）数据处理类算子包括行处理、列处理、数据融合、数据统计、特征选择和扩展编程等6个类算子。数据处理算子主要是根据用户或机器学习建模需要对数据进行处理。数据处理类算子的具体功能如下表所示：

![行处理](https://i-blog.csdnimg.cn/direct/f8f6ae6b3bc54c27b04a67fbc4b0b6b5.png#pic_center)

	填充(FillNode)	对选中的列用相应填充值进行填充。
	排序(SortNode)	数据排序。
	过滤(FilterNode)	数据过滤。
	聚合(AggregateNode)	数据聚合。
	样本(SampleNode)	数据抽样。
	时序数据重采样(TimeSeriesResampleNode)	对时序数据根据设置参数进行重新采样。
	时序数据清洗(TimeSeriesCleaningNode)	对时序数据根据设置参数进行清理补充。
	存储重分区(RePartitionNode)	根据选择的列队数据存储重新分区。
	去重(DistinctNode)	去除数据中用户定义的重复行。
	异常值缺失值填充(OutlierAndMissingValProNode)	数据异常值判定处理和缺失值填充策略。
	样本分区(PartitionNode)	将数据随机分为训练集，测试集，验证集(可以没有)。
	差分(DiffNode)	对数据进行行处理的算子，可以对指定列进行逐行求差。
	数据平衡(DatasetBalanceNode)	按照用户指定的条件来选择数据集，并根据用户指定的系数来对选中数据集进行调整。
	Shuffle(ShuffleNode)	对数据集进行按行打乱排序的处理。
	分层抽样(SampleStartifiedNode)	主要是为了数据采样使用的，可以保证在不同类别下分配更加均衡。
	生成行(GenerateRowNode)	可以生成多行数据。
	
![列处理](https://i-blog.csdnimg.cn/direct/daac3458c2204f4880dc9acc4ffb5374.png#pic_center)


	生成列(GenerateColumnNode)	根据参数设置生成一列新数据。
	数组载入(Array2DatasetNode)	矩阵数据降维处理，配合cplex算子使用。
	选择(SelectNode)	对数据按照表达式设置进行列选择处理。
	类型(TypeNode)	数据类型查看及测试类型设置。
	格式(FormatNode)	数据存储类型查看与修改。
	多项式特征构造(PolynomialExpressionNode)	将特征展开到多元空间的处理过程， 运用于特征值进行一些多项式的转化。
	删除和重命名(DeleteRenameNode)	数据列删除及列名重命名。
	行列转置(TransposeNode)	对数据表进行行列转置，即将行转换为列，将列转换为行。
	列调整(ColumnAdjustNode)	可以调整列的顺序。
	Excel函数计算(ExcelFunctionCalcuateNode)	根据Excel公式按行计算结果。
![数据融合](https://i-blog.csdnimg.cn/direct/369331fc988f44ddbcff9121c2dc5d6e.png#pic_center)

	合并(MergeNode)	数据合并。
	追加(UnionNode)	数据追加。

![统计分析](https://i-blog.csdnimg.cn/direct/916ec96d808345bdaf3b883e7e362341.png#pic_center)

	斯皮尔曼相关性系(SpearmanCorrelationNode)	实现斯皮尔曼相关性系数算法。
	皮尔森相关性系(PearsonCorrelationNode)	实现皮尔森相关性系数算法。
	描述(DescribeNode)	将流入的数据集按照列进行归纳统计，并根据参数计算出异常值数量和众数以及极值数量。
	卡方检验(ChiSquareNode)	统计样本的实际观测值与理论推断值之间的偏离程度。
	LB检验(LBTestNode)	判断一个时间序列是否为纯随机序列。
	ADF检验(ADFNode)	判断序列是否存在单位根：如果序列平稳，就不存在单位根；否则，就会存在单位根。
	ACF自相关性分析(ACFNode)	度量时间序列中每隔k个时间单位(yt和yt–k)的观测值之间的相关性。
	PACF自相关性分析(PACFNode)	描述观测值与其滞后(lag)之间的直接关系。
	
![特征工程](https://i-blog.csdnimg.cn/direct/e255e0822da84b509a938adac5a45a30.png#pic_center)


	流式分位(QuantileDiscretizerNode)	实现分位数离散化，将一列连续的数据列转成分类型数据。
	流式标准化(StandardScalerNode)	将数据进行标准化的算子。
	PCA(BuildPCANode)	实现数据的降维。通过对数据样本的分析，自动将高纬度数据降低到合适维度。
	流式归一化(RescaleNode)	实现归一化算法的算子。
	卡方检验特征选择(ChiSquareSelectorNode)	对于Label为离散类型，通过卡方检验，从离散类型的features里选取有效特征。
	归一化(BuildRescaleNode)	是流式归一化算子从dataset in 、dataset out模式变为类似建模算子，会生成一个模型，这个模型保留了建模时对数据的统计值，以便在预测时能对预测数据做同样的处理。
	标准化(BuildStandardScaler)	是流式标准化算子从dataset in 、dataset out模式变为类似建模算子，会生成一个模型，这个模型保留了建模时对数据的统计值，以便在预测时能对预测数据做同样的处理。
	分位(BuildQdNode)	是流式分位算子从dataset in 、dataset out模式变为类似建模算子，会生成一个模型，这个模型保留了建模时对数据的统计值，以便在预测时能对预测数据做同样的处理。
	二分(BinarizerNode)	是对K-means的改进，防止聚类陷入局部最优解。
	分箱(BucketizerNode)	是基于二分法进行分类的算子，可以将数据进行离散化处理。
	特征重要性(FeatureImportanceNode)	描述数据中其他特征相对label列的重要性。
	TSNE(TSNENode)	用于高维数据降维。
	RobustScaler(BulidRobustScalerNode)	使用具有鲁棒性的统计量缩放带有异常值(离群值)的数据，根据分位数范围(默认为IQR)删除中位数并缩放数据。
	IV(Information Value)	对分类模型中用于分类的feature列进行信息值计算。
	SMOTE	针对数据不平衡的一种过采样方法，通过对少数类样本进行分析并根据少数类样本人工合成新样本添加到原数据集中。
	SVD	分解讲矩阵分成奇异值和奇异值向量。
	ICA(BuildICANode)	实现从观测数据中还原出独立成分的一种降维算法。
	IG(IGFeatureSelectNode)	IG算子通过计算Feature和Label之间的互信息，作为Feature的一个重要性评价指标。用户基于这个值可以做特征选择。
	FisherScore(FisherScoreNode)	FisherScore算子通过计算Feature和Label之间的评分，作为Feature的一个重要性评价指标。用户基于这个值可以做特征选择。
	RFE(BuildRecursiveFeatureEliminationNode)	RFE算子通过多次迭代，排除不重要的特征，达到特征选择的目的。
	流式离散特征编码(CategoricalFeatureEncodingNode)	将离散型变量转化为连续型变量的算子。
	离散特征编码(BuildCategoricalFeatureEncodingNode)	将离散型变量转化为连续型变量的算子。
	流式目标编码(TargetEncodingNode)	用来对离散型特征变量做编码使用，是用目标变量的均值替换分类值的过程。
	目标编码(BuildTargetEncodingNode)	用来对离散型特征变量做编码使用，是用目标变量的均值替换分类值的过程。
	流式独热编码(OneHotEncodingNode)	使用N位状态寄存器来对N个状态进行编码，每个状态都由他独立的寄存器位，并且在任意时候，其中只有一位有效。
	独热编码(BuildOneHotEncodingNode)	使用N位状态寄存器来对N个状态进行编码，每个状态都由他独立的寄存器位，并且在任意时候，其中只有一位有效。
	流式计数编码(StreamingCountEncodingNode)	用来对离散型做计数编码，是用出现次数替换分类值的过程。
	计数编码(BuildCountEncodingNode)	用来对离散型做计数编码，是用出现次数替换分类值的过程。
	流式BaseN编码(StreamingBaseNEncodingNode)	用来对离散型特征变量做编码使用，将顺序编码为N的底数的过程。
	BaseN编码(BuildBaseNEncodingNode)	用来对离散型特征变量做编码使用，将顺序编码为N的底数的过程。
	流式Hash编码(StreamingHashEncodingNode)	用来对离散型特征变量做编码使用，将离散值使用hash算法编码。
	Hash编码(BuildHashEncodingNode)	用来对离散型特征变量做编码使用，将离散值使用hash算法编码。
	流式证据权重编码(StreamingWOEEncodingNode)	用来对离散型特征变量做编码使用。
	证据权重编码(BuildWOEEncodingNode)	用来对离散型特征变量做编码使用。

（4）机器学习类算子提供主流的机器学习算法，用户通过对数据进行建模来挖掘数据
更深层次的规律和知识。并利用模型进行预测来改进自己的商务决策。机器学习类算子包括分类，回归，聚类等算法。同时还提供了模型评估算子，方便用户评估生成模型是否满足要求。机器学习类算子的具体功能如下表所示：

![机器学习分类](https://i-blog.csdnimg.cn/direct/a6d7435f988e40c9bd63656d58dbca38.png#pic_center)

	逻辑回归分类(BuildLGRNode)	线性回归基础上指定激活函数实现数据的分类算法。
	决策树分类(BuildDTCNode)	简单易用的非参数分类器模型，它会在用户选定的特征列上不断进行分裂，使得在每一分支对目标变量纯度组建增高，达到数据的分类。
	梯度提升决策树分类(BuildGBTCNode)	梯度提升树是一个Boosting聚合模型，它是由多个决策树一起组合和来预测。多个决策树之间是顺序组合关系，每一个决策树模型都会修正之前所有模型预测的误差，实现分类。
	XGBoost分类(BuildXGBoostClassifier)	优化的分布式梯度提升增强分类算法，提供了一种并行树增强。
	随机森林分类(BuildRFCNode)	内部集成了大量的决策树模型。每个模型都会选取一部分特征和一部分训练样本。最终有多个决策树模型来共同决定预测值，实现数据的分类。
	朴素贝叶斯分类(BulidNBNode)	通过应用贝叶斯定理计算给定观测值的每个标签的条件概率分布来进行预测。
	支持向量机分类(BuildSVM)	通过在高维空间中构造超平面或者超平面集合实现数据的分类。
	多层感知机分类(BuildMLPNode)	实现人工神经网络实现数据的分类。
	LightGBM分类(BuildLightGBMClassifierNode)	属于Boosting集合模型中的一种，它和XGBoost一样是对GBDT的高效实现。LightGBM在很多方面会比XGBoost表现更为优秀。它有以下优势：更快的训练效率、低内存使用、更高的准确率、支持并行化学习、可处理大规模数据。
	因子分解机分类(BuildFMClassifierNode)	一种基于矩阵分解的机器学习算法，可以解决特征组合以及高维稀疏矩阵问题的强大的机器学习算法，首先是特征组合，通过对两两特征组合，引入交叉项特征，提高模型得分；其次是高维灾难，通过引入隐向量(对参数矩阵进行矩阵分解)，完成对特征的参数估计。目前FM算法是推荐领域被验证的效果较好的推荐方案之一。
	AdaBoost分类(BuildAdaboostClassifierNode)	一种Boosting集成方法，主要思想就是将弱的学习器提升(boost)为强学习器，根据上轮迭代得到的学习器对训练集的预测表现情况调整训练集中的样本权重, 然后据此训练一个新的基学习器，最终的集成结果是多个基学习器的组合。
	KNN分类(BuildKNNClassifierNode)	要预测一个实例，需要求出与所有实例之间的距离。即目标样本离哪个分类的样本更接近。

![机器学习回归](https://i-blog.csdnimg.cn/direct/d01b725acd7e4582a0028031a50f8d6a.png#pic_center)

	线性回归(BuildLRNode)	假设所有特征变量和目标变量之间存在线性关系。通过训练来求得各个特征的权重以及截距。
	决策树回归(BuildDTRNode)	简单易用的非参数分类器模型，它会在用户选定的特征列上不断进行分裂，使得在每一分支对目标变量纯度组建增高，达到数据的回归。
	梯度提升决策树回归(BuildGBTRNode)	梯度提升树是一个Boosting聚合模型，它是由多个决策树一起组合和来预测。多个决策树之间是顺序组合关系，每一个决策树模型都会修正之前所有模型预测的误差，实现回归。
	保序回归(BuildRNode)	是在单调的函数空间内对给定数据进行非参数估计的回归模型。
	XGBoost回归(BuildXGBoostRegression)	优化的分布式梯度提升增强回归算法，提供了一种并行树增强。
	随机森林回归(BuildRFRNode)	内部集成了大量的决策树模型。每个模型都会选取一部分特征和一部分训练样本。最终有多个决策树模型来共同决定预测值，实现数据的回归。
	广义线性回归(BuildGLRNode)	通过联结函数建立响应变量的数学期望值与线性组合的预测变量之间的关系。其特点是不强行改变数据的自然度量，数据可以具有非线性和非恒定方差结构。
	LightGBM回归(BuildLightGBMRegressionNode)	属于Boosting集合模型中的一种，它和XGBoost一样是对GBDT的高效实现。LightGBM在很多方面会比XGBoost表现更为优秀。它有以下优势：更快的训练效率、低内存使用、更高的准确率、支持并行化学习、可处理大规模数据。
	因子分解机回归(BuildFMRegressionNode)	一种基于矩阵分解的机器学习算法，可以解决特征组合以及高维稀疏矩阵问题的强大的机器学习算法，首先是特征组合，通过对两两特征组合，引入交叉项特征，提高模型得分；其次是高维灾难，通过引入隐向量(对参数矩阵进行矩阵分解)，完成对特征的参数估计。目前FM算法是推荐领域被验证的效果较好的推荐方案之一。
	AdaBoost回归(BuildAdaboostRegressionNode)	一种Boosting集成方法，主要思想就是将弱的学习器提升(boost)为强学习器，根据上轮迭代得到的学习器对训练集的预测表现情况调整训练集中的样本权重, 然后据此训练一个新的基学习器，最终的集成结果是多个基学习器的组合。
	KNN回归(BuildKNNRegressionNode)	KNN算子在做回归预测时，一般使用平均值法。
	高斯过程回归(BuildGPRegressionNode)	高斯过程回归算法是使用高斯过程先验对数据进行回归分析的非参数模型。
	多层感知机回归(BuildMLPRegressionNode)	多层感知是一种前馈人工神经网络模型，其将输入的多个数据集映射到单一的输出的数据集上，多层感知机层与层之间是全连接的，最底层是输入层，中间是隐藏层，最后是输出层。

![机器学习关联规则](https://i-blog.csdnimg.cn/direct/cb8053fc6a7f49909ed2ee8b5435d6bb.png#pic_center)


	频繁模式增长(BuildFPGromthNode)	频繁模式增长通过构造频繁模式树的方式，可以通过较少的对数据集的遍历来构造频繁项集或频繁项对。同时支持用户自定义最小置信度和最小支持级别。
	PrefixSpan(PrefixSpanNode)	是挖掘出满足最小支持度的频繁序列

![机器学习时间序列](https://i-blog.csdnimg.cn/direct/57c22d9be81b4a698c790fff90df3edf.png#pic_center)


	ARIMAX(ARIMAXNode)	带回归项的ARIMA模型，又称为扩展的ARIMA模型。回归项的引入有助于提高模型的预测效果，引入的回归项通常是和被解释变量相关程度高的变量。
	ARIMA(ARIMANode)	常用时间序列模型中的一种，如果只是根据单一目标变量的历史数据预测未来数据，可以使用ARIMA算法。如果除了目标变量还有其他输入变量可以选择ARIMAX模型。
	HoltWinters(HoltWintersNode)	常用时间序列模型中的一种，如果只是根据单一目标变量且有明显的周期性的历史数据预测未来数据，可以使用HoltWinters 算法。
	一次指数平滑预测(SESNode)	一次指数平滑预测(single exponential smoothing)，也称为单一指数平滑法，当时间数列无明显的趋势变化，可用一次指数平滑预测。
	二次指数平滑预测(HoltLinearNode)	二次指数平滑预测，二次指数平滑是对一次指数平滑的再平滑。它适用于具线性趋势的时间数列。

![机器学习生存分析](https://i-blog.csdnimg.cn/direct/4ea7e0a0332e4ef88c76dadca009c867.png#pic_center)

	加速失效时间回归(BuildAFTSRNode)	检查数据的参数生存回归模型，描述了生存时间对数的模型。

![机器学习聚类](https://i-blog.csdnimg.cn/direct/502c0c87a99741f78280823f37293cc9.png#pic_center)


	KMeans聚类(BulidKMeansNode)	迭代求解的聚类分析算法，其步骤是随机选取K个对象作为初始的聚类中心，然后计算每个对象与各个种子聚类中心之间的距离，把每个对象分配给距离它最近的聚类中心。
	二分KMeans聚类(BuildBKMeansnode)	是对K-means的改进，防止聚类陷入局部最优解。它的主要思想是：首先将所有点作为一个簇，然后将该簇一分为二。之后选择能最大限度降低聚类代价函数的簇划分为两个簇。以此进行下去，直到簇的数目等于用户给定的数目k为止。
	高斯混合模型聚类(BuildGMNode)	高斯混合模型就是用高斯概率密度函数精确地量化事物，它是一个将事物分解为若干的基于高斯概率密度函数形成的模型。
	模糊C均值聚类(BuildFCMeansNode)	基于划分的聚类算法，使得被划分到同一簇的对象之间相似度最大，而不同簇之间的相似度最小。
	Canopy聚类(BuildCanopyClusterNode)	一种快速粗聚类算法，优势是用户不用事先指定聚类数目。用户需要指定两个距离阈值，T1，T2，且T1>T2。可以认为T2为核心聚类范围，T1为外围聚类范围。
	CanopyKmeans聚类(BuildCanopyKMeansNode)	结合Canopy和Kmeans两种聚类算法的优势，首先利用Canopy聚类先对数据进行快速“粗”聚类，得到k值后再使用K-means进行进一步“细”聚类。
	文档主题生成模型聚类(BuildLDANode)	也称为一个三层贝叶斯概率模型，包含词、主题和文档三层结构。所谓生成模型，就是说，我们认为一篇文章的每个词都是通过“以一定概率选择了某个主题，并从这个主题中以一定概率选择某个词语”这样一个过程得到。文档到主题服从多项式分布，主题到词服从多项式分布。
	DBSCAN聚类 (Density-Based Spatial Clustering of Applications with Noise)	一个比较有代表性的基于密度的聚类算法。它将簇定义为密度相连的点的最大集合，能够把具有足够高密度的区域划分为簇，并可在噪声的空间数据库中发现任意形状的聚类。

![机器学习异常检测](https://i-blog.csdnimg.cn/direct/d035a7bb1c104d44b7a44235cdddb78a.png#pic_center)

	异常检测(BuildIFnode)	检测数据中的异常数据。

![机器学习推荐](https://i-blog.csdnimg.cn/direct/a29e1cec53b44389a996d16598733b18.png#pic_center)


	协同过滤(BuildALSNode)	推荐系统中常用的一种方法。该算法旨在填补用户-产品关联矩阵中缺少的项。在算法中，用户和产品都是通过一组少量的潜在因素描述，这些潜在因素可以用于预测用户-产品关联矩阵中缺少的项。

![机器学习模型评估](https://i-blog.csdnimg.cn/direct/b1c0841323814523abcf5814d57aa712.png#pic_center)

	评估(EvaluationNode)	用于评估用当前数据训练出来的模型的正确性，显示对模型各个评价指标的具体值。
	混淆矩阵(ConfusionMatrixNode)	用于展示分类算子分类结果的混淆矩阵，方便用户对分类结果进行评估。
	ROC-AUC评估(ROCAUCNode)	ROC-AUC算子(ROCAUCNode) 用在分类模型后，用于评估当前数据训练出来的分类模型的正确性，显示分类结果的ROC曲线和AUC值，方便用户对模型的分类效果进行评估。
	时间序列模型评估(TimeSeriesModeEvaluateNode)	通过时间序列模型评估算子对经过时间序列预测后的数据集进行指标评估。


（5）图表分析类算子主要是通过柱状图、饼状图及表格等图标将数据可视化。图表分析类算子的具体功能如下表所示：
![机器学习图表分析](https://i-blog.csdnimg.cn/direct/dc6f1934915b4d478229c7c09eadd672.png#pic_center)

	图表算子	散点图(ScattergramNode)	将数据转化成散点图的方式展示。
	柱状图(BarNode)	将数据转化成柱状图的方式展示。
	折线图(LineNode)	将数据转化成折线图的方式展示。
	饼状图(PieNode)	将数据转化成饼装图的方式展示。
	直方图(HistogramNode)	将数据以直方图的形式展示，显示数据的分布情况，并可以在图中显示数据的正态分布曲线。
	二维气泡图(BubbleNode)	将数据转化成二维气泡图的方式展示。
	平行关系图(ParallelgramNode)	将数据转化成平行关系图的方式展示。
	二维雷达图(RadarNode)	将数据转化成二维雷达图的方式展示。
	二维堆积图(StackBarNode)	将数据转化成二维堆积图的方式展示。
	二维线箱图(BoxplotNode)	将数据转化成二维线箱图的方式展示。
	三维散点图(Scattergram3DNode)	将数据转化成三维散点图的方式展示。
	三维气泡图(Bubble3DNode)	将数据转化成三维气泡图的方式展示。
	三维曲面图(SurfaceNode)	将数据转化成三维曲面图的方式展示。
	区域色块图(MapColorBlockNode)	将地图上的不同区域用不同的颜色显示出来，显示不同地理区域的数据分布情况。
	时序图(SequenceDiagramNode)	将时间序列数据或通过算子平台中“时序预测算子”预测得到时间序列结果用点或线的方式展示，以便于对时间数据进行观察分析。
	地图散点图(MapBubbleNode)	支持基于百度地图为底图的数据可视化算子，将数据转化成散点图的形式在地图上展示。
	地图柱状图(MapBarNode)	支持基于百度地图为底图的数据可视化算子，将数据转化成柱状图的形式在地图上展示。
	地图饼状图(MapPieNode)	支持基于百度地图为底图的数据可视化算子，将数据转化成饼状图的形式在地图上展示。
	地图热力图(MapHeatMapNode)	支持基于百度地图为底图的数据可视化算子，将数据用热力图的形式展示，显示数据的地域分布情况。
	表格(TableOutputNode)	输出数据展示。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/676bf1cd621d42948ff547e295468cf4.png#pic_center)

	扩展编程	Sql(SQLNode)	支持Sql语句查询的算子，目前只支持select操作。
	PySpark(PySparkNode)	提供编写PySpark语句处理数据的能力，使客户可以通过编程的方式快速完成一系列数据操作。

Sentosa数据科学与机器学习平台（Sentosa_DSML）是力维智联完全自主知识产权的一站式人工智能开发部署应用平台，可同时支持零代码“拖拉拽”与notebook交互式开发，旨在通过低代码方式帮助客户实现AI算法模型的开发、评估与部署，结合完善的数据资产化管理模式与开箱即用的简捷部署支持，可赋能企业、城市、高校、科研院所等不同客户群体，实现AI普惠、化繁为简。
Sentosa_DSML产品由1+3个平台组成，以数据魔方平台（Sentosa_DC）为主管理平台，三大功能平台包括机器学习平台（Sentosa_ML）、深度学习平台（Sentosa_DL）和知识图谱平台（Sentosa_KG）。力维智联凭借本产品入选“全国首批人工智能5A等级企业”，并牵头科技部2030AI项目的重要课题，同时服务于国内多家“双一流”高校及研究院所。
为了回馈社会，矢志推动全民AI普惠的实现，不遗余力地降低AI实践的门槛，让AI的福祉惠及每一个人，共创智慧未来。为广大师生学者、科研工作者及开发者提供学习、交流及实践机器学习技术，我们推出了一款轻量化安装且完全免费的Sentosa_DSML社区版软件，该软件包含了Sentosa数据科学与机器学习平台（Sentosa_DSML）中机器学习平台（Sentosa_ML）的大部分功能，以轻量化一键安装、永久免费使用、视频教学服务和社区论坛交流为主要特点，同样支持“拖拉拽”开发，旨在通过零代码方式帮助客户解决学习、生产和生活中的实际痛点问题。
该软件为基于人工智能的数据分析工具，该工具可以进行数理统计与分析、数据处理与清洗、机器学习建模与预测、可视化图表绘制等功能。为各行各业赋能和数字化转型，应用范围非常广泛，例如以下应用领域：
金融风控：用于信用评分、欺诈检测、风险预警等，降低投资风险；
股票分析：预测股票价格走势，提供投资决策支持；
医疗诊断：辅助医生进行疾病诊断，如癌症检测、疾病预测等；
药物研发：进行分子结构的分析和药物效果预测，帮助加速药物研发过程；
质量控制：检测产品缺陷，提高产品质量；
故障预测：预测设备故障，减少停机时间；
设备维护：通过分析机器的传感器数据，检测设备的异常行为；
环境保护：用于气象预测、大气污染监测、农作物病虫害防止等；
客户服务：通过智能分析用户行为数据，实现个性化客户服务，提升用户体验；
销售分析：基于历史数据分析销量和价格，提供辅助决策；
能源预测：预测电力、天然气等能源的消耗情况，帮助优化能源分配和使用；
智能制造：优化生产流程、预测性维护、智能质量控制等手段，提高生产效率。

欢迎访问Sentosa_DSML社区版的官网https://sentosa.znv.com/，免费下载体验。同时，我们在B站、CSDN、知乎、博客园等平台有技术讨论博客和应用案例分享，欢迎广大数据分析爱好者前往交流讨论。
Sentosa_DSML社区版，重塑数据分析新纪元，以可视化拖拽方式指尖轻触解锁数据深层价值，让数据挖掘与分析跃升至艺术境界，释放思维潜能，专注洞察未来。
社区版官网下载地址：https://sentosa.znv.com/
社区版官方论坛地址：http://sentosaml.znv.com/
B站地址：https://space.bilibili.com/3546633820179281
CSDN地址：https://blog.csdn.net/qq_45586013?spm=1000.2115.3001.5343
GitHub地址：https://github.com/Kennethyen/Sentosa_DSML
知乎地址：https://www.zhihu.com/people/kennethfeng-che/posts
博客园地址：https://www.cnblogs.com/KennethYuen
![Sentosa_DSML社区版官网](https://i-blog.csdnimg.cn/direct/93c9871e1bbb43b9a2f8f99648354df9.png#pic_center)

[video(video-Ii7ACWb6-1725949695276)(type-csdn)(url-https://live.csdn.net/v/embed/423332)(image-https://img-home.csdnimg.cn/images/20230724024159.png?be=1&origin_url=https://v-blog.csdnimg.cn/asset/2d9c7938e2725ef1c392db3bf80e6148/cover/Cover0.jpg)(title-Sentosa_DSML算子流开发视频)]







