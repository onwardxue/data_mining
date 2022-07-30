# -*- coding:utf-8 -*-
# @Time : 2022/7/29 8:17 下午
# @Author : Bin Bin Xue
# @File : chart7_分类
# @Project : data_mining

'''
第七章 分类
    7.1 分类概述
            数据分类也称为'监督学习'，包括学习阶段(构建和训练模型)和分类阶段（根据数据给出类标签）

    7.2 决策树归纳
            决策树集成算法目前是最重要的算法之一
            (1)决策树原理
            (2)ID3算法(信息增益选择属性)
                信息增益 - 属性对分类结果的增益
                算法：计算每个特征的信息增益，取最大的特征将数据集分组，依次迭代
                缺点：...
            (3)C4.5算法(增益率选择属性)
                增益率
                优缺点：改进了ID3..
            (4)CART算法(基尼指数选择属性)
                分类回归树-基尼指数：改进C4.5，数据越复杂效果越好
            (5)树剪枝
                目的：防止过拟合；防止受异常值干扰
                策略：预剪枝，后剪枝。后剪枝效果更好，但时间更长
            (6)决策树应用
                sklearn.tree.DecisionTreeClassifier()
                重要参数：criterion - 属性分类方法，可选'gini'或'entropy'
                                  max_depth - 决策树最大深度，防止过拟合
                                  min_samples_leaf - 叶节点包含的最小样本数

    7.3 K近邻算法
        K近邻通过测量不同特征值之间的距离进行分类
        (1)算法原理：找到离待分类样本最近的K个样本，其中最多的标签作为该样本的标签
        (2)KNN应用
                sklearn.neighbors.KNeighborsClassifier
                重要参数：n_neighbors - 近邻个数
                                  weights - 权重

    7.4 支持向量机
        (1)算法原理：使用一种非线性映射，把原始训练数据映射到高维，在
                                新的维上搜索最佳分离超平面
        (2)核函数：线性核、多项式核、高斯核、拉普拉斯核、Sigmoid核
        (3)SVM应用：
                sklearn.svm.SVC
                重要参数：kernal - 核函数
                                  gamma
                                  decision_function_shape
                                  C

    7.5 朴素贝叶斯
        (1)朴素贝叶斯应用：
            sklearn.naive_bayes import GaussianNB

    7.6 模型评估与选择
            分类器性能度量：
                泛化能力：学习方法对未知数据的预测能力（测试误差）
            1_混淆矩阵
                    TP FN
                    FP TN
                TP和TN的所占比例越大，FP、FN越接近0越好
            2_分类器常用度量指标
                1)Accuracy准确率（反应各类元组的正确识别情况，但对于类不平衡数据要配合TPR一起评价）
                2)TPR敏感度（正例占所有正例的比例）
                3)Precision精度、Recall召回率（精度为正例中判正的比例,Recall=TPR）
                4)F度量和FB度量（F为精度和召回的调和平均，FB为加权度量）
                5)P-R曲线（横轴为召回率，纵轴为精确度）
                6)ROC曲线和AUC值（AUC值越大，准确性越高，最常用）
                其他指标：
                    1)速度：计算开销
                    2)鲁棒性：对含有噪声和缺失值的数据，做出正确判断的能力
                    3)可伸缩性：不同的数据规模
                    4)可解释性

            模型选择：
                防止模型过拟合或欠拟合
                主要方法：
                1)正则化
                    增加一个增则化项，控制模型过复杂
                    奥卡姆剃刀原理：在可能的模型中选择尽可能简单的模型
                2)交叉验证
                    简单交叉验证：随机划分数据集为训练测试集(7:3)，选出测试误差最小的
                    k-折交叉验证：随机数据集为k个，其中k-1个用于训练，剩下1个测试，取平均最小误差
                    留一交叉验证：k=N时，数据留匮乏时使用

    7.7 组合分类
            1_组合分类方法简介
                采用集成方法将一系列基分类器组进行集成，得到集成模型。
                集成方法包括Bagging、Boosting和随机森林
            2_Bagging(权重相同，取票数最多)
                方法描述：从含有d个元组的数据集D中有放回的提取d个元组，
                        生成数据集D1（D1中有重复数据），用D1数据训练得到
                        一个基分类器。反复执行k次，得到由k个基分类器组合成
                        的组合分类器。用组合分类器进行预测，结果取票数最多的
                实现：
                        sklearn.ensemble import BaggingClassifier
            3_Boosting和AdaBoost（元组权重不同，分类器权重为其准确率；加权表决结果）
                    方法描述：每个训练元组赋予一个权重，根据第Mi个分类器的预
                        测结果更新这个权重（目的是使其后的分类器更关注误分类的
                        训练元组。最终的分类器组合每个分类器的表决，各基分类器
                        的投票权重则是其准确率）
                    AdaBoost(Adaptive Boosting)：
                        描述：给定包含d个具有类标号的数据集。起始时，AdaBoost对
                        每个训练元组赋予相等的权重1/d，执行k轮，产生k个基分类器。
                        在第i轮，从D中元组有放回抽样，形成大小为d的训练集Di，每
                        个元组被选中的概率由其权重决定。从训练集Di，得到分类器Mi，
                        然后再用Di验证Mi的误差，用误差更新训练元组的权重。
                        （元组分类错误，权重增加）
                    实现：
                        sklearn.ensemble import AdaBoostClassifier

            4_随机森林
                方法描述：随机森林由多棵决策树组成。(1)训练总样本数为N，单
                    棵决策树从N个训练集中随机有放回抽取N个作为单颗树的训练样本。
                    (2)每棵树根据特征重要性独自向下分裂，直到该节点的所有训练样
                    例都属于同一类。(3)每棵树投票并返回得票最多的类。(4)还可以得
                    出特征重要性。
                随机森林两种形式：Forest-RI、Forest-RC
                随机森林不足：训练数据中存在噪声时，随机森林容易出现过拟合现象
                随机森林实现：
                    sklearn.tree.DecisionTreeClassifier

'''
print('利用决策树算法对Iris数据集构建决策树')
from sklearn.datasets import load_iris
import pandas as pd
from sklearn import tree
from sklearn.tree import export_graphviz
import graphviz

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data,iris.target)
dot_file = 'tree.dot'
tree.export_graphviz(clf,out_file=dot_file)
with open('tree.dot','w') as f:
    f = export_graphviz(clf,out_file=f,feature_names=['SL','SW','PL','PW'])
