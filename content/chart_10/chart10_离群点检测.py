# -*- coding:utf-8 -*-
# @Time : 2022/7/29 8:20 下午
# @Author : Bin Bin Xue
# @File : chart10_离群点检测
# @Project : data_mining

'''
第10章 离群点检测
    10.1 离群点概述
      离群点(异常值)：显著偏离一般水平的观测对象（数据）
      离群点检测：找出离群点
      离群点与噪声：离群点可能由噪声产生，也可能由真实数据产生。
      （偶尔的高消费是噪声，却不是离群点）
      离群点类型：
        (1)全局离群点
            一个数据对象偏离了数据集中绝大多数对象
        (2)条件离群点
            某种特定条件下，产生的离群点
        (3)集体离群点
            数据集中的一些对象显著偏离整个数据集时
            （但集体离群点中的个体不一定是离群点）
      离群点检测挑战：
        （1）正常和离群点边界的不清晰
        （2）相似度/距离度量选择多样
        （3）可能错误地将噪声识别为离群点
        （4）可解释性

    10.2 离群点检测
      四种：
        （1）基于统计学
        （2）基于邻近性
        （3）基于聚类
        （4）基于分类

    10.3 sklearn中的异常检测方法
    （1）奇异点(Novelty)检测：训练数据中无离群点，目标是用训练好的模型去检测新样本里
        （OneClassSVM）
    （2）异常点(Outlier)检测：训练数据中存在离群点，模型训练用中心样本，忽视其他异常点
        （Isolation Forest、LOF）
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

# OneClassSvm是奇异点检测方法
def OneClassSvmTest():
    xx,yy = np.meshgrid(np.linspace(-5,5,500),np.linspace(-5,5,500))
    # 生成训练数据
    X = 0.3*np.random.randn(100,2)
    X_train = np.r_[X+2,X-2]
    # 生成一些有规律的奇异点
    X = 0.3*np.random.randn(20,2)
    X_test = np.r_[X+2, X-2]
    # 生成一些异常的奇异点
    X_Outliers = np.random.uniform(low=-4,high=4,size=(20,2))
    # 训练模型
    clf = svm.OneClassSVM(nu=0.1,kernel='rbf',gamma=0.1)
    clf.fit(X_train)
    # 异常点检测
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    y_pred_outliers = clf.predict(X_Outliers)
    n_error_train = y_pred_train[y_pred_train==-1].size
    n_error_test = y_pred_test[y_pred_test==-1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers==-1].size
    # 绘制图形
    z = clf.decision_function(np.c_[xx.ravel(),yy.ravel()])
    z = z.reshape(xx.shape)
    plt.title('Novelty Detection')
    plt.contourf(xx,yy,z,levels=np.linspace(z.min(),0.7),cmap=plt.cm.PuBu)
    a = plt.contour(xx,yy,z,levels=[0],linewidths=2,colors='darkred')
    plt.contour(xx,yy,z,levels=[0,z.max()],colors='palevioletred')
    s = 40
    # 绘制点
    b1 = plt.scatter(X_train[:,0],X_train[:,1],c='white',s=s)
    b2 = plt.scatter(X_test[:,0],X_test[:,1],c='blueviolet',s=s)
    c = plt.scatter(X_Outliers[:,0],X_Outliers[:,1],c='gold',s=s)
    # 设置范围、图例、坐标标签
    plt.axis('tight')
    plt.xlim((-5,5))
    plt.ylim((-5,5))
    plt.legend([a.collections[0],b1,b2,c],
               ['learned frontier','training observations',
                'new regular observations','new abnormal observations'],
                loc='upper left',
                prop=matplotlib.font_manager.FontProperties(size=11))
    plt.xlabel(
        'error train:%d/200;errors novel regular:%d/40;'
        'error novel abnormal:%d/40'
        % (n_error_train,n_error_test,n_error_outliers)
    )
    plt.show()

# EllipticEnvelopeTest是对高斯分布数据集的离群点检验方法，该方法在高维度下表现欠佳
from sklearn.covariance import EllipticEnvelope
def EllipticEnvelopeTest():
    xx,yy = np.meshgrid(np.linspace(-5,-5,500),np.linspace(-5,-5,500))
    # 生成训练数据（100行2列，正态分布）
    X = 0.3 * np.random.randn(100,2)
    #np.r_按列堆叠（上下）
    X_train = np.r_[X+2,X-2]
    # 生成用于测试的数据
    X = 0.3 * np.random.randn(10,2)
    X_test = np.r_[X+2, X-2]
    # 模型拟合
    clf = EllipticEnvelope()
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    print('X_train:\n',X_train)
    print('X_test:\n',X_test)
    print('novelty detection result:\n',y_pred_train)
    print('novelty detection result:\n',y_pred_test)


def main():
    # OneClassSvmTest()
    EllipticEnvelopeTest()

if __name__ == '__main__':
    main()