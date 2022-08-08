# -*- coding:utf-8 -*-
# @Time : 2022/7/29 8:15 下午
# @Author : Bin Bin Xue
# @File : chart5_回归分析
# @Project : data_mining

'''
第5章
    5.1 回归分析概述
        通过对数据进行分析实现数值预测（预测新的X对应的y值）
        - 线性回归只能用于存在线性回归的数据中

    5.2 一元线性回归（单变量）
        模型：y = b0 + b1x + ei
        参数估计方法包括：最小二乘法（最常用）、矩方法、极大似然法
        实现：
            sklearn.linear_model.LinearRegression

    5.3 多元线性回归（多变量）
        模型：y = b0 + b1x1 + b2x2 + ... + u
        参数估计方法：最小二乘法
        实现：
            sklearn.linear_model.LinearRegression

    5.4 逻辑回归（线性回归的结果通过sigmoid函数转为非线性）
        模型：z = b0 + b1x1 + ..
            g(z) = 1 / (1 + e^(-z))
        实现：
            sklearn.linear_model.LogisticRegression

    5.5 其他回归分析
        (1)多项式回归
            特点：对于线性回归的改进，不再只是一次幂，适用于非线性关系
            实现：
                sklearn.linear_model.LinearRegression(fit(几次幂))

        (2)岭回归
            特点：适用于过拟合严重或各变量之间存在多重共线性的情况（或特征数量比样本多的情况）
                 改进了最小二乘法的无偏性，能判断特征重要性
            实现：
                sklearn.linear_model.Ridge,RidgeCV

        (3)Lasso回归
            特点：能将一些不重要的回归系数缩减为0，达到剔除变量的目的
            实现：
                sklearn.linear_model.Lasso

        (4)逐步回归
            目的是使用最少的预测变量最大化预测能力

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression

def LinearRegressionTest():
    print('一元线性回归')
    iris = load_iris()
    data = pd.DataFrame(iris.data)
    data.columns = ['sepal-length','sepal-width','petal-length','petal-width']
    data.head()
    # 一元线性回归分析
    x = data['petal-length'].values
    y = data['petal-width'].values
    x = x.reshape(len(x),1)
    y = y.reshape(len(y),1)
    clf = LinearRegression()
    clf.fit(x,y)
    pre = clf.predict(x)
    plt.scatter(x,y,s=50)
    plt.plot(x,pre,'r-',linewidth=2)
    plt.xlabel('petal-length')
    plt.ylabel('petal-width')
    for idx, m in enumerate(x):
        plt.plot([m,m],[y[idx],pre[idx]],'g-')
    plt.show()
    # 显示回归线的参数
    print(u'系数',clf.coef_ )
    print(u'截距',clf.intercept_)
    print(np.mean(y-pre)**2)
    # 预测'花萼宽度'
    print(clf.predict([[3,9]]))

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def MultiRegressionTest():
    print('多元线性回归')
    d = datasets.load_boston()
    data = pd.DataFrame(d.data)
    data['price'] = d.target
    data.sample(5)
    # 进行多元线性回归分析
    simple2 = LinearRegression()
    x_train,x_test,y_train,y_test = train_test_split(d.tata,d.target,random_state=666)
    simple2.fit(x_train, y_train)
    print('多元线性回归模型系数：\n',simple2.coef_)
    print('多元线性回归模型常数项：\n',simple2.intercept_)
    # 预测
    y_predict = simple2.predict(x_test)
    # 模型分析
    print('预测值的均方误差：', mean_squared_error(y_test,y_predict))
    print(r2_score(y_test,y_predict))
    print(simple2.score(x_test,y_test))
    print('各系数间的系数矩阵：\n',simple2.coef_)
    print('影响房价的特征排序：\n',np.argsort(simple2.coef_))
    print('影响房价的特征排序：：\n',d.feature_names[np.argsort(simple2.coef_)])

from sklearn.preprocessing import StandardScaler
def LogisticRegressionTest():
    print('逻辑回归')
    X = load_iris().data
    y = load_iris().target
    print('前8条数据：\n',X[:8])
    print('前8条数据对应的类型：\n',y[:8])
    # 划分训练集，进行归一化
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)
    print(x_train[:5])
    # 逻辑回归预测


def main():
    pass

if __name__ == '__main__':
    main()