# -*- coding:utf-8 -*-
# @Time : 2022/7/29 8:19 下午
# @Author : Bin Bin Xue
# @File : chart8_聚类
# @Project : data_mining

'''
第八章
    8.1 聚类分析概述
        数据聚类是无监督学习的主要应用之一。
        分为四类：
          (1)基于划分方法
            K-Means、K中心点
          (2)基于层次方法
            自顶向下、自顶向上
          (3)基于密度方法
            DBSCAN、OPTICS、DENCLUE
          (4)基于网格方法

    8.2 K-Means聚类
        评价：最常用的一种，需要设置聚类个数
        实现：sklearn.cluster.KMeans

    8.3 层次聚类
        评价：核心问题是测量两个簇之间的距离方法（方法不同，结果不同）
        实现：sklearn.cluster.AgglomerativeClustering

    8.4 基于密度的聚类
        评价：能更好地发现任意形状的聚类簇（前面两个主要发现凸型聚类簇）
        实现：DBSCAN - 非调包

    8.5 其他聚类方法
        网格聚类STING
        概念聚类COBWEB
        模糊聚类FCM

    8.6 聚类评估
        目标：对聚类可行性和使用的方法产生结果的质量进行评估
        结果度量指标：
            (1)外在方法：
                熵、纯度、精度、召回率、F度量、兰德系数RI和调整兰德系数ARI
            (2)内在方法：
                轮廓系数
'''
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# 通过KMeans划分的结果跟实际标签差别还是有点大
def KMeansTest():
    iris = load_iris()
    x = iris.data
    y = iris.target
    estimator = KMeans(n_clusters=3)
    estimator.fit(x)
    label_pred = estimator.labels_
    print('label_pred:\n',label_pred)
    print('label_reality:\n',y)

from sklearn.datasets._samples_generator import make_blobs
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

def AgglomerTest():
    # 产生随机数据的中心
    centers = [[1,1],[-1,-1],[1,-1]]
    # 产生的数据个数
    n_samples = 3000
    # 产生数据
    X,labels_true = make_blobs(n_samples=n_samples,centers=centers,
                               cluster_std=0.6,random_state=0)
    # 设置分层聚类函数
    linkages = ['ward','average','complete']
    n_clusters_ = 3
    ac = AgglomerativeClustering(linkage=linkages[2],n_clusters=n_clusters_)
    # 训练数据
    ac.fit(X)
    # 每个数据的分类
    labels = ac.labels_
    plt.figure(1)
    plt.clf()
    colors = cycle('bgrcmy')
    for k,col in zip(range(n_clusters_),colors):
        my_members = labels == k
        plt.plot(X[my_members,0],X[my_members,1],col+'.')
    plt.title('Estimated number of clusters:%d' % n_clusters_)
    plt.show()

import random
from sklearn import datasets
def DBSCANTest():
    def findNeighbor(j,X,eps):
        N = []
        for p in range(X.shape[0]):
            temp = np.sqrt(np.sum(np.square(X[j]-X[p])))
        # 欧式距离
            if(temp <= eps):
                N.append(p)
        return N

    def dbscan(X,eps,min_Pts):
        k = -1
        NeighborPts = []
        Ner_NeighborPts = []
        fil = []
        gama = [x for x in range(len(X))]
        cluster = [-1 for y in range(len(X))]
        while len(gama)>0:
            j = random.choice(gama)
            gama.remove(j)
            fil.append(j)
            NeighborPts = findNeighbor(j,X,eps)
            if len(NeighborPts)<min_Pts:
                cluster[j]=-1
            else:
                k = k+1
                cluster[j]=k
                for i in NeighborPts:
                    if i not in fil:
                        gama.remove(i)
                        fil.append(i)
                        Ner_NeighborPts = findNeighbor(i,X,eps)
                        if len(Ner_NeighborPts) >= min_Pts:
                            for a in Ner_NeighborPts:
                                if a not in NeighborPts:
                                    NeighborPts.append(a)
                        if (cluster[i]==-1):
                            cluster[i]=k
        return cluster

    x1,y1 = datasets.make_circles(n_samples=1000,factor=.6,noise=.05)
    x2,y2 = make_blobs(n_samples=300,n_features=2,centers=[[1.2,1.2]],cluster_std=[[.1]],random_state=9)
    X = np.concatenate((x1,x2))
    eps = 0.08
    min_pts = 10
    C = dbscan(X,eps,min_pts)
    plt.figure(figsize=(12,9),dpi=80)
    plt.scatter(X[:,0],X[:,1],c=C)
    plt.show()

from sklearn import metrics
def ARITest():
    labels_true = [0,0,0,1,1,1]
    labels_pred = [0,0,1,1,2,2]
    print(metrics.adjusted_rand_score(labels_true,labels_pred))


def silhouette_scoreTest():
    X = load_iris().data
    kmeans_model = KMeans(n_clusters=3,random_state=1).fit(X)
    labels = kmeans_model.labels_
    print(metrics.silhouette_score(X,labels,metric='euclidean'))


def main():
    # KMeansTest()
    # AgglomerTest()
    # DBSCANTest()
    # 调整兰德系数（反应聚类结果的准确性）
    ARITest()
    # 轮廓系数（反应簇的紧凑性，值在-1～1，越接近1越好，负数说明存在问题）
    silhouette_scoreTest()


if __name__ == '__main__':
    main()
