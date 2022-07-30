# -*- coding:utf-8 -*-
# @Time : 2022/7/29 8:12 下午
# @Author : Bin Bin Xue
# @File : chart4_数据预处理
# @Project : data_mining

'''
第四章 数据预处理
    4.1 数据预处理的必要性
            数据挖掘所需的高质量数据：
            (1)准确性 - 信息准确
            (2)完整性 - 无缺失值
            (3)一致性 - 统一的格式
            (4)时效性 - 近期数据
            (5)可信性 - 真实性
            (6)可解释性 - 容易理解

    4.2 数据清洗
            pandas数据清洗：
                1_缺失值检测
                    (1) 缺失值的检测与统计：df.isnull()/df.info()
                    (2) 缺失值统计：df.isnull().sum()
                2_缺失值处理
                    (1) 删除缺失值：df.dropna() - 删除所有含缺失值的行
                                                df.dropna(thresh) - 若有thresh个不缺失值，则保留该行
                                                df.dropna(axis=1,how='all') - 按列删除（全缺失才删）
                    (2) 填充缺失值：df.fillna({列号:填充值,..}) - 自定义数值，按列填充
                                                df.fillna(method='ffill') - 前向填充
                                                df[列].fillna(df[列].mean()) - 用列的均值填充列
                3_数据值替换
                    (1)多值替换：df.replace([v1,v2],[v3,v4]) - v3替换v1,v4替换v2
                                            df.replace({v1:v3,v2:v4})
                4_利用函数或映射进行数据转换
                    (1)使用函数多列每个元素操作：df['新列名']=df[列名].map(函数名)
                5_异常值检测
                    (1)散点图检测：绘制散点图观察
                    (2)箱型图分析：plt.boxplot(df['列'].values,notch=True)
                    (3)3sigma法则：。。。

    4.3 数据集成
            1_数据集成过程中的关键问题
                实体识别：识别不同名的同一实体
                数据冗余：一个属性能由另一个或一组属性值推出，则可能冗余
                相关性分析：检测数据冗余问题。
                                    标称属性：用卡方检验进行分析；
                                    数值属性：相关系数或协方差
                                    （df.列1.cov(df.列2);df.列1.corr(df.列2))
                元组重复
                数据值冲突：特征量纲/评分制度不同
            2_利用pandas合并数据
                按关键值合并：pd.merge(left,right,left_on,right_on,how,suffixes)
                堆叠合并：pd.concat([df1,df2..],axis)
                存在重复索引时合并：df1.combine_first(df2)

    4.4 数据标准化（消除不同特征取值范围不同带来的差异）
            1_极差标准化数据（原始数据映射到0、1区间）
                x1=(x-min)/(max-min)
            2_标准差标准化数据（用的最广泛，均值为0和标准差为1）
                x1=(x-mean)/std

    4.5 数据规约
            1_维规约（减少特征）
                属性子集选择
                小波变换
                主成分分析
            2_数量规约（减少数据集量）
                抽样等
            3_数据压缩（可逆）
                小波变换（有损压缩）

    4.6 数据变换与数据离散化
            1_数据变换的策略
                光滑：去除噪声
                属性构造：构造新特征
                聚集：聚集数据，构造数据立方体
                规范化：数据范围统一
                离散化：数据属性分区间
            2_Python数据变换与离散化
                (1) 数据规范化（最小-最大规范，零均值规范，代码见下）
                (2) 类别数据的哑变量处理 - pd.get_dummies(df)
                (3) 连续型变量的离散化 - pd.cut(data,bins)
              (bins为list时，等宽法：按列表值分割区间；为int时，等频法：按频数划分区间）

    4.7 利用sklearn进行数据预处理
            sklearn.preprocessing
        1_数据标准化、均值和方差缩放
            -标准化至（均值为0，标准差为1）
                preprocessing.scale(X_train)
            -用训练集标准化的均值和标准差得到转换器，对测试集标准化
                scaler = preprocessing.StandardScaler().fit(X_train)
                scaler.transform(X_train)
                scaler.transform(X_test)
        2_特征缩放（也是标准化的一种方式）
            (1)一般特征值缩放
            - 特征缩放到指定区间（默认为0-1）
                min_max_scaler = preprocessing.MinMaxScaler()
                min_max_scaler.fit_transform(X_train)
                X_test_minmax=min_max_scaler.transform(X_test)
                    可指定缩放范围（min,max）：
                        X_std = (X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
                        X_scaled = X_std*(max-min)+min
            - 特征缩放到(-1,1)，适合稀疏数据或已经零中心化的数据
                max_Abs_scaler = preprocessing.MaxAbsScaler()
                max_Abs_scaler.fit_transform(X_train)
                max_Abs_scaler.transform(X_test)
            (2)稀疏缩放数据
                MaxAbsScaler和maxabsscale适合
            (3)带异常值的缩放数据
                robust_scale、RobustScaler
        3_非线性变换
            (1)映射到均匀分布（数据映射到值为0-1的均匀分布,保证特征值的秩）
                quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
                quantile_transformer.fit_transform(X_train)
                quantile_transformer.transform(X_test)
            (2)映射到高斯分布（将指定数据集从任意分布映射到尽可能接近高斯分布）
                -Box-Cox变换用于严格的正数据
                    pt = preproceessing.PowerTransform(method='box-cox',standardize=False)
                    pt.fit_transform(X_train)
                -QuantileTransformer实现
                    quantile_transformer=QuantileTransformer(output_distribution='normal',random_state=0)
                    ..
        4_正则化
            将单个样本缩放到单位范数(每个样本的范数为1)
                preprocessing.normalize(X,norm='l2')

        5_编码分类特征(类别特征转数值)
            1)序数编码（类1、类2、类3 -> 1、2、3）
                enc = preprocessing.OrdinalEncoder()
                enc.fit(X)
                enc.transform('列1值','列2值') -> 输出对应序数编码
            2)One-hot编码或dummy编码
                enc = preprocessing.OneHotEncoder()
                enc.fit(X)
                R=enc.transform(..).toarray()
                display(R)
        6_离散化（连续特征转离散）
            1)K桶离散化（将特征离散到K个桶(bin)中）
                est = preprocessing.KBinsDiscretizer(n_bins=[1桶数,..],encoding='ordinal').fit(X)
                est.transform(x)
            2)特征二值化（对数字特征进行阈值化后便于获得布尔值）
                binarizer=preprocessing.Binarizer().fit(X)
                Y1=binaizer.transform(X)
                    可调整阈值(边界值)：.Binarizer(threshold=1.1)

'''
print('数据标准化')
# 离散标准化数据
def MinMaxScale(data):
    data = (data-data.min())/(data.max()-data.min())
    return data
# 标准差标准化数据
def StandardScale(data):
    data = (data-data.mean())/data.std()
    return data

print('利用sklearn进行数据预处理')
from sklearn import preprocessing
import numpy as np

print('数据的标准化、均值和标准差示例求解：scale、scaler')
X_train = np.array([[1.,-2.,1.5],[2.2,1.3,0.5],[0.3,1.,-1.5]])
X_scaled = preprocessing.scale(X_train)
print('X_train:\n',X_train)
print('X_scaled\n',X_scaled)
print('均值',X_scaled.mean(axis=0))
print('标准差',X_scaled.std(axis=0))

print('程序类实现标准化：StandardScaler')
scaler = preprocessing.StandardScaler().fit(X_train)
print('scaler.scale_',scaler.scale_)
print('scaler.mean_',scaler.mean_)
scaler.transform(X_train)
X_test = [[-1.,1.,0]]
X_test_scale = scaler.transform(X_test)
print('X_test_scale',X_test_scale)

X_train = np.array([[1.,-1.,2.],[2.,0.,0.],[0.,1.,-1.]])
print('一般特征值缩放：MinMaxScaler或MaxAbsScaler')
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
print('原始数据：\n',X_train)
print('归一化：\n',X_train_minmax)
X_test = np.array([[-3.,-1.,4.]])
X_test_minmax = min_max_scaler.fit(X_test)
print('测试数据：',X_test)
print('归一化的测试数据：\n',X_test_minmax)
print(' ',min_max_scaler.scale_)
print(' ',min_max_scaler.min_)

print('非线性变换-映射到0-1均匀分布：QuantileTransform')
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
X,y = load_iris(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X, y,random_state=0)
quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
X_train_trans = quantile_transformer.fit_transform(X_train)
X_test_trans = quantile_transformer.transform(X_test)
print(np.percentile(X_train[:,0],[0,25,50,75,100]))
print(np.percentile(X_train_trans[:,0],[0,25,50,75,100]))

print('非线性变换-映射到高斯分布：PowerTransform')
pt = preprocessing.PowerTransformer(method='box-cox',standardize=False)
X_lognormal = np.random.RandomState(616).lognormal(size=(3,3))
print(X_lognormal)
T = pt.fit_transform(X_lognormal)
print(T)

print('非线性变换-映射到高斯分布：QuantileTransformer')
X,y=load_iris(return_X_y=True)
quantile_transformer = preprocessing.QuantileTransformer(
    output_distribution='normal',random_state=0
)
X_trans = quantile_transformer.fit_transform(X)
print(quantile_transformer.quantiles_)

print('正则化：normalize')
X = [[1.,-1.,2.],[2.,0.,0.],[0.,1.,-1.]]
X_normalized = preprocessing.normalize(X,norm='l2')
print(X_normalized)

print('数据编码-序数编码')
enc = preprocessing.OrdinalEncoder()
X = [['male','from US','uses Safari'],['female','from Europe','uses Firefox']]
enc.fit(X)
print(enc.transform([['female','from Europe','uses Firefox']]))

print('数据编码-one-hot编码')
enc = preprocessing.OneHotEncoder()
enc.fit(X)
R=enc.transform(X).toarray()
print(R)

print('数据离散化-K桶离散化')
X = np.array([[-3.,5.,15],[0.,6,14],[6.,3.,11]])
est = preprocessing.KBinsDiscretizer(n_bins=[3,2,2],encode='ordinal').fit(X)
X_est = est.transform(X)
print(X_est)

print('数据离散化-特征二值化')
X = [[1.,-1.,2.],[2.,0.,0.],[0.,1.,-1.]]
binarizer = preprocessing.Binarizer().fit(X)
Y1 = binarizer.transform(X)
print(Y1)
binarizer = preprocessing.Binarizer(threshold=1.1)
Y2 = binarizer.transform(X)
print(Y2)

