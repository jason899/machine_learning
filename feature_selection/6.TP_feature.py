# -*- coding: utf-8 -*-

'''
之所以叫做顶层，是因为他们都是建立在基于模型的特征选择方法基础之上的，
例如回归和SVM，在不同的子集上建立模型，然后汇总最终确定特征得分。

'''

'''
稳定性选择 Stability selection
在不同的数据子集和特征子集上运行特征选择算法，不断的重复，最终汇总特征选择结果，
比如可以统计某个特征被认为是重要特征的频率（被选为重要特征的次数除以它所在的子集被测试的次数）

好的特征不会因为有相似的特征、关联特征而得分为0，这跟Lasso是不同的。
对于特征选择任务，在许多数据集和环境下，稳定性选择往往是性能最好的方法之一。
'''

from sklearn.linear_model import RandomizedLasso
from sklearn.datasets import load_boston
boston = load_boston()

#using the Boston housing data.
#Data gets scaled automatically by sklearn's implementation
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]

rlasso = RandomizedLasso(alpha=0.025)
rlasso.fit(X, Y)

print "Features sorted by their score:"
print sorted(zip(map(lambda x: round(x, 4), rlasso.scores_),
                 names), reverse=True)


'''
反复的构建模型（如SVM或者回归模型）然后选出最好的（或者最差的）的特征（可以根据系数来选），
把选出来的特征放到一边，然后在剩余的特征上重复这个过程，直到所有特征都遍历了。

RFE的稳定性很大程度上取决于在迭代的时候底层用哪种模型。
例如，假如RFE采用的普通的回归，没有经过正则化的回归是不稳定的，那么RFE就是不稳定的；
假如采用的是Ridge，而用Ridge正则化的回归是稳定的，那么RFE就是稳定的。
'''

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

boston = load_boston()
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]

#use linear regression as the model
lr = LinearRegression()
#rank all features, i.e continue the elimination until the last one
rfe = RFE(lr, n_features_to_select=1)
rfe.fit(X,Y)

print "Features sorted by their rank:"
print sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names))

