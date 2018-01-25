# -*- coding: utf-8 -*-

'''
平均不纯度减少 mean decrease impurity
利用不纯度可以确定节点（最优条件），对于分类问题，通常采用 基尼不纯度 或者 信息增益 ，
对于回归问题，通常采用的是 方差 或者最小二乘拟合。
当训练决策树的时候，可以计算出每个特征减少了多少树的不纯度。
对于一个决策树森林来说，可以算出每个特征平均减少了多少不纯度，并把它平均减少的不纯度作为特征选择的值。
'''

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import numpy as np

#Load boston housing dataset as an example
boston = load_boston()
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]

rf = RandomForestRegressor()
rf.fit(X, Y)

rf.feature_importances_

print "Features sorted by their score:"
print sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),
             reverse=True)

'''
这里特征得分实际上采用的是 Gini Importance 。使用基于不纯度的方法的时候，要记住：
1、这种方法存在 偏向 ，对具有更多类别的变量会更有利；
2、对于存在关联的多个特征，其中任意一个都可以作为指示器（优秀的特征），
并且一旦某个特征被选择之后，其他特征的重要度就会急剧下降。在理解数据时，这就会造成误解
'''

# rf = RandomForestRegressor(n_estimators=20, max_features=2)
# rf.fit(X, Y)
# print "Scores for X0, X1, X2:", map(lambda x:round (x,3),
#                                     rf.feature_importances_)


'''
平均精确率减少 Mean decrease accuracy
打乱每个特征的特征值顺序，并且度量顺序变动对模型的精确率的影响。
对于不重要的变量来说，打乱顺序对模型的精确率影响不会太大，
但是对于重要的变量来说，打乱顺序就会降低模型的精确率。
'''

from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict

X = boston["data"]
Y = boston["target"]

rf = RandomForestRegressor()
scores = defaultdict(list)

#crossvalidate the scores on a number of different random splits of the data
for train_idx, test_idx in ShuffleSplit(len(X), 100, .3):
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    r = rf.fit(X_train, Y_train)
    acc = r2_score(Y_test, rf.predict(X_test))

    for i in range(X.shape[1]):
        X_t = X_test.copy()
        np.random.shuffle(X_t[:, i])
        shuff_acc = r2_score(Y_test, rf.predict(X_t))
        scores[names[i]].append((acc-shuff_acc)/acc)

print "Features sorted by their score:"
print sorted([(round(np.mean(score), 4), feat) for
              feat, score in scores.items()], reverse=True)
'''
LSTAT和RM这两个特征对模型的性能有着很大的影响，
打乱这两个特征的特征值使得模型的性能下降了73%和57%。
'''





