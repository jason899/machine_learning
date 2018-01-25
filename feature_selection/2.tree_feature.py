# -*- coding: utf-8 -*-

# 基于树的方法比较易于使用，因为他们对非线性关系的建模比较好，并且不需要太多的调试。
# 但要注意过拟合问题，因此树的深度最好不要太大，再就是运用交叉验证。

from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor

import numpy as np

#Load boston housing dataset as an example
boston = load_boston()

# observe
boston.keys()
# ['data', 'feature_names', 'DESCR', 'target']
# print boston.DESCR

X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]

rf = RandomForestRegressor(n_estimators=20, max_depth=4)
scores = []
for i in range(X.shape[1]):
     score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",
                              cv=ShuffleSplit(len(X), 3, .3))
     scores.append((round(np.mean(score), 3), names[i]))

print sorted(scores, reverse=True)














