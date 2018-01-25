# -*- coding: utf-8 -*-

'''
正则化就是把额外的约束或者惩罚项加到已有模型（损失函数）上，以防止过拟合并提高泛化能力。
损失函数由原来的E(X,Y)变为E(X,Y)+alpha||w||，w是模型系数组成的向量，
||·||一般是L1或者L2范数，alpha是一个可调的参数，控制着正则化的强度。
当用在线性模型上时，L1正则化和L2正则化也称为Lasso和Ridge
'''

#### Lasso
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston

import numpy as np


boston = load_boston()
scaler = StandardScaler()
X = scaler.fit_transform(boston["data"])
Y = boston["target"]
names = boston["feature_names"]

lasso = Lasso(alpha=.3)
lasso.fit(X, Y)

#A helper method for pretty-printing linear models
def pretty_print_linear(coefs, names=None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)

    if sort:
        lst = sorted(lst, key=lambda x:-np.abs(x[0]))

    val = " + ".join("%s * %s" % (round(coef,3), name) for coef, name in lst)
    return val

print "Lasso model: ", pretty_print_linear(lasso.coef_, names, sort = True)


#### Ridge
'''
不同的数据上线性回归得到的模型（系数）相差甚远，
但对于L2正则化模型来说，结果中的系数非常的稳定，差别较小，都比较接近于1，能够反映出数据的内在结构。
'''

from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

size = 100

#We run the method 10 times with different random seeds
for i in range(10):
    print "Random seed %s" % i
    np.random.seed(seed=i)
    X_seed = np.random.normal(0, 1, size)
    X1 = X_seed + np.random.normal(0, .1, size)
    X2 = X_seed + np.random.normal(0, .1, size)
    X3 = X_seed + np.random.normal(0, .1, size)
    Y = X1 + X2 + X3 + np.random.normal(0, 1, size)
    X = np.array([X1, X2, X3]).T
    lr = LinearRegression()
    lr.fit(X,Y)
    print "Linear model:", pretty_print_linear(lr.coef_)
    ridge = Ridge(alpha=10)
    ridge.fit(X,Y)
    print "Ridge model:", pretty_print_linear(ridge.coef_)
    print



#
# np.random.seed(0)
# size = 5000
# #A dataset with 3 features
# X = np.random.normal(0, 1, (size, 3))
# #Y = X0 + 2*X1 + noise
# Y = X[:,0] + 2*X[:,1] + np.random.normal(0, 2, size)
#
# lr = LinearRegression()
# lr.fit(X, Y)
#
# #A helper method for pretty-printing linear models
# def pretty_print_linear(coefs, sort = False):
#     names = ["X%s" % x for x in range(len(coefs))]
#     lst = zip(coefs, names)
#
#     if sort:
#         lst = sorted(lst, key=lambda x:-np.abs(x[0]))
#
#     val = " + ".join("%s * %s" % (round(coef,3), name) for coef, name in lst)
#     return val
#
# print "Linear model:", pretty_print_linear(lr.coef_)

