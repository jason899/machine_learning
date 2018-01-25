# -*- coding: utf-8 -*-

'''
在噪音不多的数据上，或者是数据量远远大于特征数的数据上，
如果特征之间相对来说是比较独立的，那么即便是运用最简单的线性回归模型也一样能取得非常好的效果。

当过个特征互联关联，模型预测就会很困难，出现多重共线性问题
例如，假设我们有个数据集，它的真实模型应该是Y=X1+X2，当我们观察的时候，发现Y=X1+X2+e，e是噪音
如果X1和X2之间存在线性关系，例如X1约等于X2，这个时候由于噪音e的存在，
我们学到的模型可能就不是Y=X1+X2了，有可能是Y=2X1，或者Y=-X1+3X2
'''

from sklearn.linear_model import LinearRegression
import numpy as np

np.random.seed(0)
size = 5000
#A dataset with 3 features
X = np.random.normal(0, 1, (size, 3))
#Y = X0 + 2*X1 + noise
Y = X[:,0] + 2*X[:,1] + np.random.normal(0, 2, size)

lr = LinearRegression()
lr.fit(X, Y)

#A helper method for pretty-printing linear models
def pretty_print_linear(coefs, sort = False):
    names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)

    if sort:
        lst = sorted(lst, key=lambda x:-np.abs(x[0]))

    val = " + ".join("%s * %s" % (round(coef,3), name) for coef, name in lst)
    return val

print "Linear model:", pretty_print_linear(lr.coef_)


