# -*- coding: utf-8 -*-
# 作为特征排序机制，他只对线性关系敏感。
# 如果关系是非线性的，即便两个变量具有一一对应的关系，Pearson相关性也可能会接近0。
# Scipy的 pearsonr 方法能够同时计算相关系数和p-value


import numpy as np
from scipy.stats import pearsonr

np.random.seed(0)
size = 300
x = np.random.normal(0, 1, size)

# help(pearsonr)
# returns (Pearson's correlation coefficient, 2-tailed p-value)

print "Lower noise", pearsonr(x, x + np.random.normal(0, 1, size))
print "Higher noise", pearsonr(x, x + np.random.normal(0, 10, size))


# 非线性测试：x和x^2的检验
x = np.random.uniform(-1, 1, 100000)
print pearsonr(x, x**2)[0]



