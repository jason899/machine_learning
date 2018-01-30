#coding=utf-8
#%matplotlib inline 
import numpy
import matplotlib.pyplot as plt

import matplotlib as mpl
#mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] #指定默认字体  
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体  
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
# sns.axes_style()，可以看到是否成功设定字体为微软雅黑。

#correlations = data.corr()  #计算变量之间的相关系数矩阵
correlations = [[1,0.94],[0.94,1]]

# plot correlation matrix
fig = plt.figure() #调用figure创建一个绘图对象
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)  #绘制热力图，从-1到1
fig.colorbar(cax)  #将matshow生成热力图设置为颜色渐变条


names = [u'新闻报道位置（可多选）',u'新闻态度与语气']
#names = [u'位置',u'态度与语气']
#names = [u'yuqi',u'taidu']
ticks = numpy.arange(0,2,1) #生成0-2，步长为1
ax.set_xticks(ticks)  #生成刻度
ax.set_yticks(ticks)
ax.set_xticklabels(names) #生成x轴标签
ax.set_yticklabels(names)
plt.show()