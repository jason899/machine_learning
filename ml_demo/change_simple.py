# -*- coding: utf-8 -*-
"""
Created on 20171108

@author: cjx
"""


import pandas as pd
import numpy as np
#np.random.seed(1000)
np.random.seed(10)


import sklearn.metrics as mt
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split



df = pd.read_csv("./data/change_data.csv", encoding='gb2312')
labels = pd.read_csv("./data/change_result.csv", encoding='gb2312')

# 用户id	账期（6、7、8月；YYYYMM）	基本月租	在网时长(月)	ARPU(元)	年龄	当前终端品牌	当前终端型号	当前终端使用开始时间	上一终端品牌	上一终端型号	上一终端使用开始时间	上一终端使用结束时间	终端平均使用时长(月)	总使用流量(MB)	4G使用流量	省外漫游流量	上网天数	总通话次数	总通话时长(MIN)	主叫通话次数	主叫通话时长	被叫通话次数	被叫通话时长	本地通话次数	本地通话时长	漫游通话次数	漫游通话时长	点对点短信次数
# user_id	bill_month	monthly_rent	net_dur	ARPU	age	terminal_brand	terminal_type	terminal_start_date	last_terminal_brand	last_terminal_type	last_terminal_start_date	last_terminal_end_date	avg_usage_time	totle_vol	4g_vol	roaming_vol	web_days	call_times	totle_call_dur	calling_times	calling_dur	called_times	called_dur	local_times	local_dur	roaming_times	roaming_dur	msg_times

# 用户id ,账期（6、7、8月；YYYYMM） ,基本月租 ,在网时长(月) ,ARPU(元) ,年龄 ,当前终端品牌 ,当前终端型号 ,当前终端使用开始时间 ,上一终端品牌 ,上一终端型号 ,上一终端使用开始时间 ,上一终端使用结束时间 ,终端平均使用时长(月) ,总使用流量(MB) ,4G使用流量 ,省外漫游流量 ,上网天数 ,总通话次数 ,总通话时长(MIN) ,主叫通话次数 ,主叫通话时长 ,被叫通话次数 ,被叫通话时长 ,本地通话次数 ,本地通话时长 ,漫游通话次数 ,漫游通话时长 ,点对点短信次数
# user_id ,bill_month ,monthly_rent ,net_dur ,ARPU ,age ,terminal_brand ,terminal_type ,terminal_start_date ,last_terminal_brand ,last_terminal_type ,last_terminal_start_date ,last_terminal_end_date ,avg_usage_time ,totle_vol ,4g_vol ,roaming_vol ,web_days ,call_times ,totle_call_dur ,calling_times ,calling_dur ,called_times ,called_dur ,local_times ,local_dur ,roaming_times ,roaming_dur ,msg_times

#df.describe()    # 如何查看每一列的最值情况？
# labels = df['flag'].astype(int).values


''' 缺失值处理 '''
# #last_terminal_brand	last_terminal_type: terminal_brand terminal_type
# # terminal_brand	last_terminal_brand	last_terminal_type terminal_type
# # switch_phone online_dates
# # 部分用其他值替换
# df.loc[df.last_terminal_brand.isnull(),'last_terminal_brand']= \
#     df.loc[df.last_terminal_brand.isnull(),'terminal_brand']
# df.loc[df.last_terminal_type.isnull(),'last_terminal_type']= \
#     df.loc[df.last_terminal_type.isnull(),'terminal_type']
#
# # 余下部分缺失值也可用null代替
# #df = df.fillna('Missing')
#
# # 特殊字符替代
# #df.replace("\N", 0, inplace=True)

# # 替换标注标签
# df.rename(columns={'switch_phone':'flag'}, inplace = True)

# 删除空白列
del df['web_days']
# 直接删除难以处理的分类变量
drop_columns = ['terminal_brand', 'terminal_type', 'terminal_start_date', 'last_terminal_brand', 'last_terminal_type', 'last_terminal_start_date', 'last_terminal_end_date']
df = df.drop(drop_columns, axis=1)


# ''' 特殊值处理 '''
# # 观察各列类型，object要作类型转换，一般是数值类型变量中含有\N等字符所致，逐一转成数值或string型
# # df.dtypes
# # df.bill_month.drop_duplicates()
# # aa = df['ARPU']
# # aa.to_csv('data/debug.csv')
#
# df.loc[(df.ARPU=='\N'), 'ARPU'] = 0
# df['ARPU'] = df['ARPU'].astype('float64')
# df.loc[(df.monthly_rent=='\N'), 'monthly_rent'] = 0
# df['monthly_rent'] = df['monthly_rent'].astype('float64')
#
#
# # # 观察缺失值所在列
# # df.isnull().any()
# # df[df.isnull().values==True].head(6)
# # #某个元素所在行和列: df.columns, df.index
# # np.where(df==201706)
# # df.loc[(df.monthly_rent==np.nan), 'monthly_rent'] = 0
#
# df.loc[(df.monthly_rent.isnull()==True), 'monthly_rent'] = 0
# df.loc[(df.age.isnull()==True), 'age'] = 0


# 用均值填空值或\N
df.loc[(df.ARPU=='\N'), 'ARPU'] = np.mean(df.loc[(df.ARPU!='\N'), 'ARPU'].astype('float64'))

df.loc[(df.monthly_rent=='\N'), 'monthly_rent'] = \
    np.mean(df.loc[(df.monthly_rent!='\N'), 'monthly_rent'].astype('float64'))

df['ARPU'] = df['ARPU'].astype('float64')
df['monthly_rent'] = df['monthly_rent'].astype('float64')
df.loc[(df.monthly_rent.isnull()==True), 'monthly_rent'] = np.mean(df.monthly_rent)
df.loc[(df.age.isnull()==True), 'age'] = np.mean(df.age)



'''import data pivot'''
def data_pivot(df):
    df1 = pd.merge(df[df.iloc[:, 1] == 201706], df[df.iloc[:, 1] == 201707]\
                   , on='user_id', how='left', suffixes=('_06', ''))
    df2 = pd.merge(df1, df[df.iloc[:, 1] == 201708], on='user_id', how='left', suffixes=('_07', '_08'))
    return df2


view_data = data_pivot(df)
#data和label先关联再拆分，防止2部分数据user_id顺序不一致
total_data = pd.merge(view_data, labels, on='user_id')
drop_columns= ['user_id', 'bill_month_06', 'bill_month_07', 'bill_month_08', 'flag']
x, y = total_data.drop(drop_columns, axis=1), total_data['flag']



'''train test split'''
test_size = 0.3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)


''' 随机森林 '''
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, criterion='entropy')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)          # 预测值

print mt.f1_score(y_test, y_pred)







######################################################## test set ####################################
df = pd.read_csv("./data/change_data_test.csv", encoding='gb2312')


# 删除空白列
del df['web_days']
# 直接删除难以处理的分类变量
drop_columns = ['terminal_brand', 'terminal_type', 'terminal_start_date', 'last_terminal_brand', 'last_terminal_type', 'last_terminal_start_date', 'last_terminal_end_date']
df = df.drop(drop_columns, axis=1)
# df.dtypes


# object类型转换
df.loc[(df.ARPU=='\N'), 'ARPU'] = 0
df['ARPU'] = df['ARPU'].astype('float64')
df.loc[(df.monthly_rent=='\N'), 'monthly_rent'] = 0
df['monthly_rent'] = df['monthly_rent'].astype('float64')
# 缺失值处理
df.loc[(df.monthly_rent.isnull()==True), 'monthly_rent'] = 0
df.loc[(df.age.isnull()==True), 'age'] = 0



data_val = data_pivot(df)
drop_columns= ['user_id', 'bill_month_06', 'bill_month_07', 'bill_month_08']
x_val = data_val.drop(drop_columns, axis=1)

# # practice by all sample data again
# clf.fit(x, y)

y_val = clf.predict(x_val)          # 预测值

''' save predicted label '''
data_val.insert(1,'flag', y_val)
result = data_val[['user_id', 'flag']]
result = result.set_index(['user_id'])
result.to_csv('./data/result/change.csv')



















######################################################## 其他模型 ####################################
''' gbdt '''
from sklearn.ensemble import GradientBoostingClassifier
n_estimator = 100
clf = GradientBoostingClassifier(n_estimators=n_estimator, max_features=0.2)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)          # 预测值

print mt.f1_score(y_test, y_pred)


''' GridSearchCV '''
# gbc_cv_gs = model_selection.GridSearchCV(gbc, parameters, scoring="f1")
# from sklearn.grid_search import GridSearchCV
# grid_search = GridSearchCV(gbc, parameters, scoring="f1")

from sklearn import model_selection
gbc = GradientBoostingClassifier()
parameters = {'n_estimators': [10, 100, 200], 'max_features': [0.1, 0.2, 0.3]}
grid_search = model_selection.GridSearchCV(gbc, parameters, scoring="f1")
grid_search.fit(x_train, y_train)

print('最佳效果：')
print grid_search.best_score_
print('最优参数：')
print grid_search.best_estimator_



# ######################################################## 数据处理 ####################################
#data_val.insert(1,'flag', y_val)
# labels = df['flag'].astype(int).values

# ######################################################## 结果分析 ####################################
# ''' confusion '''
# confusion = mt.confusion_matrix(y_test, y_pred)
# print confusion
#
# #print 'Accuracy:', accuracy_score(y_test, y_pred)
# print 'Precision:', mt.precision_score(y_test, y_pred)
# print 'Recall:', mt.recall_score(y_test, y_pred)
# print 'F1-score:', mt.f1_score(y_test, y_pred)
#
# ''' auc'''
# fpr, tpr, thresholds = mt.roc_curve(y_test, y_pred)
# auc_grd = mt.auc(fpr, tpr)
# print 'AUC:', auc_grd


# ###################################################### 纵表转横表 ###################################
# def data_pivot(df):
#     tmp_data = df.loc[df.loc[:, 'bill_month'] == 201706]
#     tmp_data = tmp_data.rename(columns=lambda x: x+'_201706')
#     tmp_data.rename(columns={'user_id_201706':'user_id'}, inplace = True)
#     data = tmp_data
#
#     tmp_data = df.loc[df.loc[:, 'bill_month'] == 201707]
#     tmp_data = tmp_data.rename(columns=lambda x: x+'_201707')
#     tmp_data.rename(columns={'user_id_201707':'user_id'}, inplace = True)
#     data = pd.merge(data, tmp_data, on='user_id')
#
#     tmp_data = df.loc[df.loc[:, 'bill_month'] == 201708]
#     tmp_data = tmp_data.rename(columns=lambda x: x+'_201708')
#     tmp_data.rename(columns={'user_id_201708':'user_id'}, inplace = True)
#     data = pd.merge(data, tmp_data, on='user_id')
#     return data
#
#
# total_data = data_pivot(df)
# drop_columns = ['user_id', 'bill_month_201706', 'bill_month_201707', 'bill_month_201708', 'flag_201706', 'flag_201707', 'flag_201708']
# features = total_data.drop(drop_columns, axis=1)
# labels = total_data['flag_201708']
# x, y = features, labels



# ###################################################### 纵表转横表 ###################################
# def data_pivot(df):
#     tmp_data = df.loc[df.loc[:, 'bill_month'] == 201706]
#     tmp_data = tmp_data.rename(columns=lambda x: x+'_201706')
#     tmp_data.rename(columns={'user_id_201706':'user_id'}, inplace = True)
#     data = tmp_data
#
#     tmp_data = df.loc[df.loc[:, 'bill_month'] == 201707]
#     tmp_data = tmp_data.rename(columns=lambda x: x+'_201707')
#     tmp_data.rename(columns={'user_id_201707':'user_id'}, inplace = True)
#     data = pd.merge(data, tmp_data, on='user_id')
#
#     tmp_data = df.loc[df.loc[:, 'bill_month'] == 201708]
#     tmp_data = tmp_data.rename(columns=lambda x: x+'_201708')
#     tmp_data.rename(columns={'user_id_201708':'user_id'}, inplace = True)
#     data = pd.merge(data, tmp_data, on='user_id')
#     return data
#
#
# total_data = data_pivot(df)
# drop_columns = ['user_id', 'bill_month_201706', 'bill_month_201707', 'bill_month_201708', 'flag_201706', 'flag_201707', 'flag_201708']
# features = total_data.drop(drop_columns, axis=1)
# labels = total_data['flag_201708']
# x, y = features, labels


# ###################################################### xgboost ###################################
# import xgboost as xgb
# # XGBoost
# y_train[y_train == 3] = 0
# y_test[y_test == 3] = 0
# data_train = xgb.DMatrix(x_train, label=y_train)
# data_test = xgb.DMatrix(x_test, label=y_test)
# watch_list = [(data_test, 'eval'), (data_train, 'train')]
# param = {'max_depth': 3, 'eta': 1, 'silent': 0, 'objective': 'multi:softmax', 'num_class': 3}
# bst = xgb.train(param, data_train, num_boost_round=4, evals=watch_list)
# y_hat = bst.predict(data_test)
# show_accuracy(y_hat, y_test, 'XGBoost ')


