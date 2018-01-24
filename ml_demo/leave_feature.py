# -*- coding: utf-8 -*-
"""
Created on 20171108

@author: cjx

# 用户id	账期（6、7、8月；YYYYMM）	基本月租	ARPU(元)	总使用流量(MB)	4G使用流量	省外漫游流量	上网天数	总通话次数	总通话时长(MIN)	主叫通话次数	主叫通话时长	被叫通话次数	被叫通话时长	本地通话次数	本地通话时长	漫游通话次数	漫游通话时长	点对点短信次数	停机次数	停机天数
# user_id	bill_month	monthly_rent	ARPU	totle_vol	4g_vol	roaming_vol	web_days	call_times	totle_call_dur	calling_times	calling_dur	called_times	called_dur	local_times	local_dur	roaming_times	roaming_dur	msg_times	halt_times	halt_days

# 用户id ,账期（6、7、8月；YYYYMM） ,基本月租 ,ARPU(元) ,总使用流量(MB) ,4G使用流量 ,省外漫游流量 ,上网天数 ,总通话次数 ,总通话时长(MIN) ,主叫通话次数 ,主叫通话时长 ,被叫通话次数 ,被叫通话时长 ,本地通话次数 ,本地通话时长 ,漫游通话次数 ,漫游通话时长 ,点对点短信次数 ,停机次数 ,停机天数
# user_id ,bill_month ,monthly_rent ,ARPU ,totle_vol ,4g_vol ,roaming_vol ,web_days ,call_times ,totle_call_dur ,calling_times ,calling_dur ,called_times ,called_dur ,local_times ,local_dur ,roaming_times ,roaming_dur ,msg_times ,halt_times ,halt_days
#df.describe()    # 如何查看每一列的最值情况？
# labels = df['flag'].astype(int).values
"""


import pandas as pd
import numpy as np
#np.random.seed(1000)
np.random.seed(10)


import sklearn.metrics as mt
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
# from sklearn import preprocessing


df = pd.read_csv("./data/leave_data.csv", encoding='gb2312')
labels = pd.read_csv("./data/leave_result.csv", encoding='gb2312')


''' 缺失值处理 '''
# #缺失值所在列
# df.isnull().any()
# df[df.isnull().values==True].head(6)
# #某个元素所在行和列: df.columns, df.index
# np.where(df==201706)
# df.loc[(df.monthly_rent==np.nan), 'monthly_rent'] = 0

#df.loc[(df.monthly_rent.isnull()==True), 'monthly_rent'] = 0
# df.loc[(df.monthly_rent.isnull()==True), 'monthly_rent'] = np.mean(df.monthly_rent)
df = df.fillna(-1).astype('float64')


'''import data pivot'''
def data_pivot(df):
    df1 = pd.merge(df[df.iloc[:, 1] == 201706], df[df.iloc[:, 1] == 201707]\
                   , on='user_id', how='left', suffixes=('_06', ''))
    df2 = pd.merge(df1, df[df.iloc[:, 1] == 201708], on='user_id', how='left', suffixes=('_07', '_08'))
    return df2

''' feature difference '''
def feature_diff(df):
    new_feature = df
    columns = list(df.columns)
    columns.remove('user_id')
    ii = 0
    delta = len(columns)/3
    for col in columns:
        new_feature[col+'76'] = df.iloc[:,ii+delta] - df.iloc[:,ii]
        new_feature[col+'86'] = df.iloc[:,ii+delta*2] - df.iloc[:,ii]
        new_feature[col+'87'] = df.iloc[:,ii+delta*2] - df.iloc[:,ii+delta]
        ii += 1
    return new_feature

view_data = data_pivot(df)
#data和label先关联再拆分，防止2部分数据user_id顺序不一致
total_data = pd.merge(view_data, labels, on='user_id')
# drop_columns = ['user_id', 'bill_month_06', 'bill_month_07', 'bill_month_08', 'flag']
# features = total_data.drop(drop_columns, axis=1)
# x, y = features, total_data['flag']
drop_columns = ['bill_month_06', 'bill_month_07', 'bill_month_08', 'flag']
features = feature_diff(total_data.drop(drop_columns, axis=1))
x, y = features.drop('user_id', axis=1), total_data['flag']
# scaled_features = preprocessing.scale(features.astype(float))
# x, y = scaled_features, total_data['flag']


'''train test split'''
test_size = 0.3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)



''' 随机森林 '''
from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=100, criterion='gini')
clf = RandomForestClassifier(n_estimators=100, criterion='entropy')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)          # 预测值

print mt.f1_score(y_test, y_pred)




######################################################## test set ####################################
dftest = pd.read_csv("./data/leave_data_test.csv", encoding='gb2312')

# dftest[dftest.isnull().values==True]
dftest.loc[(dftest.monthly_rent.isnull()==True), 'monthly_rent'] = 0
data_val = data_pivot(dftest)

# drop_columns= ['user_id', 'bill_month_06', 'bill_month_07', 'bill_month_08']
# x_val = data_val.drop(drop_columns, axis=1)
drop_columns = ['bill_month_06', 'bill_month_07', 'bill_month_08', 'flag']
features = feature_diff(total_data.drop(drop_columns, axis=1))
x_val = features.drop('user_id', axis=1)


# # practice by all sample data again
# clf.fit(x, y)

y_val = clf.predict(x_val)          # 预测值

''' save predicted label '''
data_val['flag'] = y_val
result = data_val[['user_id', 'flag']]
result = result.set_index(['user_id'])
result.to_csv('./data/result/leave_feature.csv')






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
# parameters = {'n_estimators': [10, 100, 200], 'max_features': [0.1, 0.2, 0.3]}
parameters = {'n_estimators': [10, 100, 200, 500, 1000], 'max_features': [0.1, 0.2, 0.3,0.5, 0.9, 1]}
grid_search = model_selection.GridSearchCV(gbc, parameters, scoring="f1")
grid_search.fit(x, y)

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




