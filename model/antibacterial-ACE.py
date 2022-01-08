# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 16:50:05 2021

@author: zhong
"""
#算法实践
#17
#(分割0.05，45)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics# 导入roc方法
from sklearn.neighbors import KNeighborsClassifier# 导入sklearn.neighbors模块中KNN类
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.model_selection  import cross_val_score
np.random.seed(0)# 设置随机种子，不设置的话默认是按系统时间作为参数，因此每次调用随机模块时产生的随机数都不一样设置后每次产生的一样
data = np.loadtxt(open("E:/学校文件/论文/paper/data/new_data/2-ACE-antibacterial_315.csv"),delimiter=",",skiprows=1)
data_p=np.loadtxt(open("E:/学校文件/论文/paper/data/新建文件夹/ACE_inhibitor_antibacterial_35.csv"),delimiter=",",skiprows=1)
data_x=data[:,0:2]
data_y=data[:,4]
data_x=preprocessing.StandardScaler().fit_transform(data_x)
data_xp=data_p[:,0:2]
data_yp=data_p[:,4]
data_xp=preprocessing.StandardScaler().fit_transform(data_xp)
print(data_x)
print(data_y)
print(data_yp)
print(data_xp)
train_X, test_X , train_y , test_y = train_test_split(data_x,data_y,test_size=0.2,random_state=33)


print("a: " + str(test_y))
print(len(test_X))
#knn kdtree
knn = KNeighborsClassifier(n_neighbors = 24,weights='distance',metric = 'chebyshev',algorithm="auto")# 定义一个knn分类器对象
knn.fit(train_X, train_y)# 调用该对象的训方法，主要接收两个参数：训练数据集及其样本标签
y_predict = knn.predict(data_xp)# 调用该对象的测试方法，主要接收一个参数：测试数据集
probility = knn.predict_proba(test_X)[:,1]# 计算各测试样本基于概率的预测
score = knn.score(test_X, test_y, sample_weight=None)# 调用该对象的打分方法，计算出准确率
score_p=knn.score(data_xp,data_yp, sample_weight=None)
print("预测： "+str(score_p))
#输出测试结果绘图
p_testxdata=knn.predict(test_X)


#绘制
fpr,tpr,thresholds=metrics.roc_curve(test_y,probility,pos_label=1)

#auc
AUC=sklearn.metrics.auc(fpr,tpr)
print("AUC: "+str(AUC))
print("准确率： "+str(score))

 
roc_auc = auc(fpr, tpr)
 
plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(AUC), lw=2)
 
plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
plt.title('ROC Curve')
plt.legend(loc="lower right")

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck') # 画对角线
plt.show()

conf_mat = confusion_matrix(test_y,p_testxdata)
print(conf_mat)
print(classification_report(test_y,p_testxdata))

# 寻找最好的K图
k_range = range(1, 31)
k_error = []
#循环，取k=1到k=31，查看误差效果
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k,weights='distance',metric = 'chebyshev',algorithm="auto")
    #cv参数决定数据集划分比例，这里是按照5:1划分训练集和测试集
    scores = cross_val_score(knn, data_x, data_y, cv=7, scoring='accuracy')
    k_error.append(1 - scores.mean())

#画图，x轴为k值，y值为误差值
plt.plot(k_range, k_error)
plt.xlabel('Value of K for KNN')
plt.ylabel('Error')

plt.show()

#绘制散点分布图
plt.scatter(data_x[316:630,1], y=data_x[316:630,2],c="b",alpha=0.2,cmap='viridis',label="antibacterial")
plt.scatter(data_x[0:315,1], y=data_x[0:315,2],c="r",alpha=0.2,cmap='viridis',label="ACE")
plt.xlabel("sp2")
plt.ylabel("numO")
plt.legend()
plt.title("ACE_inhibitor-antibacterial")
#绘制三维情况下的散点分布
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_x[316:630,2], ys=data_x[316:630,1],zs=data_x[316:630,0],c="b") # 绘制数据点
ax.scatter(data_x[0:315,2], ys=data_x[0:315,1],zs=data_x[0:315,0],c="r")

ax.set_zlabel('Z') # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
ax.view_init(elev=10, azim=180)
plt.show()