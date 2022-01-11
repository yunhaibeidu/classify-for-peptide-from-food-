# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 21:05:01 2021

@author: zhong
"""
#随机数据生成
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import csv
#打开文件格式转换
tmp_lst = []
with open('E:/学校文件/论文/paper/data/辅助表三antioxidative.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        tmp_lst.append(row)
df = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0]) 
print(df)
#输出数据集

out=df.sample(n=35,random_state=123,axis=0)
print(len(out))
print(out)
k=out.values
a=k[:,3:7]
print(a)


#转换格式写入 
output = open('35_ACE.xls','w',encoding='gbk')
output.write('sp2\tsp3\tnumO\tnumN\t\n')
for i in range(len(a)):
	for j in range(len(a[i])):
		output.write(str(a[i][j]))    #write函数不能写int类型的参数，所以使用str()转化
		output.write('\t')   #相当于Tab一下，换一个单元格
	output.write('\n')       #写完一行立马换行
output.close()