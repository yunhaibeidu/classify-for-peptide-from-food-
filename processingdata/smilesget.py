# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 23:00:27 2021

@author: zhong
"""

from rdkit import Chem
import pandas as pd
import xlrd
import xlwt
import pprint
import numpy as np
#读取数据
data=pd.read_csv("E:/学校文件/论文/paper/data/biopep-20200925-clean2.csv",header=0,encoding="ISO-8859-1")
yuan=data["sequence"]
#设置空集合
result=[]
for i in range(0,4015,1):
  hchange_one=Chem.MolToSmiles(Chem.MolFromFASTA(yuan[i]))
  result.append(hchange_one)
print(len(result))
yuan1=list(yuan)
result=list(result)
w=list(zip(yuan1,result))
print(w)


#写入数据
#转换格式
output = open('datasmlies.xls','w',encoding='gbk')
output.write('pepseq\t smiles\t \n')
for i in range(len(w)):
	for j in range(len(w[i])):
		output.write(str(w[i][j]))    #write函数不能写int类型的参数，所以使用str()转化
		output.write('\t')   #相当于Tab一下，换一个单元格
	output.write('\n')       #写完一行立马换行
output.close()