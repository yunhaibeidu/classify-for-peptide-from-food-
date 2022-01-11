# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 16:49:40 2021

@author: zhong
"""

import rdkit
from rdkit import Chem
from rdkit.Chem import rdchem
import pandas as pd
import xlrd
import xlwt
import numpy as np

#读取数据
data=pd.read_csv("E:/学校文件/2020大创/c_抗肿瘤和抗氧化.csv",header=0)
yuan=data["smiles"]

#设置空集合
result_sp2=[]
result_sp3=[]
result_N=[]
result_O=[]
#循环提取第特征值SP2,SP3,O,N
for i in range(0,106,1):
    mol= Chem.MolFromSmiles(yuan[i])
    #设置空集合序列
    index = 0 
    m=[]
    k=[]
    a_sp2=[]
    a_sp3=[]
    a_N=[]
    a_O=[]
    #遍历循环
    while index <= mol.GetNumAtoms()-1:
      atomObj = mol.GetAtomWithIdx(index) 
      symbol = str(atomObj.GetSymbol())
      Hybridization = str(atomObj.GetHybridization())
      m.append(str(symbol))
      k.append(str(Hybridization))
      index+=1
      #列表计数的方法得到
      count_sp2=k.count("SP2")
      a_sp2.append(count_sp2)
      count_sp3=k.count("SP3")
      a_sp3.append(count_sp3)
      count_O=m.count("O")
      a_O.append(count_O)
      count_N=m.count("N")
      a_N.append(count_N)
    
    #结果提取
    result_sp2.append(a_sp2[-1])
    result_sp3.append(a_sp3[-1])
    result_O.append(a_O[-1])
    result_N.append(a_N[-1])
#展示结果
print("SP2: " + str(result_sp2))
print("SP3: " + str(result_sp3))
print("O: " + str(result_O))
print("N: " + str(result_N))
#打包
yuan1=list(yuan)
result_sp2=list(result_sp2)
result_sp3=list(result_sp3)
result_O=list(result_O)
result_N=list(result_N)
w=list(zip(yuan1,result_sp2,result_sp3,result_O,result_N))
print(w)
#写入结果到表格
output = open('c_抗肿瘤和抗氧化_特征.xls','w',encoding='gbk')
output.write('smiles\t sp2\t sp3\t numO\t numN \n')
for i in range(len(w)):
	for j in range(len(w[i])):
		output.write(str(w[i][j]))    #write函数不能写int类型的参数，所以使用str()转化
		output.write('\t')   #相当于Tab一下，换一个单元格
	output.write('\n')       #写完一行立马换行
output.close()