# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 22:25:50 2021

@author: zhong
"""

from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd

data=pd.read_csv("E:/学校文件/论文/biopep数据格式改.csv",header=0)

def change(shortone):
    hchange_one=Chem.MolToSmiles(Chem.MolFromFASTA(shortone))
    return hchange_one
    
    
a=change("R")
print(a)




