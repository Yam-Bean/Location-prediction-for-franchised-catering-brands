# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 18:35:37 2023

@author: HP
"""

import pandas as pd
from geopy.distance import geodesic
import numpy as np

df = pd.read_csv('C:/Users/HP/Desktop/newfeature.csv')
df2 = pd.read_csv('C:/Users/HP/Desktop/fullfeature.csv')

n = df.shape[0]
n2 = df2.shape[0]

for i in range(n):
    temp =np.array([0, 0, 0])
    for j in range(n):
        distance = geodesic((df.iat[i, 2], df.iat[i, 3]), (df.iat[j, 2], df.iat[j, 3])).km
        if distance < 2:
            temp = temp + np.array([1, distance, df.iat[j, 1]])
        df.iat[i, 5] = temp[0]  #包含自己的店铺数量
        df.iat[i, 6] = temp[1]/(temp[0]-1)  #不含自己的汉堡店平均距离
        df.iat[i, 4] = temp[2]    #包含自己的店铺销量

for i in range(n):
    temp = np.array([0, 0])
    for j in range(n2):
        distance = geodesic((df.iat[i, 2], df.iat[i, 3]), (df2.iat[j, 2], df2.iat[j, 3])).km
        if distance < 2:
            temp = temp + np.array([df2.iat[j, 1], 1])
        df.iat[i, 7] = temp[0]  #包含自己的店铺销量
        df.iat[i, 8] = temp[1]  #包含自己的店铺数量
            
outputpath2 = 'C:/Users/HP/Desktop/feature1234.csv'
df.to_csv(outputpath2, sep = ',', index = 'FALSE', header = 'TRUE', encoding = 'UTF-8')