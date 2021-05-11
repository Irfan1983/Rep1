# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 17:57:15 2021

@author: Irfan
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import scale

df=pd.read_csv('Mall_Customers.csv')
df.columns.values

df=df.iloc[:,-4:5]

df1=pd.get_dummies(df)



clus1=[]
for i in range(1,7):
    clus=KMeans(n_clusters=i,random_state=1234).fit(df1)
    clus1.append(clus.inertia_)

    
clusd=pd.Series(clus1,range(1,7))
#
clusd.plot.line()
#
#
clus_dist=df1.copy()
K_Dist=KMeans(n_clusters=5,random_state=1234).fit(df1)
clus_dist['cluster']=K_Dist.labels_

Male_cus=clus_dist[clus_dist['Gender_Male']==1]
Male_cus=Male_cus.drop('Gender_Female',1)

Male_profile=Male_cus.groupby("cluster").agg(np.mean)


for i in Male_cus.columns:
    plt.figure()
    Male_profile[i].plot.bar()
    plt.title(i)
    


sns.pairplot(Male_cus,hue='cluster')