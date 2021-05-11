# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 22:04:44 2021

@author: Irfan
"""

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns

wine=pd.read_csv('redwine.csv')
hyper=pd.Series(0.0,index=range(1,7))
for i in range(1,7):
    hyper_tuning=KMeans(random_state=1234,n_clusters=i).fit(wine)
    hyper[i]=hyper_tuning.inertia_
    
hyper.plot.line()

wine_new=wine.copy()
wine_Cluster2=KMeans(random_state=1234,n_clusters=2).fit(wine)
wine_new['Cluster']=wine_Cluster2.labels_

Wine_profile=wine_new.groupby('Cluster').agg(np.mean)

#for i in Wine_profile.columns:
#    plt.figure()
#    plt.title(i)
#    Wine_profile[i].plot.bar()
#    plt.show()

sns.pairplot(wine_new, hue = "Cluster")

clus1=wine_new[wine_new.Cluster==0]
clus2=wine_new[wine_new.Cluster==2]