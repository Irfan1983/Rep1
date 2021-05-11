# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 16:16:04 2021

@author: Irfan
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.preprocessing import scale


wine=pd.read_csv("wine.csv")

wine.columns = ["Wine_Class", "Alcohol","Malic_acid","Ash","Alcalinity_of_ash",
                       "Magnesium","Total_phenols","Flavanoids","Nonflavanoid_phenols",
                       "Proanthocyanins","Color_intensity","Hue","OD280_OD315",
                       "Proline"]
newwine = wine.iloc[:,1:]
newwine.info()
newwine.describe()
newwine.isnull().sum()
newwine.std()/newwine.mean()
wine_Scaled=scale(newwine)
wine_Scaled=pd.DataFrame(wine_Scaled,columns=newwine.columns)

wine_dis=[]
for i in range(1,7):
    newwine_clus=KMeans(n_clusters=i,random_state=1234).fit(wine_Scaled)
    wine_dis.append(newwine_clus.inertia_)

wine_dist_elbow=pd.Series(wine_dis,index=range(1,7))

wine_dist_elbow.plot.line()
plt.grid()
plt.plot(range(1,20),wine_dist_elbow)

wine_cluster_3=newwine.copy()
wine_cls=KMeans(n_clusters=3,random_state=1234).fit(wine_Scaled)
wine_cluster_3["cluster"]=wine_cls.labels_

wine_prof=wine_cluster_3.groupby('cluster').agg(np.mean)


for i in wine_prof.columns:
    plt.figure()
    wine_prof[i].plot.bar()
    plt.title(i)

sns.pairplot(wine_cluster_3,hue='cluster')


# Cluster 0: High Proline, Flavanoids, Phenols, OD280; Less alcalanity of ash
# Cluster 1: High Color Intensity, Non flavanoid phenols, Malic acid; Less OD280, Hue, Pronathocyanins, Flavanoids
# Cluster 2: Less Color intensity, Alcohol


