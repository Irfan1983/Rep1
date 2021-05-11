# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 11:09:10 2021

@author: Irfan
"""

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
Mall_Cus=pd.read_csv('Mall_Customers.csv')

Mall_Customer=Mall_Cus.drop(['CustomerID','Genre'],axis=1)

Mall_Customer.columns

Mall_Customer=Mall_Customer.rename(columns={'Annual Income (k$)':'Annual Income','Spending Score (1-100)':'Spending Score'})


#Hyper parameter 
A=pd.Series(0.0,index=range(1,7))
for i in range(1,7):
    Mall_Cluster=KMeans(random_state=1234,n_clusters=i).fit(Mall_Customer_New)
    A[i]=Mall_Cluster.inertia_
    

A.plot.line()


# 5 cluster
Mall_Customer_New=Mall_Customer.copy()

Mall_cluster5=KMeans(n_clusters=5,random_state=1234).fit(Mall_Customer_New)
Mall_Customer_New["Cluster"]=Mall_cluster5.labels_

Mall_Cus["Cluster"]=Mall_cluster5.labels_

Mall_Cus=Mall_Cus.drop('CustomerID',axis=1)
Mall_cus_profile=Mall_Cus.groupby('Cluster').agg(np.mean)

for i in Mall_cus_profile.columns:
   plt.figure()
   Mall_cus_profile[i].plot.bar()
   plt.title(i)
sns.pairplot(Mall_Cus, hue = "Cluster")
