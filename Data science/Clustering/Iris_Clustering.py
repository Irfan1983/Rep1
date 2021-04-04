import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
import seaborn as sns



irisdata = pd.read_csv("iris.csv")
newiris = irisdata.drop('Species',axis=1)

## Step 1: Exploratory Analysis
#sns.pairplot(newiris)
#newiris.columns
#newiris.plot.scatter("Petal.Length","Petal.Width")

#newiris_clus2=newiris.copy()
#newiris_clu=KMeans(n_clusters=2,random_state=1234).fit(newiris)
#newiris_clus2["Cluster"]=newiris_clu.labels_
k=[]
for i in range(1,7):
    newiris_clu=KMeans(n_clusters=i,random_state=1234).fit(newiris)
    k.append(newiris_clu.inertia_)
irisser=pd.Series(k,index=range(1,7))


irisser.plot.line()

# curve happens on 2 ,so we can divide to 2 cluster

newiris_clus2=newiris.copy()
newiris_clu=KMeans(n_clusters=2,random_state=1234).fit(newiris)
newiris_clus2["Cluster"]=newiris_clu.labels_

newiris_clus2_profile=newiris_clus2.groupby("Cluster").agg(np.mean)


for i in newiris_clus2_profile.columns:
    plt.figure()
    newiris_clus2_profile[i].plot.bar()
    plt.title(i)

sns.pairplot(newiris_clus2, hue = "Cluster")