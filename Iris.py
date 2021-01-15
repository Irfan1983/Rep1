#clustering


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
 ## Step 0: Business understanding, Data cleaning, Preprocessing
#import the data
iris=pd.read_csv("iris.csv")
iris.info()
iris.describe()
iris.isnull().sum()
iris.std()/iris.mean()  # P.Length and P.Width having good variance 
#Exploratory 
iris.columns.values
Iris_Data=iris.loc[::,['Sepal.Length','Sepal.Width','Petal.Length', 'Petal.Width']]
sns.pairplot(Iris_Data)

#scale is not required

#hyper parameter tuning
iris_dist=[]
for i in range(1,6):
    iris_clus=KMeans(n_clusters=i,random_state=1234).fit(Iris_Data)
    iris_dist.append(iris_clus.inertia_)
    
iris_Series=pd.Series(iris_dist,index=range(1,6))    

iris_Series.plot.line()

#elbow curve is 2 ,it can be clustered to 2
Iris_with_cluster=Iris_Data.copy()
Iris_Cluster=KMeans(n_clusters=2,random_state=1234).fit(Iris_Data)
Iris_with_cluster["cluster"]=Iris_Cluster.labels_


Iris_dis=Iris_with_cluster.groupby("cluster").agg(np.mean)

for i in Iris_dis.columns:
    plt.Figure()
    plt.title(i)
    Iris_dis[i].plot.bar()
    plt.show()





