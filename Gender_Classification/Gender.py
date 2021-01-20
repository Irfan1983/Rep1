# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 17:20:42 2021

@author: Irfan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
df=pd.read_csv("Transformed Data Set - Sheet1.csv")

df.columns.values

df["Favorite Music Genre"].unique()



en=LabelEncoder()
df['Gender']=en.fit_transform(df['Gender'])
df=pd.get_dummies(df)
X=df.drop(columns=['Gender'])
Y=df.Gender

Train_x,Test_x,Train_y,Test_y=train_test_split(X,Y,random_state=1234)

#Kneigh


Knn=KNeighborsClassifier(n_neighbors=3).fit(X,Y)
knn_pred=Knn.predict(Test_x)
accuracy_score(Test_y,knn_pred)
pd.crosstab(Test_y,knn_pred)   # 71 %


# decison tree
Dtc=DecisionTreeClassifier(criterion="gini",max_depth=3,min_samples_split=10).fit(X,Y)
Dtc_Prd=Dtc.predict(Train_x)
accuracy_score(Train_y,Dtc_Prd)
Crit=["gini","entropy"]
max_spl=[1,2,3,4,5,6]
min_sample=[2,5,10,20]
cvacc=[]
for c in Crit:
    for m in max_spl:
        for mi in min_sample:
           cvacc.append([c,m,mi,np.mean(cross_val_score(DecisionTreeClassifier(criterion=c,max_depth=m,min_samples_split=mi),X,Y,cv=5))])
           
Desc_accu=pd.DataFrame(cvacc,columns=["c","m","mi","Acc"])


#Random_Forest
n_estimato=[1,3,5,7,10]
Ran_Crit=["gini","entropy"]
Ran_max_spl=[1,2,3,4,5,6]
Ran_min_sample=[2,5,10,20]
Ran_cvacc=[]

for n in n_estimato:
    for rc in Ran_Crit:
        for ra in Ran_max_spl:
            for rm in Ran_min_sample:
                Ran_cvacc.append(np.mean(cross_val_score(RandomForestClassifier(n_estimators=n,criterion=rc,max_depth=ra,min_samples_split=rm),X,Y,cv=5)))


                
iris_cvacc_gbm = []
# Hyper parameter tuning
ntrees = [10,20,50,70,100,150]
le_rate = [0.1,0.2,0.3]
max_depth = [1,2,3,4,5,6]
min_samp_split = [2,5,10,20]
for ntr in ntrees:
    for lr in le_rate:
        for md in max_depth:
            for ms in min_samp_split:
                cvacc = np.mean(cross_val_score(
                        GradientBoostingClassifier(n_estimators=ntr,
                                learning_rate = lr, max_depth = md, 
                                min_samples_split=ms, random_state=1234),
                            X, #IDV
                            Y, #DV
                            cv = 5))
                iris_cvacc_gbm.append([ntr,lr,md,ms,cvacc])
iris_cvacc_gbm = pd.DataFrame(iris_cvacc_gbm,
                                columns = ["No of Trees","Learning Rate","Max Depth","Minimum Sample Split","CV Accuracy"])

                
                
         
##

logi=LogisticRegression(random_state=1234).fit(X,Y)

Log_prd=logi.predict(Train_x)

pd.crosstab(Log_prd,Train_y)

accuracy_score(Log_prd,Train_y)
#accuarcy score 71
    

Gas=GaussianNB().fit(X,Y)

Gas_Prd=Gas.predict(Train_x)


pd.crosstab(Gas_Prd,Train_y)
accuracy_score(Gas_Prd,Train_y)
