

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

iris=pd.read_csv("iris.csv")

iris.describe()

X=iris.iloc[:,:4]
Y=iris.Species

train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.1,random_state=1234)
KNeign_Iris=KNeighborsClassifier(n_neighbors=2).fit(X,Y)

Model_KN=KNeign_Iris.predict(train_x)
pd.crosstab(Model_KN,train_y)
accuracy_score(Model_KN,train_y) # 0.9777777777777777

KNN_Acc=pd.Series(0.0,range(1,10))

for i in range(1,10):
    KNN_Acc[i]=np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=i),X,Y,cv = 5))

# k=6 accuracy 98


Decis=DecisionTreeClassifier(criterion="gini",max_depth=3,min_samples_split=10).fit(X,Y)
iris_dtree_feature_imp = pd.Series(Decis.feature_importances_,index = X.columns)


crit = ["gini","entropy"]
max_depth = [1,2,3,4,5,6]
min_samp_split = [2,5,10,20]
cavv=[]
for c in crit:
    for maxi in max_depth:
        for min_sam in min_samp_split:
            cavv.append([c,maxi,min_sam,np.mean(cross_val_score(DecisionTreeClassifier(criterion=c,max_depth=maxi,min_samples_split=min_sam),X,Y,cv=5))])
            
Desc_Hy=pd.DataFrame(cavv,columns=["criterion","Max_Depth","Min_sample","Accuracy"])            

Desc_Hy.sort_values("Accuracy")
# 97 accuracy max


ntrees = [1,3,5,7,10]
crit = ["gini","entropy"]
max_depth = [1,2,3,4,5,6]
min_samp_split = [2,5,10,20]

rand=[]
for n in ntrees:
    for c in crit:
        for ma in max_depth:
            for min_s in min_samp_split:
               rand.append([n,c,ma,min_s,np.mean(cross_val_score(RandomForestClassifier(criterion=c,n_estimators=n,max_depth=ma,min_samples_split=min_s),X,Y,cv=5))])
                

rand_df=pd.DataFrame(rand,columns=["Ntrees","Crit","Max_depth","min_samp_split","Accuracy"])

rand_df.sort_values("Accuracy")

#5     gini          6               2  0.973333

#gradint boosting

ntrees = [10,20,50,70,100,150]
le_rate = [0.1,0.2,0.3]
max_depth = [1,2,3,4,5,6]
min_samp_split = [2,5,10,20]
gr=[]
for n in ntrees:
    for l in le_rate:
        for m in max_depth:
            for mi in min_samp_split:
                gr.append(np.mean(cross_val_score(GradientBoostingClassifier(n_estimators=n,learning_rate=l,max_depth=m,min_samples_split=mi,random_state=1234),X,Y,cv=5)))

#96%
                
#above all KNeighborsClassifier give best accuracy

Final_Kn=KNeighborsClassifier(n_neighbors=6).fit(X,Y)
KNN_Prd=Final_Kn.predict(train_x)

pd.crosstab(KNN_Prd,train_y)

accuracy_score(KNN_Prd,train_y)# training data accuracy 96

KNN_Prd1=Final_Kn.predict(test_x)
accuracy_score(test_y,KNN_Prd1)

model_file_location = "irisknn6.sav"
model_file_handler = open(model_file_location,"wb")
pickle.dump(Final_Kn,model_file_handler)
new_live_data = pd.DataFrame({"Sepal.Length":[5.8,5.2,3.6,2.2],
                              "Sepal.Width":[2.5,3,1.8,0.6],
                              "Petal.Length":[4.5, 3.8, 4.2,0.8],
                              "Petal.Width":[1.8, 1.4, 1.6, 0.1]})
    
model_file_location = "irisknn6.sav"
model_file_handler = open(model_file_location,"rb")
model_loaded = pickle.load(model_file_handler)
model_loaded.predict(new_live_data)
