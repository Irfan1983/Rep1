import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
import pickle
Iris_Data=pd.read_csv('iris.csv')
Iris_Data.isnull().sum()

X=Iris_Data.iloc[:,:4]
Y=Iris_Data.Species

Train_X,Train_Y,Test_X,Test_Y=train_test_split(X,Y,random_state=1234)

KNeigh_Model=KNeighborsClassifier(n_neighbors=6).fit(X,Y)

#Evaluating Train and Test Data

Iris_Pre_Train=KNeigh_Model.predict(Train_X)
accuracy_score(Iris_Pre_Train,Test_X)  # 0.9642857142857143 Training
pd.crosstab(Iris_Pre_Train,Test_X)
Iris_Pre_Test=KNeigh_Model.predict(Train_Y)
accuracy_score(Iris_Pre_Test,Test_Y)  # 100
pd.crosstab(Iris_Pre_Test,Test_Y)     

# Building model deployment


Model_FileLocation="Iris_KNN.sav"
Model_FileHandler=open(Model_FileLocation,"wb")
pickle.dump(KNeigh_Model,Model_FileHandler)


