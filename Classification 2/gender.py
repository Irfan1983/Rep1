import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

label_encoder = preprocessing.LabelEncoder()
df=pd.read_csv("Transformed Data Set - Sheet1.csv")
label_enc=label_encoder.fit(df['Gender'])
df['Gender']=label_encoder.transform(df['Gender'])

X=df.iloc[:,:4]
Y=df.Gender
X=pd.get_dummies(X)
train_X,test_X,train_Y,test_Y=train_test_split(X,Y,test_size=0.1,random_state=1234)
Logi_Model=LogisticRegression().fit(X,Y)
Train_Pred=Logi_Model.predict(train_X)
pd.crosstab(Train_Pred,train_Y)
accuracy_score(Train_Pred,train_Y)
Test_Pred=Logi_Model.predict(test_X)
pd.crosstab(Test_Pred,test_Y)
accuracy_score(Test_Pred,test_Y)


Model_location="Gender.sav"
Model_Handler=open(Model_location,"wb")
pickle.dump(Logi_Model,Model_Handler,)


model_file_handler = open("Gender.sav","rb")
model_loaded = pickle.load(model_file_handler)
label_encoder.inverse_transform(model_loaded.predict(train_X))

