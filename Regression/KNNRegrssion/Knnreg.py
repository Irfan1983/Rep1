# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 14:40:00 2021

@author: Irfan
"""
import mglearn 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.metrics import mean_squared_error as mse
data = pd.read_csv('train_cleaned.csv')
data.shape

x = data.drop(['Item_Outlet_Sales'], axis=1)
y = data['Item_Outlet_Sales']
x.shape, y.shape

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

x = pd.DataFrame(x_scaled)


train_x,test_x,train_y,test_y = train_test_split(x,y, random_state = 56)
ms=[]
for i in range(1,13):
    reg = KNN(n_neighbors = i)
    reg.fit(train_x, train_y)
    test_predict = reg.predict(test_x)
    ms.append(mse(test_predict, test_y))
    
plt.plot(range(1,13),ms)


KNNmodel=KNN(n_neighbors=9).fit(x,y)

KNN_Predict=KNNmodel.predict(test_x)
mse(KNN_Predict, test_y)

