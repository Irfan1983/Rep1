

import mglearn 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.metrics import mean_squared_error as mse
data = pd.read_csv('Salary_Prediction.csv')
data.shape

x = data.drop(['Salary'], axis=1)
y = data.drop(['Exp'], axis=1)


knn=KNN(n_neighbors=5).fit(x,y)
knn_pr=knn.predict(x)
y1=pd.Series(1,index=[12])
knn.predict(y1)