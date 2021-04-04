import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf



class Pipeline:
    
    ''' When we call the FeaturePreprocessor for the first time
    we initialise it with the data set we use to train the model,
    plus the different groups of variables to which we wish to apply
    the different engineering procedures'''
    
    
    def __init__(self, target,features, test_size = 0.1, random_state = 0,percentage = 0.01):
        
        
        # data sets
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.target="Exp"
        self.mtcars_ols=None
        # more parameters
        self.test_size = test_size
        self.random_state = random_state

    

    # ====   master function that orchestrates feature engineering =====

    def split_data(self, data):
        '''pipeline to learn parameters from data, fit the scaler and lasso'''
        
        # setarate data sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                data, data[self.target],
                test_size = self.test_size,
                random_state = self.random_state)
        return self
    
    def evaluate_model(self,data):
        data.plot.scatter("Exp","Salary")
    
    def fit(self,data1):
        mtcars_ols = smf.ols("Exp ~ Salary", data = data1).fit()
        return mtcars_ols
       
