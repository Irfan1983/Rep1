# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 17:44:05 2021

@author: Irfan
"""

import pandas as pd
import pickle

new_live_data = pd.DataFrame({"Sepal.Length":[5.8,5.2,3.6,2.2],
                              "Sepal.Width":[2.5,3,1.8,0.6],
                              "Petal.Length":[4.5, 3.8, 4.2,0.8],
                              "Petal.Width":[1.8, 1.4, 1.6, 0.1]})
    
model_file_location = "irisknn6.sav"
model_file_handler = open(model_file_location,"rb")
model_loaded = pickle.load(model_file_handler)
aa=model_loaded.predict(new_live_data)
print(aa)
