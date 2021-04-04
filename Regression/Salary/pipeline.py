import pandas as pd
import numpy as np

from preprocessors import Pipeline
import config



pipeline = Pipeline(target =  config.TARGET,features = config.FEATURES)




if __name__ == '__main__':
    
    # load data set
    print(config.PATH_TO_DATASET)
    data = pd.read_csv(config.PATH_TO_DATASET)
    pipeline.split_data(data)
    print('model performance')
    pipeline.evaluate_model(data)
#    print()
#    print('Some predictions:')
    yy=pipeline.fit(data)
    
    aa=yy.predict([8])
    print(aa)