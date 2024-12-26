import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn import metrics
from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, model):
    try:
        report = {}  # Dictionary to store evaluation metrics
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict testing data
        y_test_pred = model.predict(X_test)
        
        # Calculate evaluation metrics
        r2 = metrics.r2_score(y_test, y_test_pred)
        adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
        mae = metrics.mean_absolute_error(y_test, y_test_pred)
        mse = metrics.mean_squared_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)
        
        # Store metrics in the report dictionary
        report['R2'] = r2
        report['Adjusted R2'] = adjusted_r2
        report['MAE'] = mae
        report['MSE'] = mse
        report['RMSE'] = rmse       
        # Return the report dictionary
        return report
    
    except Exception as e:
        logging.error('Exception occurred during model evaluation', exc_info=True)
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)