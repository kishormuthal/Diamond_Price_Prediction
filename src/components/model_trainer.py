# Basic Import
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'Elasticnet':ElasticNet(),
                'DecisiscionTreeRegressor':DecisionTreeRegressor(),
                'RandomForestRegressor': RandomForestRegressor(),
                'KNeighborsRegressor': KNeighborsRegressor()
        }
            cv_results_rms = {}
            for model_name, model in models.items():

                logging.info(f"Performing cross-validation for model: {model_name}")
                cv_score = cross_val_score(
                    model, X_train, y_train, scoring="neg_root_mean_squared_error", cv=10
                )
                mean_rmse = -cv_score.mean()  # Convert to positive RMSE
                cv_results_rms[model_name] = mean_rmse
                print(f"{model_name}: Mean RMSE = {mean_rmse:.4f}")
                logging.info(f"{model_name}: Mean RMSE = {mean_rmse:.4f}")
        
            # Select the best model based on RMSE
            best_model_name = min(cv_results_rms, key=cv_results_rms.get)
            best_model = models[best_model_name]
            logging.info(f"Best Model based on CV: {best_model_name} with RMSE = {cv_results_rms[best_model_name]:.4f}")

            # Evaluate the best model on test data
            model_report = evaluate_model(X_train, y_train, X_test, y_test, best_model)
            
            # Print the dictionary to the console
            print("Model Evaluation Report:")
            for key, value in model_report.items():
                print(f"{key}: {value:.4f}")
        
            # Log the metrics
            logging.info("Model Evaluation Report:")
            for key, value in model_report.items():
                logging.info(f"{key}: {value:.4f}")
            
            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Model saved at: {self.model_trainer_config.trained_model_file_path}")
        
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)