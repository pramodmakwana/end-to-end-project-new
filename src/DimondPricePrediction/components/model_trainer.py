
import pandas as pd
import numpy as np
import os
import sys
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception
from dataclasses import dataclass
from src.DimondPricePrediction.utils.utils import save_object
from src.DimondPricePrediction.utils.utils import evaluate_model

from skelearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet

@dataclass
class ModelTrainerConfig:
    trained_model_file_path =os.path.join('artifacts','model.pkl')


class  ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_training(self):
        try:
            logging.info('Splitting Dependent and Indepedent variables from train and test data')
            X_train, y_train, X_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models=(
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'Elastinet':ElasticNet()
                
            
            )
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n==============================================================================================\n')
            loggging.info(f'Model Report : {model_report}')
            
            # To get best model score from dictionary
            best_model_score= max(sorted(model_repor.values()))
            
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model=models[best_model_name]
            
            print(f'Best Model Found,Model Name: {Best_Model_name},R2 Score :{best_model_score}')
            print('\n=================================================================================================\n')
            logging.info(f'Best Model Found, Model Name : {best_model_name},R2 score : {best_model_score}')
            
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            
        except Exception  as e:
            logging.info('Exception occured at Model Training')
            raise customexception(e,sys)
        