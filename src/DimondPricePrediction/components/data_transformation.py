

import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.DimondPricePrediction.exception import customexception
from src.DimondPricePrediction.logger import logging


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler


from src.DimondPricePrediction.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransfromation:
    def __init__(self):
        self.get_data_transformation_config=DataTransformationConfig()
    
    def get_data_transformation(self):
        
        try:
            logging.info("Exception occured in the initiate_datatransformation")t
            # Define which columns should be ordinalencoded and which should be sclaed
            
            categorical_cols=['cut','color','clarity']
            numerical_cols=['carat','depath','table','x','y','z']
            
            # Define the custom ranking for each ordinal variable
            cut_categories=['fair','Good','Very Good','Pemium','Ideal']
            color_categories=['D','E','F','G','H','I','J']
            clarity_categories=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            
            logging.info('Pipeline Initiated')
            
            ## Numerical pipeline
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
                
                
            )
            
            # Categorical Pipeline
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories]))
                    ('scaler',StandardScaler())     
                ]
                
            )
            
            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols)
                ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            
            return preprocessing
            
            
        
            raise customexception(e,sys)
            except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation ")
            
    def initialize_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("read train and test data complete")
            logging.info(f'Train Dataframe Head: \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n {train_df.head().to_string()}')
            
            preprocessing_obj = self.get_data_transformation()
            
            target_column_name='price'
            drop_columns=[target_column_name,'id']
            
            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            logging.info("Applying preprocessing object on training and testing datasets")
            
            save_object(
               file_path= self.data_transformation_config.preprocessor_obj_file_path,
               obj=preprocessing_obj
            )
            
    
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            
            raise customexception(e,sys)