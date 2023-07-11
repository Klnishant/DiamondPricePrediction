from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys,os
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

## Data transformation config

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file=os.path.join('artifacts','preprocessor.pkl')

## data transformation

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data transformation initiated')

            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']
            
            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            ## Categorical pipeline
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scaler',StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
            ])

            return preprocessor
        
        except Exception as e:
            logging.info('Error in data transformation')
            raise CustomException(e,sys)
        
    def initiate_data_transforming(self,train_path,test_path):
        try:
            ## REading train and test data
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Read test and train data is completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            preprocessing_obj=self.get_data_transformation_object()
            logging.info('Obtainend preprocessing object')

            target_col='price'
            drop_col=[target_col,'id']

            ## segregate features in dependent and independent features

            input_feature_train_df=train_df.drop(columns=drop_col,axis=1)
            target_feature_train_df=train_df[target_col]


            input_feature_test_df=test_df.drop(columns=drop_col,axis=1)
            target_feature_test_df=test_df[target_col]

            logging.info('Apply the transformation')

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info('Applying preprocessing')
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file,
                obj=preprocessing_obj
            )
            logging.info('Preprocessor pickle file in and save')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file
            )
        except Exception as e:
            logging.info('Exception occured in initiating data transformation')
            raise CustomException(e,sys)