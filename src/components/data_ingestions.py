import sys
import os
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
import pandas as pd
from dataclasses import dataclass

## initialize the dataIngestion configuration

@dataclass
class DataIngestionconfig():
    train_data_path=os.path.join('artifacts','train_data.csv')
    test_data_path=os.path.join('artifacts','test_data.csv')
    raw_data_path=os.path.join('artifacts','raw_data.csv')

## create dataIngestion class

class DataIngestion():
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()
    
    def initiate_data_ingestion(self):
        logging.info('Data Ingestiation Starts')

        try:
            df=pd.read_csv(os.path.join('notebook/data','gemstone.csv'))
            logging.info('Datasets read as pandas dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False)


            logging.info('Train Test Split')

            train_set,test_set=train_test_split(df,test_size=0.3,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of data is completed')

            return(self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)

        except Exception as e:
            logging.info('Error Occured In Data Ingestion')