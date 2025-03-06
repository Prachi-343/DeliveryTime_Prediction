import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path 
from src.DeliveryTime_Prediction.exception import customexception
from src.DeliveryTime_Prediction.logger import logging
from sklearn.model_selection import train_test_split

class DataIngestionConfig:
    raw_data_path:str=os.path.join("artifacts","raw_data.csv")
    train_data_path:str=os.path.join("artifacts","train_data.csv")
    test_data_path:str=os.path.join("artifacts","test_data.csv")
        
class DataIngestion:
    def __init__(self):
        self.config=DataIngestionConfig()
    def initate_data_ingestion(self):
        try:
            logging.info("Starting Data Ingestion")
            data=pd.read_csv(Path(os.path.join("notebooks/data","Food_Delivery_Times.csv")))
            logging.info("data read successfully")
            
            os.makedirs(os.path.dirname(os.path.join(self.config.raw_data_path)),exist_ok=True)
            data.to_csv(self.config.raw_data_path,index=False)
            logging.info("raw data saved in artifacts directory")
            
            train,test=train_test_split(data,test_size=0.2,random_state=42)
            logging.info("train test split done")
            
            train.to_csv(self.config.train_data_path,index=False)
            test.to_csv(self.config.test_data_path,index=False)
            logging.info("train and test data saved in artifacts directory")
            logging.info("Data Ingestion Completed")
            
            return (self.config.train_data_path,
                    self.config.test_data_path)
        
        except customexception as e:
            logging.info("error in data ingestion", e)
            raise customexception(e,sys)

