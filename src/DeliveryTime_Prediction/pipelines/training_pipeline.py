import os
import sys
import pandas as pd
import numpy as np
from src.DeliveryTime_Prediction.exception import customexception
from src.DeliveryTime_Prediction.logger import logging

from src.DeliveryTime_Prediction.components.data_ingestion import DataIngestion
from src.DeliveryTime_Prediction.components.data_transformation import DataTransformation
from src.DeliveryTime_Prediction.components.model_trainer import ModelTrainer

class TraningPipline:
    def start_data_ingestion(self):
        try:
            dataingestion=DataIngestion()
            train_data_path,test_data_path=dataingestion.initate_data_ingestion()
            return train_data_path,test_data_path
        
        except customexception as e:
            logging.info("error in data ingestion", e)
            raise customexception(e,sys)
    
    def start_data_transformation(self,train_data_path,test_data_path):
        try:
            datatransformation=DataTransformation()
            train_arr,test_arr=datatransformation.initiate_data_transformation(train_data_path,test_data_path)
            return train_arr,test_arr
        except customexception as e:
            logging.info("error in data transformation", e)
            raise customexception(e,sys)
        
    def start_traning(self):
        try:
            logging.info("Starting Training Pipeline")
            train_data_path,test_data_path=self.start_data_ingestion()
            train_arr,test_arr=self.start_data_transformation(train_data_path,test_data_path)
        except customexception as e:
            logging.info("error in training pipeline", e)
            raise customexception(e,sys)
        
trainer=TraningPipline()
trainer.start_traning()
print("Training Pipeline run successfully")