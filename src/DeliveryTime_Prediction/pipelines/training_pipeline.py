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
    def start_traning(self):
        try:
            logging.info("Starting Training Pipeline")
        except customexception as e:
            logging.info("error in training pipeline", e)
            raise customexception(e,sys)