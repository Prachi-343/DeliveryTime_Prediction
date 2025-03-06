import os
import sys
import pandas as pd
import numpy as np
from src.DeliveryTime_Prediction.exception import customexception
from src.DeliveryTime_Prediction.logger import logging

class DataIngestion:
    def initate_data_ingestion(self):
        try:
            logging.info("Starting Data Ingestion")
        except customexception as e:
            logging.info("error in data ingestion", e)
            raise customexception(e,sys)