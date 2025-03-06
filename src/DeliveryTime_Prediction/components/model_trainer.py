import os
import sys
import pandas as pd
import numpy as np
from src.DeliveryTime_Prediction.exception import customexception
from src.DeliveryTime_Prediction.logger import logging

class ModelTrainer:
    def initate_model_traning(self):
        try:
            logging.info("Starting Training Pipeline")
        except customexception as e:
            logging.info("error in training pipeline", e)
            raise customexception(e,sys)