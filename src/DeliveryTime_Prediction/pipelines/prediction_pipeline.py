import os
import sys
import pandas as pd
import numpy as np
from src.DeliveryTime_Prediction.exception import customexception
from src.DeliveryTime_Prediction.logger import logging

class PredictionPipline:
    def start_prediction(self):
        try:
            logging.info("Starting prediction Pipeline")
        except customexception as e:
            logging.info("error in prediction pipeline", e)
            raise customexception(e,sys)
