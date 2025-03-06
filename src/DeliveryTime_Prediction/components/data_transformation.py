import os
import sys
import pandas as pd
import numpy as np
from src.DeliveryTime_Prediction.exception import customexception
from src.DeliveryTime_Prediction.logger import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.impute import SimpleImputer
from src.DeliveryTime_Prediction.utils.utils import save_object

class DataTransformationConfig:
    preprocessor_obj_path:str=os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.config=DataTransformationConfig()
        
    def get_data_transformation(self):
        try:
            logging.info("geting data transformation(piplines)")
            
            cat_cols=['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']
            num_cols=['Distance_km', 'Preparation_Time_min','Courier_Experience_yrs']
            
            Weather_categories= ["Windy","Snowy","Foggy","Rainy","Clear"]
            Traffic_Level_categories=["High","Low","Medium"]
            Time_of_Day_categories= ["Night","Afternoon","Evening","Morning"]
            Vehicle_Type_categories= ["Car","Scooter","Bike"]
            logging.info("geting catagories for ordinal encoder")
            
            num_pipeline=Pipeline(
                [
                    ("impute",SimpleImputer()),
                    ("scaler",StandardScaler())
                ]
            )
            
            cat_pipeline=Pipeline(
                [
                    ("impute",SimpleImputer(strategy="most_frequent")),
                    ("encoder",OrdinalEncoder(categories=[Weather_categories,Traffic_Level_categories,Time_of_Day_categories,Vehicle_Type_categories]))
                ]
            )
            logging.info("creating pipelines")
            
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_cols),
                    ("cat_pipeline",cat_pipeline,cat_cols)
                ]
            )
            logging.info("creating preprocessor")
            return preprocessor
            
        except customexception as e:
            logging.info("error in data transformation", e)
            raise customexception(e,sys)
        
    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            logging.info("Starting Data Transformation")
            train_data=pd.read_csv(train_data_path)
            test_data=pd.read_csv(test_data_path)
            logging.info("read train and test data from artifacts")
            logging.info(f"this is train data\n{train_data.head(2).to_string()}")
            logging.info(f"this is test data\n{test_data.head(2).to_string()}")
            
            preprocessor=self.get_data_transformation()
            
            target_feature="Delivery_Time_min"
            drop_features=[target_feature,"Order_ID"]
            
            input_feature_train_data=train_data.drop(drop_features,axis=1)
            target_feature_train_data=train_data[target_feature]
            
            input_feature_test_data=test_data.drop(drop_features,axis=1)
            target_feature_test_data=test_data[target_feature]
            logging.info("splitting data into input and target features")
            
            input_feature_train_arr=preprocessor.fit_transform(input_feature_train_data)
            input_feature_test_arr=preprocessor.transform(input_feature_test_data)
            logging.info("applying preprocessing on traning and testing datasets")
            
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_data)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_data)]
            logging.info("merging input and target features")
            
            save_object(file_path=self.config.preprocessor_obj_path,
                        obj=preprocessor)
            logging.info("saving preprocessor object")
            
            return train_arr,test_arr
        except customexception as e:
            logging.info("error in data transformation", e)
            raise customexception(e,sys)