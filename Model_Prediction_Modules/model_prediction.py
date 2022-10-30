'''
Author: Liaw Yi Xian
Last Modified: 30th October 2022
'''

import pandas as pd
from Application_Logger.logger import App_Logger
from Application_Logger.exception import CustomException
import sys

class model_predictor:


    def __init__(self, file_object):
        '''
            Method Name: __init__
            Description: This method initializes instance of model_predictor class
            Output: None

            Parameters:
            - file_object: String path of logging text file
        '''
        self.file_object = file_object
        self.log_writer = App_Logger()

    
    def model_prediction(self, datapath, model, pipeline):
        '''
            Method Name: model_prediction
            Description: This method performs all the model prediction tasks for the data.
            Output: None
            On Failure: Logging error and raise exception

            Parameters:
            - data: Dataframe object
            - model: Model object
            - pipeline: Imblearn pipeline object
        '''
        self.log_writer.log(
            self.file_object, 'Start performing model prediction using best model identified on given data batch.')
        self.datapath = datapath
        self.model = model
        self.pipeline = pipeline
        try:
            pred_data = pd.read_csv(self.datapath)
            index = pred_data['Wafer']
            pred_data.drop('Wafer',axis=1, inplace=True)
            sub_pred_data = self.pipeline.transform(pred_data)
            sub_pred_data['Pred_Output'] = self.model.predict(sub_pred_data)
            pred_results = pd.concat(
                [index, sub_pred_data['Pred_Output']],axis=1)
            pred_results.to_csv(
                'Intermediate_Pred_Results/Predictions.csv',index=False)
        except Exception as e:
            self.log_writer.log(self.file_object, str(CustomException(e,sys)))
            raise CustomException(e,sys)
        self.log_writer.log(
            self.file_object, 'Finish performing model prediction using best model identified on given data batch.')