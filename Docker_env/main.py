from Model_Prediction_Modules.validation_pred_data import rawpreddatavalidation
from Model_Prediction_Modules.pred_preprocessing import pred_Preprocessor
from Model_Prediction_Modules.model_prediction import model_predictor
from Model_Training_Modules.validation_train_data import rawtraindatavalidation
from Model_Training_Modules.train_preprocessing import train_Preprocessor
from Model_Training_Modules.model_training import model_trainer
import pandas as pd
import streamlit as st
import joblib

def main():
    st.title("Wafer Status Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Wafer Status Prediction App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    if st.button("Training Data Validation"):
        trainvalidator = rawtraindatavalidation(
            tablename = 'sensordata', file_object = "Training_Logs/Training_Main_Log.txt", gooddir = "Good_Training_Data/", baddir = "Bad_Training_Data/")
        folders = ['Good_Training_Data/','Bad_Training_Data/','Archive_Training_Data/','Training_Data_FromDB/','Intermediate_Train_Results/','Caching/','Saved_Models/']
        trainvalidator.initial_data_preparation(
            schemapath = 'schema_training.json', folders = folders,batchfilepath = "Training_Batch_Files", goodfilepath = 'Good_Training_Data/', archivefilepath= 'Archive_Training_Data/',compileddatapath= 'Training_Data_FromDB/Training_Data.csv')
        st.success("This step of the pipeline has been completed successfully. Check the local files for more details.")
    if st.button("Exploratory Data Analysis"):
        preprocessor = train_Preprocessor(
            file_object= "Training_Logs/Training_Preprocessing_Log.txt", result_dir= 'Intermediate_Train_Results/')
        preprocessor.eda(
            data_path = 'Training_Data_FromDB/Training_Data.csv',target_col = 'Output')
        st.success("This step of the pipeline has been completed successfully. Check the local files for more details.")
    if st.button("Training Data Preprocessing"):
        preprocessor = train_Preprocessor(
            file_object= "Training_Logs/Training_Preprocessing_Log.txt", result_dir= 'Intermediate_Train_Results/')
        preprocessor.data_preprocessing(
            data_path= 'Training_Data_FromDB/Training_Data.csv', col_drop_path= 'Columns_Drop_from_Original.csv', index_col= 'Wafer', target_col = 'Output')
        st.success("This step of the pipeline has been completed successfully. Check the local files for more details.")
    if st.button("Model Selection"):
        trainer = model_trainer(
            file_object= "Training_Logs/Training_Model_Log.txt")
        X = pd.read_csv('Intermediate_Train_Results/X.csv')
        y = pd.read_csv('Intermediate_Train_Results/y.csv')
        trainer.model_selection(
            input = X, output = y, num_trials = 20, folderpath = 'Intermediate_Train_Results/')
        st.success("This step of the pipeline has been completed successfully. Check the local files for more details.")
    if st.button("Final Model Training"):
        trainer = model_trainer(
            file_object= "Training_Logs/Training_Model_Log.txt")
        X = pd.read_csv('Intermediate_Train_Results/X.csv')
        y = pd.read_csv('Intermediate_Train_Results/y.csv')
        trainer.final_model_tuning(
            input_data = X, output_data = y, num_trials = 20, folderpath = 'Intermediate_Train_Results/')
        st.success("This step of the pipeline has been completed successfully. Check the local files for more details.")
    if st.button("Prediction Data Validation"):
        predvalidator = rawpreddatavalidation(
            tablename = 'predsensordata', file_object= "Prediction_Logs/Prediction_Main_Log.txt", gooddir= "Good_Prediction_Data/", baddir= "Bad_Prediction_Data/")
        folders = ['Good_Prediction_Data/','Bad_Prediction_Data/','Archive_Prediction_Data/','Prediction_Data_FromDB/','Intermediate_Pred_Results/']
        predvalidator.initial_data_preparation(
            schemapath = 'schema_prediction.json', folders = folders,batchfilepath= "Prediction_Batch_Files/", goodfilepath= 'Good_Prediction_Data/', archivefilepath= 'Archive_Prediction_Data/',compileddatapath= 'Prediction_Data_FromDB/Prediction_Data.csv')
        st.success("This step of the pipeline has been completed successfully. Check the local files for more details.")
    if st.button("Prediction Data Preprocessing"):
        preprocessor = pred_Preprocessor(
            file_object= "Prediction_Logs/Prediction_Preprocessing_Log.txt")
        preprocessor.data_preprocessing(
            data_path= 'Prediction_Data_FromDB/Prediction_Data.csv',train_path= 'Intermediate_Train_Results/', index_col= 'Wafer')
        st.success("This step of the pipeline has been completed successfully. Check the local files for more details.")
    if st.button("Model Prediction"):
        predictor = model_predictor(
            file_object= "Prediction_Logs/Prediction_Model_Log.txt")
        predictor.model_prediction(
            datapath = 'Intermediate_Pred_Results/Prediction_Processed_Data.csv', model = joblib.load('Saved_Models/FinalModel.pkl'), pipeline = joblib.load('Saved_Models/Preprocessing_Pipeline.pkl'))
        st.success("This step of the pipeline has been completed successfully. Check the local files for more details.")
        
if __name__=='__main__':
    main()