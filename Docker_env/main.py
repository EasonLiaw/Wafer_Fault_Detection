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
            'sensordata', "Training_Logs/Training_Main_Log.txt", "Good_Training_Data/", "Bad_Training_Data/")
        folders = ['Good_Training_Data/','Bad_Training_Data/','Archive_Training_Data/','Training_Data_FromDB/','Intermediate_Train_Results/','Caching/','Saved_Models/']
        trainvalidator.initial_data_preparation(
            'schema_training.json',folders,"Training_Batch_Files",'Good_Training_Data/','Archive_Training_Data/','Training_Data_FromDB/Training_Data.csv')
        st.success("This step of the pipeline has been completed successfully. Check the local files for more details.")
    if st.button("Exploratory Data Analysis"):
        preprocessor = train_Preprocessor(
            "Training_Logs/Training_Preprocessing_Log.txt", 'Intermediate_Train_Results/')
        preprocessor.eda('Training_Data_FromDB/Training_Data.csv','Output')
        st.success("This step of the pipeline has been completed successfully. Check the local files for more details.")
    if st.button("Training Data Preprocessing"):
        preprocessor = train_Preprocessor(
            "Training_Logs/Training_Preprocessing_Log.txt", 'Intermediate_Train_Results/')
        preprocessor.data_preprocessing(
            'Training_Data_FromDB/Training_Data.csv', 'Columns_Drop_from_Original.csv','Wafer','Output')
        st.success("This step of the pipeline has been completed successfully. Check the local files for more details.")
    if st.button("Model Selection"):
        trainer = model_trainer("Training_Logs/Training_Model_Log.txt")
        X = pd.read_csv('Intermediate_Train_Results/X.csv')
        y = pd.read_csv('Intermediate_Train_Results/y.csv')
        trainer.model_selection(X, y,'Intermediate_Train_Results/')
        st.success("This step of the pipeline has been completed successfully. Check the local files for more details.")
    if st.button("Final Model Training"):
        trainer = model_trainer("Training_Logs/Training_Model_Log.txt")
        X = pd.read_csv('Intermediate_Train_Results/X.csv')
        y = pd.read_csv('Intermediate_Train_Results/y.csv')
        trainer.final_model_tuning(X, y,'Intermediate_Train_Results/')
        st.success("This step of the pipeline has been completed successfully. Check the local files for more details.")
    if st.button("Prediction Data Validation"):
        predvalidator = rawpreddatavalidation(
            'predsensordata', "Prediction_Logs/Prediction_Main_Log.txt", "Good_Prediction_Data/", "Bad_Prediction_Data/")
        folders = ['Good_Prediction_Data/','Bad_Prediction_Data/','Archive_Prediction_Data/','Prediction_Data_FromDB/','Intermediate_Pred_Results/']
        predvalidator.initial_data_preparation(
            'schema_prediction.json',folders,"Prediction_Batch_Files/",'Good_Prediction_Data/','Archive_Prediction_Data/','Prediction_Data_FromDB/Prediction_Data.csv')
        st.success("This step of the pipeline has been completed successfully. Check the local files for more details.")
    if st.button("Prediction Data Preprocessing"):
        preprocessor = pred_Preprocessor(
            "Prediction_Logs/Prediction_Preprocessing_Log.txt")
        preprocessor.data_preprocessing(
            'Prediction_Data_FromDB/Prediction_Data.csv','Intermediate_Train_Results/','Wafer')
        st.success("This step of the pipeline has been completed successfully. Check the local files for more details.")
    if st.button("Model Prediction"):
        predictor = model_predictor("Prediction_Logs/Prediction_Model_Log.txt")
        predictor.model_prediction(
            'Intermediate_Pred_Results/Prediction_Processed_Data.csv',joblib.load('Saved_Models/FinalModel.pkl'),joblib.load('Saved_Models/Preprocessing_Pipeline.pkl'))
        st.success("This step of the pipeline has been completed successfully. Check the local files for more details.")
        
if __name__=='__main__':
    main()