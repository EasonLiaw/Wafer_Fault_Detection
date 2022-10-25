'''
Author: Liaw Yi Xian
Last Modified: 25th October 2022
'''

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from Application_Logger.logger import App_Logger
import numpy as np
import joblib
import feature_engine.selection as fes
import feature_engine.imputation as fei
import seaborn as sns
import matplotlib.pyplot as plt
import os
import plotly.express as px
import plotly.figure_factory as ff
from tqdm import tqdm

random_state=120

class train_Preprocessor:


    def __init__(self, file_object, data_path, result_dir):
        '''
            Method Name: __init__
            Description: This method initializes instance of train_Preprocessor class
            Output: None

            Parameters:
            - file_object: String path of logging text file
            - result_dir: String path for storing intermediate results from running this class
        '''
        self.file_object = file_object
        self.data_path = data_path
        self.result_dir = result_dir
        self.log_writer = App_Logger()


    def extract_compiled_data(self):
        '''
            Method Name: extract_compiled_data
            Description: This method extracts data from a csv file and converts it into a pandas dataframe.
            Output: A pandas dataframe
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(
            self.file_object, "Start reading compiled data from database")
        try:
            data = pd.read_csv(self.data_path)
        except Exception as e:
            self.log_writer.log(
                self.file_object, f"Fail to read compiled data from database with the following error: {e}")
            raise Exception(
                f"Fail to read compiled data from database with the following error: {e}")
        self.log_writer.log(
            self.file_object, "Finish reading compiled data from database")
        return data


    def remove_irrelevant_columns(self, data):
        '''
            Method Name: remove_irrelevant_columns
            Description: This method removes columns from a pandas dataframe, which are not relevant for analysis.
            Output: A pandas DataFrame after removing the specified columns. In addition, columns that are removed will be stored in a separate csv file.
            On Failure: Logging error and raise exception

            Parameters:
            - data: Dataframe object
        '''
        self.log_writer.log(
            self.file_object, "Start removing irrelevant columns from the dataset")
        try:
            data = data.drop(self.index_col, axis=1)
            result = pd.concat(
                [pd.Series(self.index_col, name='Columns_Removed'), pd.Series(["Irrelevant column"]*len([self.index_col]), name='Reason')], axis=1)
            result.to_csv(self.result_dir+self.col_drop_path, index=False)
        except Exception as e:
            self.log_writer.log(
                self.file_object, f"Irrelevant columns could not be removed from the dataset with the following error: {e}")
            raise Exception(
                f"Irrelevant columns could not be removed from the dataset with the following error: {e}")
        self.log_writer.log(
            self.file_object, "Finish removing irrelevant columns from the dataset")
        return data


    def remove_duplicated_rows(self, data):
        '''
            Method Name: remove_duplicated_rows
            Description: This method removes duplicated rows from a pandas dataframe.
            Output: A pandas DataFrame after removing duplicated rows. In addition, duplicated records that are removed will be stored in a separate csv file labeled "Duplicated_Records_Removed.csv"
            On Failure: Logging error and raise exception

            Parameters:
            - data: Dataframe object
        '''
        self.log_writer.log(
            self.file_object, "Start handling duplicated rows in the dataset")
        if len(data[data.duplicated()]) == 0:
            self.log_writer.log(
                self.file_object, "No duplicated rows found in the dataset")
        else:
            try:
                data[data.duplicated()].to_csv(
                    self.result_dir+'Duplicated_Records_Removed.csv', index=False)
                data = data.drop_duplicates(ignore_index=True)
            except Exception as e:
                self.log_writer.log(
                    self.file_object, f"Fail to remove duplicated rows with the following error: {e}")
                raise Exception(
                    f"Fail to remove duplicated rows with the following error: {e}")
        self.log_writer.log(
            self.file_object, "Finish handling duplicated rows in the dataset")
        return data
    

    def features_and_labels(self,data,target_col):
        '''
            Method Name: features_and_labels
            Description: This method splits a pandas dataframe into two pandas objects, consist of features and target labels.
            Output: Two pandas/series objects consist of features and labels separately.
            On Failure: Logging error and raise exception

            Parameters:
            - data: Dataframe object
        '''
        self.log_writer.log(
            self.file_object, "Start separating the data into features and labels")
        try:
            X = data.drop(target_col, axis=1)
            y = data[target_col]
        except Exception as e:
            self.log_writer.log(
                self.file_object, f"Fail to separate features and labels with the following error: {e}")
            raise Exception(
                f"Fail to separate features and labels with the following error: {e}")
        self.log_writer.log(
            self.file_object, "Finish separating the data into features and labels")
        return X, y
    

    def handling_large_zeros(self, data, continuous_columns, threshold):
        '''
            Method Name: handling_large_zeros
            Description: This method adds columns (binary indicator) for features that have proportion of zero values exceeding a given threshold.
            Output: A pandas dataframe, where zero value indicator was included for variables with values of zeros.
            On Failure: Logging error and raise exception

            Parameters:
            - data: Dataframe object
            - continuous_columns: List of continuous columns
            - threshold: Value set for proportion of zero values
        '''
        self.log_writer.log(
            self.file_object, "Start checking and handling variables with zero values")
        try:
            cols_with_large_zeros = []
            for col in continuous_columns:
                if (data[col] == 0).sum() > threshold*len(data):
                    cols_with_large_zeros.append(col)
                    data[col+"_zero"] = np.where(data[col]==0, 1, 0)
            joblib.dump(
                cols_with_large_zeros, self.result_dir+"ZeroIndicator.pkl")
        except Exception as e:
            self.log_writer.log(
                self.file_object, f"Fail to check and handle variables with zero values with the following error: {e}")
            raise Exception(
                f"Fail to check and handle variables with zero values with the following error: {e}")
        self.log_writer.log(
            self.file_object, "Finish checking and handling variables with zero values")
        return data


    def add_missing_indicator(self, data):
        '''
            Method Name: add_missing_indicator
            Description: This method adds missing indicator to variables that contain missing values
            Output: A pandas dataframe, where missing indicator was included for variables with missing values.
            On Failure: Logging error and raise exception

            Parameters:
            - data: Dataframe object
        '''
        self.log_writer.log(
            self.file_object, f"Start adding missing indicator for features with missing values")
        try:
            selector = fei.AddMissingIndicator()
            data = selector.fit_transform(data)
            joblib.dump(
                selector,open(self.result_dir + f'AddMissingIndicator.pkl','wb'))
            self.log_writer.log(
                self.file_object, f"Missing indicator was created for the following set of features: {selector.variables_}")
        except Exception as e:
            self.log_writer.log(
                self.file_object, f"Fail to add missing indicator for features with missing values with the following error: {e}")
            raise Exception(
                f"Fail to add missing indicator for features with missing values with the following error: {e}")
        self.log_writer.log(
            self.file_object, f"Finish adding missing indicator for features with missing values")
        return data


    def drop_constant_variance(self, data, threshold):
        '''
            Method Name: drop_constant_variance
            Description: This method removes variables that have constant variance from the dataset.
            Output: A pandas dataframe, where variables with constant variance are removed.  In addition, variables that were removed due to constant variance are stored in a csv file.
            On Failure: Logging error and raise exception

            Parameters:
            - data: Dataframe object
            - threshold: Value set for detecting constant features
        '''
        self.log_writer.log(
            self.file_object, f"Start removing features with constant variance")
        try:
            selector = fes.DropConstantFeatures(
                missing_values='ignore',tol=threshold)
            data = selector.fit_transform(data)
            joblib.dump(
                selector,open(self.result_dir + f'Dropconstantfeatures.pkl','wb'))
            result = pd.concat(
                [pd.Series(selector.features_to_drop_, name='Columns_Removed'), pd.Series(["Constant variance"]*len(selector.features_to_drop_), name='Reason')], axis=1)
            result.to_csv(
                self.result_dir + self.col_drop_path, index=False, mode='a+', header=False)
            self.log_writer.log(
                self.file_object, f"Following set of features were removed due to having constant variance: {selector.features_to_drop_}")
        except Exception as e:
            self.log_writer.log(
                self.file_object, f"Fail to remove features with constant variance with the following error: {e}")
            raise Exception(
                f"Fail to remove features with constant variance with the following error: {e}")
        self.log_writer.log(
            self.file_object, f"Finish removing features with constant variance")
        return data


    def eda(self, target_col):
        '''
            Method Name: eda
            Description: This method performs exploratory data analysis on the entire dataset, while generating various plots/csv files for reference.
            Output: None

            Parameters:
            - data_path: String path where data compiled from database is located
            - target_col: Name of column related to target variable
        '''
        self.log_writer.log(
            self.file_object, 'Start performing exploratory data analysis')
        path = os.path.join(self.result_dir, 'EDA')
        if not os.path.exists(path):
            os.mkdir(path)
        scat_path = os.path.join(path, 'High_Correlation_Scatterplots')
        if not os.path.exists(scat_path):
            os.mkdir(scat_path)
        data = self.extract_compiled_data()
        # Extract basic information about dataset
        pd.DataFrame({"name": data.columns, "non-nulls": len(data)-data.isnull().sum().values, "type": data.dtypes.values}).to_csv(self.result_dir + "EDA/Data_Info.csv",index=False)
        # Extract summary statistics about dataset
        data.describe().T.to_csv(
            self.result_dir + "EDA/Data_Summary_Statistics.csv")
        X, y = self.features_and_labels(data, target_col)
        # Plotting proportion of null values and zero values of dataset
        zero_prop = []
        null_prop = []
        for col in X.columns:
            zero_prop.append(np.round((X[col]==0).mean()*100,2))
            null_prop.append(np.round(X[col].isnull().mean(),2))
        zero_results = pd.DataFrame(
            [X.columns, zero_prop], index=['Variable','Proportion']).T
        null_results = pd.DataFrame(
            [X.columns, null_prop], index=['Variable','Proportion']).T
        zero_results = zero_results[zero_results['Proportion']>0].sort_values(by='Proportion',ascending=False)
        null_results = null_results[null_results['Proportion']>0].sort_values(by='Proportion',ascending=False)
        plt.figure(figsize=(16, 48),dpi=100)
        barplot = sns.barplot(
            data=zero_results,y='Variable',x='Proportion',palette='flare_r')
        for rect in barplot.patches:
            width = rect.get_width()
            plt.text(
                rect.get_width(), rect.get_y()+0.5*rect.get_height(),'%.2f' % width, ha='left', va='center')
        plt.title("Proportion of zero values", fontdict={'fontsize':20})
        plt.savefig(
            self.result_dir+"EDA/Proportion of zero values",bbox_inches='tight', pad_inches=0.2)
        plt.clf()
        plt.figure(figsize=(16, 48),dpi=100)
        barplot = sns.barplot(
            data=null_results,y='Variable',x='Proportion',palette='flare_r')
        for rect in barplot.patches:
            width = rect.get_width()
            plt.text(
                rect.get_width(), rect.get_y()+0.5*rect.get_height(),'%.2f' % width, ha='left', va='center')
        plt.title("Proportion of null values", fontdict={'fontsize':20})
        plt.savefig(
            self.result_dir+"EDA/Proportion of null values",bbox_inches='tight', pad_inches=0.2)
        plt.clf()
        for col in tqdm(X.columns[1:]):
            col_path = os.path.join(path, col)
            if not os.path.exists(col_path):
                os.mkdir(col_path)
            # Plotting boxplot of features based on target variable
            fig = px.box(
                data,y=col,x=target_col,title=f"{col} Boxplot by Class")
            fig.write_image(
                self.result_dir + f"EDA/{col}/{col}_Boxplot_By_Class.png")
            # Plotting boxplot of features
            fig2 = px.box(data,x=col,title=f"{col} Boxplot")
            fig2.write_image(
                self.result_dir + f"EDA/{col}/{col}_Boxplot.png")
            try:
                # Plotting kdeplot of features
                fig3 = ff.create_distplot(
                    [X[col].dropna()], [col], show_hist=False,show_rug=False)
                fig3.layout.update(
                    title=f'{col} Density curve (Skewness: {np.round(X[col].dropna().skew(),4)})')
                fig3.write_image(
                    self.result_dir + f"EDA/{col}/{col}_Distribution.png")
            except:
                continue
            if X[col].isnull().mean() != 0:
                # Plotting histogram of number of missing values of features by target class
                fig4 = px.histogram(
                    data,x=X[col].isnull(),color=target_col,title=f"{col} Number of Missing Values by Class",text_auto=True)
                fig4.write_image(
                    self.result_dir + f"EDA/{col}/{col}_Count_Missing_By_Class.png")
            if (X[col] == 0).sum() !=0:
                # Plotting histogram of number of zero values of features by target class
                fig5 = px.histogram(
                    data,x=(X[col]==0),color=target_col,title=f"{col} Number of Zero Values by Class",text_auto=True)
                fig5.write_image(
                    self.result_dir + f"EDA/{col}/{col}_Count_Zeros_By_Class.png")
        # Plotting target class distribution
        target_dist = data['Output'].value_counts().reset_index().rename(columns={'index': 'Output','Output':'Number_Obs'})
        fig6 = px.bar(
            target_dist,x='Output',y='Number_Obs',title=f"Target Class Distribution",text_auto=True)
        fig6.write_image(self.result_dir + f"EDA/Target_Class_Distribution.png")
        # Plotting scatterplot between features that are highly correlated (>0.8) with each other based on absolute value of spearman correlation
        corr_matrix = data.corr(method='spearman')
        c1 = corr_matrix.stack().sort_values(ascending=False).drop_duplicates()
        high_cor = c1[c1.values!=1]
        results = high_cor[(high_cor>0.8) | (high_cor<-0.8)].reset_index()
        for col1, col2 in tqdm(zip(results['level_0'],results['level_1'])):
            fig7 = px.scatter(
                data,x=col1,y=col2,title=f"Scatterplot of {col1} vs {col2} (Spearman corr.: {np.round(corr_matrix.loc[col1, col2],4)})")
            fig7.write_image(
                self.result_dir + f"EDA/High_Correlation_Scatterplots/Scatterplot_{col1}_vs_{col2}.png")
        self.log_writer.log(
            self.file_object, 'Finish performing exploratory data analysis')


    def data_preprocessing(self, col_drop_path, index_col, target_col):
        '''
            Method Name: data_preprocessing
            Description: This method performs all the data preprocessing tasks for the data.
            Output: None
            
            Parameters:
            - data_path: String path where data compiled from database is located
            - col_drop_path: String path that stores list of columns that are removed from the data
            - index_col: Name of column related to unique IDs
            - target_col: Name of column related to target variable
        '''
        self.log_writer.log(self.file_object, 'Start of data preprocessing')
        self.col_drop_path = col_drop_path
        self.index_col = index_col
        self.target_col = target_col
        data = self.extract_compiled_data()
        data = self.remove_irrelevant_columns(data = data)
        data = self.remove_duplicated_rows(data = data)
        X, y = self.features_and_labels(data = data, target_col=self.target_col)
        y = y.replace(-1,0)
        continuous_columns = X._get_numeric_data().columns.tolist()
        X = self.add_missing_indicator(data = X)
        X = self.handling_large_zeros(
            data = X, continuous_columns = continuous_columns, threshold = 0.01)
        X= self.drop_constant_variance(data = X, threshold = 0.98)
        X.to_csv(self.result_dir+'X.csv',index=False)
        y.to_csv(self.result_dir+'y.csv',index=False)
        self.log_writer.log(self.file_object, 'End of data preprocessing')


