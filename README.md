# Wafer Status Classification Project

## Background
---

<img src="https://www.semiconductorforu.com/wp-content/uploads/2021/02/silicon-wafer.jpg">

In electronics, a wafer (also called a slice or substrate) is a thin slice of semiconductor used for the fabrication of integrated circuits. Monitoring working conditions of these wafers present its challenges of having additional resources required for manual monitoring with insights and decisions that need to be made quickly for replacing wafers that are not in good working conndition when required. Using IIOT (Industrial Internet of Things) helps to overcome this challenge through a collection of real-time data from multiple sensors. 

Thus, the main goal of this project is to design a machine learning model that predicts whether a wafer is in a good working condition or not based on inputs from 590 different sensors for every wafer. The quality of wafer sensors can be classified into two different categories: 0 for "good wafer" and 1 for "bad wafer".

Dataset is provided in .csv format by client under <b>Training_Batch_Files</b> folder for model training, while dataset under <b>Prediction_Batch_Files</b> folder will be used for predicting quality of wafer sensors.

In addition, schema of datasets for training and prediction is provided in .json format by the client for storing seperate csv files into a single MySQL database.

## Contents
- [Code and Resources Used](#code-and-resources-used)
- [Model Training Setting](#model-training-setting)
- [Project Findings](#project-findings)
  - [EDA](#1-eda-exploratory-data-analysis)
  - [Best classification model and pipeline configuration](#2-best-classification-model-and-pipeline-configuration)
  - [Summary of model evaluation metrics from best classification model](#3-summary-of-model-evaluation-metrics-from-best-classification-model)
  - [Hyperparameter importances from Optuna (Final model)](#4-hyperparameter-importances-from-optuna-final-model)
  - [Hyperparameter tuning optimization history from Optuna](#5-hyperparameter-tuning-optimization-history-from-optuna)
  - [Overall confusion matrix and classification report from final model trained](#6-overall-confusion-matrix-and-classification-report-from-final-model-trained)
  - [Discrimination Threshold for binary classification](#7-discrimination-threshold-for-binary-classification)
  - [Learning Curve Analysis](#8-learning-curve-analysis)
  - [Feature Importance based on Shap Values](#9-feature-importance-based-on-shap-values)
- [CRISP-DM Methodology](#crisp-dm-methodology)
- [Project Architecture Summary](#project-architecture-summary)
- [Project Folder Structure](#project-folder-structure)
- [Project Instructions (Local Environment)](#project-instructions-local-environment)
- [Project Instructions (Docker)](#project-instructions-docker)
- [Project Instructions (Heroku with Docker)](#project-instructions-heroku-with-docker)
- [Initial Data Cleaning and Feature Engineering](#initial-data-cleaning-and-feature-engineering)
- [Machine Pipelines Configuration](#machine-pipelines-configuration)
  - [Handling missing values](#i-handling-missing-values)
  - [Handling imbalanced data](#ii-handling-imbalanced-data)
  - [Handling outliers by capping at extreme values](#iii-handling-outliers-by-capping-at-extreme-values)
  - [Gaussian transformation on non-gaussian variables](#iv-gaussian-transformation-on-non-gaussian-variables)
  - [Feature Scaling](#v-feature-scaling)
  - [Feature Selection](#vi-feature-selection)
  - [Cluster Feature representation](#vii-cluster-feature-representation)
- [Legality](#legality)

## Code and Resources Used
---
- **Python Version** : 3.10.0
- **Packages** : borutashap, feature-engine, featurewiz, imbalanced-learn, joblib, catboost, lightgbm, matplotlib, mysql-connector-python, numpy, optuna, pandas, plotly, scikit-learn, scipy, seaborn, shap, streamlit, tqdm, xgboost, yellowbrick
- **Dataset source** : Education materials from OneNeuron platform
- **Database**: MySQL
- **MySQL documentation**: https://dev.mysql.com/doc/
- **Optuna documentation** : https://optuna.readthedocs.io/en/stable/
- **Feature Engine documentation** : https://feature-engine.readthedocs.io/en/latest/
- **Imbalanced Learn documentation** : https://imbalanced-learn.org/stable/index.html
- **Scikit Learn documentation** : https://scikit-learn.org/stable/modules/classes.html
- **Shap documentation**: https://shap.readthedocs.io/en/latest/index.html
- **XGBoost documentation**: https://xgboost.readthedocs.io/en/stable/
- **LightGBM documentation**: https://lightgbm.readthedocs.io/en/latest/index.html
- **CatBoost documentation**: https://catboost.ai/en/docs/
- **Numpy documentation**: https://numpy.org/doc/stable/
- **Pandas documentation**: https://pandas.pydata.org/docs/
- **Plotly documentation**: https://plotly.com/python/
- **Matplotlib documentation**: https://matplotlib.org/stable/index.html
- **Seaborn documentation**: https://seaborn.pydata.org/
- **Yellowbrick documentation**: https://www.scikit-yb.org/en/latest/
- **Scipy documentation**: https://docs.scipy.org/doc/scipy/
- **Streamlit documentation**: https://docs.streamlit.io/

## Model Training Setting
---
For this project, nested cross validation with stratification is used for identifying the best model class to use for model deployment. The inner loop of nested cross validation consists of 3 fold cross validation using Optuna (TPE Multivariate Sampler with 20 trials on optimizing average F1 score) for hyperparameter tuning on different training and validation sets, while the outer loop of nested cross validation consists of 5 fold cross validation for model evaluation on different test sets.

The diagram below shows how nested cross validation works:
<img src="https://mlr.mlr-org.com/articles/pdf/img/nested_resampling.png" width="600" height="350">

Given the dataset for this project is small (less than 1000 samples), nested cross validation is the most suitable cross validation method to use for model algorithm selection to provide a more realistic generalization error of machine learning models.

The following list of classification models are tested in this project:
- Logistic Regression
- Linear SVC
- K Neighbors Classifier
- Gaussian Naive Bayes
- Decision Tree Classifier
- Random Forest Classifier
- Extra Trees Classifier
- Ada Boost Classifier
- Gradient Boosting Classifier
- XGBoost Classifier
- LightGBM Classifier
- CatBoost Classifier

For model evaluation on binary classification, the following metrics are used in this project:
- Balanced accuracy
- Precision
- Recall
- F1 score (Main metric for Optuna hyperparameter tuning)
- Matthew's correlation coefficient
- Average precision score

## Project Findings
---

#### 1. EDA (Exploratory Data Analysis)

All plots generated from this section can be found in Intermediate_Train_Results/EDA folder.

#### i. Basic metadata of dataset
On initial inspection, the current dataset used in this project has a total of 591 features and 1 target label ("Output"). A single "Wafer" feature has "object" data type, which represents unique identifier of a given record and the remaining features have "float" data type that initially indicates continuous features.

![Target_Class_Distribution](https://user-images.githubusercontent.com/34255556/194898271-511dc280-e6ee-4d67-a053-500d3386cdd0.png)

From the diagram above, there is a very clear indication of target imbalance between class -1 (non-faulty) and class 1 (faulty) for binary classification. This indicates that target imbalancing needs to be addressed during model training.

![Proportion of null values](https://user-images.githubusercontent.com/34255556/194897104-3d6291f7-6431-4f83-bf8b-93fe34a724d9.png)

From the diagram above, features with missing values identified have missing proportions approximately grouped into one of the following: 1%, 2%, 3%, 6%, 9%, 34%, 60%, 63%, 67%, 73% and 92%.

![Proportion of zero values](https://user-images.githubusercontent.com/34255556/194897249-72e62b69-0c87-4b8e-a1b1-fa584eef23c3.png)

On another note, there's more than 120 features identified having more than 98% of zero values from the figure above, which might suggest that those features have very little variance which are not relevant for model training.

From performing spearman correlation analysis, there are 393 pairs of features having high spearman correlation (with absolute value of greater than 0.8) with one another. The scatterplot diagram below shows an example of two features having very high positive and negative spearman correlation with one another respectively:
<p float="left">
<img src="https://user-images.githubusercontent.com/34255556/194897320-8f23f3b7-6a8e-436e-af9d-9cd5d12ebe37.png" width="400">
<img src="https://user-images.githubusercontent.com/34255556/194897454-8991d997-97ed-4a36-9b33-2eb0e0fa9afd.png" width="400">
</p>

Although there are many pairs of features having high correlation with one another, this doesn't mean that those features with high correlation with other features should be removed from the dataset, which may result in sub-optimal model performance along with feature selection. More scatterplot diagrams can be found within EDA folder, which shows other pairs of features that have high spearman correlation.

Furthermore, the following sets of plots are created for every feature of the dataset:
1. Box plot
2. Box plot by target label
3. Bar plot (Number of missing values by target label) - For features with missing values
4. Bar plot (Number of zero values by target label) - For features with zero values
5. Kernel density estimation plot

Both box plots and kernel density estimation plot for every feature helps to identify distribution of data (gaussian vs non-gaussian) and identifying potential outliers.

The set of figures below shows an example of the following plots mentioned above for Sensor102:

<p float="left">
<img src="https://user-images.githubusercontent.com/34255556/194897845-39f360e2-51b5-43b9-8d06-c78cc6d4ba95.png" width="400">
<img src="https://user-images.githubusercontent.com/34255556/194897883-40b76270-c614-45ff-88e6-635c36d44230.png" width="400">
<img src="https://user-images.githubusercontent.com/34255556/194897903-8d960410-b571-4fdf-8f35-c2c55c164a74.png" width="400">
<img src="https://user-images.githubusercontent.com/34255556/194897924-8fd57fc4-5ed2-4228-8ab2-5b7ead0d07b1.png" width="400">
<img src="https://user-images.githubusercontent.com/34255556/194897953-b467e9a1-82e5-4b37-a997-c291227a829f.png" width="400">
</p>

---
#### 2. Best classification model and pipeline configuration

The following information below summarizes the configuration of the best model identified in this project:

  - <b>Best model class identified</b>: Linear Support Vector Classifier

  - <b>Method of handling imbalanced data</b>: SMOTEENN

  - <b>Method of handling outliers</b>: Retain outliers

  - <b>Method of feature scaling</b>: RobustScaler

  - <b>Removing highly correlated features (>0.8)</b>: Yes

  - <b>Feature selection method</b>: ANOVA

  - <b>Number of features selected</b>: 21

  - <b>List of features selected</b>: ['Sensor95_zero', 'Sensor419_zero', 'Sensor501_zero', 'Sensor104', 'Sensor57', 'Sensor122', 'Sensor112', 'Sensor102_zero', 'Sensor500_zero', 'Sensor434', 'Sensor337', 'Sensor96_zero', 'Sensor184', 'Sensor484_zero', 'Sensor101_zero', 'Sensor269', 'Sensor434_zero', 'Sensor101', 'Sensor113_na', 'Sensor419', 'Sensor110_na']
  
  - <b>Clustering as additional feature</b>: No

  - <b>Best model hyperparameters</b>: {'penalty': 'l2', 'loss': 'squared_hinge', 'dual': False, 'tol': 0.0001, 'C': 0.052092773149107, 'multi_class': 'ovr', 'fit_intercept': True, 'intercept_scaling': 1,'class_weight': None, 'verbose': 0, 'random_state': 120, 'max_iter': 1000}

Note that the results above may differ by changing search space of hyperparameter tuning or increasing number of trials used in hyperparameter tuning or changing number of folds within nested cross validation.

For every type of classification model tested in this project, a folder is created for every model class within Intermediate_Train_Results folder with the following artifacts:

- Confusion Matrix with default threshold from 5 fold cross validation (.png format)
- Classification Report with default threshold from 5 fold cross validation (.png format)
- HP_Importances for every fold (.png format - 5 in total)
- Hyperparameter tuning results for every fold (.csv format - 5 in total)
- Optimization history plot for every fold (.png format - 5 in total)
- Optuna study object for every fold (.pkl format - 5 in total)
- Precision-Recall curve (.png format)

In addition, the following artifacts are also created for the best model class identified after final hyperparameter tuning on the entire dataset:

- Confusion matrix with default threshold (.png format)
- Classification report with default threshold (.png format)
- HP_Importances (.png format)
- Hyperparameter tuning results (.csv format)
- Optimization history plot (.png format)
- Optuna study object (.pkl format)
- Learning curve plot (.png format)
- Shap plots for feature importance (.png format - 2 in total)
- Discrimination threshold plot (.png format)

---
#### 3. Summary of model evaluation metrics from best classification model

The following information below summarizes the evaluation metrics *(average (standard deviation)) from the best model identified in this project along with the confusion matrix from nested cross validation (5 outer fold with 3 inner fold): 

<p float="left">
<img src="https://user-images.githubusercontent.com/34255556/194896776-ff09e7eb-3c64-4752-b090-6efaef960761.png" width="400">
<img src="https://user-images.githubusercontent.com/34255556/194898691-5ccae77b-7193-4781-9c21-42373012b881.png" width="400">
</p>

  - <b>Balanced accuracy (Training set - 3 fold)</b>: 0.7735 (0.0942)
  - <b>Balanced accuracy (Validation set - 3 fold)</b>: 0.6145 (0.0316)
  - <b>Balanced accuracy (Test set - 5 fold)</b>: 0.5709 (0.0909)

  - <b>Precision (Training set - 3 fold)</b>: 0.2591 (0.0587)
  - <b>Precision (Validation set - 3 fold)</b>: 0.1608 (0.0386)
  - <b>Precision (Test set - 5 fold)</b>: 0.1133 (0.0720)

  - <b>Recall (Training set - 3 fold)</b>: 0.6653 (0.2135)
  - <b>Recall (Validation set - 3 fold)</b>: 0.3556 (0.1215)
  - <b>Recall (Test set - 5 fold)</b>: 0.2393 (0.1915)

  - <b>F1 score (Training set - 3 fold)</b>: 0.3604 (0.0858)
  - <b>F1 score (Validation set - 3 fold)</b>: 0.1831 (0.0222)
  - <b>F1 score (Test set - 5 fold)</b>: 0.1415 (0.0887)

  - <b>Matthews Correlation Coefficient (Training set - 3 fold)</b>: 0.3593 (0.1056)
  - <b>Matthews Correlation Coefficient (Validation set - 3 fold)</b>: 0.1583 (0.0351)
  - <b>Matthews Correlation Coefficient (Test set - 5 fold)</b>: 0.0993 (0.1109)
  
  - <b>Average Precision (Training set - 3 fold)</b>: 0.2037 (0.0824)
  - <b>Average Precision (Validation set - 3 fold)</b>: 0.0781 (0.0113)
  - <b>Average Precision (Test set - 5 fold)</b>: 0.1495 (0.0813)

Note that the results above may differ by changing search space of hyperparameter tuning or increasing number of trials used in hyperparameter tuning or changing number of folds within nested cross validation

---
#### 4. Hyperparameter importances from Optuna (Final model)

![HP_Importances_LinearSVC_Fold_overall](https://user-images.githubusercontent.com/34255556/194896701-0a6f16e5-6541-47ee-bf7e-d9a8e4d59833.png)

From the image above, determining the method for handling imbalanced data as part of preprocessing pipeline for Linear SVC model provides the highest influence (0.35), followed by selecting hyperparameter value of "C", feature selection and feature scaling method. Setting hyperparameter value of class weight and penalty for Linear SVC model provides little to zero influence on results of hyperparameter tuning. This may suggest that both class weight and penalty hyperparameters of Linear SVC model can be excluded from hyperparameter tuning in the future during model retraining to reduce complexity of hyperparameter tuning process.

---
#### 5. Hyperparameter tuning optimization history from Optuna

![Optimization_History_LinearSVC_Fold_overall](https://user-images.githubusercontent.com/34255556/194896645-c5402dae-0612-4c56-a91c-b2dd83c3896a.png)

From the image above, the best objective value (average of F1 scores from 3 fold cross validation) is identified at the end of the Optuna study (approximately 0.17). This may suggest that the number of Optuna trials can be increased further (more than 20 trials) within a reasonable budget, which better sets of hyperparameters may be identified towards later Optuna study trials.

---
#### 6. Overall confusion matrix and classification report from final model trained

<p float="left">
<img src="https://user-images.githubusercontent.com/34255556/194896442-59b34588-d2c5-4dd6-90c7-9140979d756d.png" width="400">
<img src="https://user-images.githubusercontent.com/34255556/194896481-376fe3bb-ab81-499c-a2f0-940ca7fe4a89.png" width="400">
</p>

From the image above, the classification model performs better for status of wafers in bad condition (1) with less false negatives (25 samples), as compared to false positives (51 samples). Given that the model evaluation criteria emphasize the costly impact of having both false positives and false negatives equally, the current classification model is optimized to improve F1 score.

---
#### 7. Discrimination Threshold for binary classification

![Binary_Threshold_LinearSVC](https://user-images.githubusercontent.com/34255556/194896328-35ff3021-3ee5-4e3c-a947-46d13c39a57b.png)

From the diagram above, performing 5 fold cross validation on the best model identified shows that the best threshold for optimizing F1 score is 0.13 (meaning that wafers with fault probability of 0.13 are identified as faulty). This threshold can be set when performing model predictions using the best model identified, however the current threshold used for model prediction in this project remains at default value of 0.5. Under the "Saved_Models" folder, a pickle file labeled "Binary_Threshold.pkl" can be used to extract the optimized threshold identified for model prediction with the following syntax below:

```
visualizer = joblib.load('Saved_Models/Binary_Threshold.pkl')
best_threshold = visualizer.thresholds_[visualizer.cv_scores_[visualizer.argmax].argmax()]
```

---
#### 8. Learning Curve Analysis

![LearningCurve_LinearSVC](https://user-images.githubusercontent.com/34255556/194896233-1f1e2e4c-d176-40e4-9c72-1dad15791bef.png)

From the diagram above, the gap between train and test F1 scores (from 5-fold cross validation) gradually decreases as number of training sample size increases.
Since the gap between both scores are very narrow, this indicates that adding more training data may not help to improve generalization of model.

---
#### 9. Feature Importance based on Shap Values

<p float="left">
<img src="https://user-images.githubusercontent.com/34255556/194895936-b0057e51-46e5-4ffa-bcfe-28c2fdd009a0.png" width="400">
<img src="https://user-images.githubusercontent.com/34255556/194895875-0947361f-a3a3-4e04-9fc1-ddfedd8dabfd.png" width="400">
</p>

From both diagrams above, zero value indicator of Sensor 95 is the most influential variable, while missing value indicator of Sensor 110 is the least influential variable from the top 21 variables identified from feature selection using ANOVA.

From observing Shap's summary plot (right figure), most continuous features with higher values have higher probability of wafer identified as faulty with the exception of Sensor112, Sensor101 and Sensor419 where lower values indicate higher probability of having a faulty wafer. In addition, most binary categorical features with zero value indicate higher probability of wafer identified as faulty, except for Sensor95, Sensor419 and Sensor500 with zero value indicator and Sensor113 with missing value indicator being the opposite scenario.

## CRISP-DM Methodology
---
For any given Machine Learning projects, CRISP-DM (Cross Industry Standard Practice for Data Mining) methodology is the most commonly adapted methodology used.
The following diagram below represents a simple summary of the CRISP-DM methodology for this project:

<img src="https://www.datascience-pm.com/wp-content/uploads/2018/09/crisp-dm-wikicommons.jpg" width="450" height="400">

Note that an alternative version of this methodology, known as CRISP-ML(Q) (Cross Industry Standard Practice for Machine Learning and Quality Assurance) can also be used in this project. However, the model monitoring aspect is not used in this project, which can be considered for future use.

## Project Architecture Summary
---
The following diagram below summarizes the structure for this project:

![image](https://user-images.githubusercontent.com/34255556/164873790-34d8826f-2acc-43c9-9d7c-6aafd2e2b355.png)

Note that all steps mentioned above have been logged accordingly for future reference and easy maintenance, which are stored in <b>Training_Logs</b> and <b>Prediction_Logs</b> folders. Any bad quality data identified for model training and model prediction will be archived accordingly in <b>Archive_Training_Data</b> and <b>Archive_Prediction_Data</b> folders.

## Project Folder Structure
---
The following points below summarizes the use of every file/folder available for this project:
1. Application_Logger: Helper module for logging model training and prediction process
2. Archive_Prediction_Data: Stores bad quality prediction csv files that have been used previously for model prediction
3. Archive_Training_Data: Stores bad quality training csv files that have been used previously for model training
4. Bad_Prediction_Data: Temporary folder for identifying bad quality prediction csv files
5. Bad_Training_Data: Temporary folder for identifying bad quality prediction csv files
6. Good_Prediction_Data: Temporary folder for identifying good quality prediction csv files
7. Good_Training_Data: Temporary folder for identifying good quality training csv files
8. Intermediate_Pred_Results: Stores results from model prediction
9. Intermediate_Train_Results: Stores results from EDA, data preprocessing and model training process
10. Model_Prediction_Modules: Helper modules for model prediction
11. Model_Training_Modules: Helper modules for model training
12. Prediction_Batch_Files: Stores csv batch files to be used for model prediction
13. Prediction_Data_FromDB: Stores compiled data from SQL database for model prediction
14. Prediction_Logs: Stores logging information from model prediction for future debugging and maintenance
15. Saved_Models: Stores best models identified from model training process for model prediction
16. Training_Batch_Files: Stores csv batch files to be used for model training
17. Training_Data_FromDB: Stores compiled data from SQL database for model training
18. Training_Logs: Stores logging information from model training for future debugging and maintenance
19. Dockerfile: Additional file for Docker project deployment
20. main.py: Main file for program execution
21. README.md: Details summary of project for presentation
22. requirements.txt: List of Python packages to install for project deployment
23. setup.py : Script for installing relevant python packages for project deployment
24. schema_prediction.json: JSON file that contains database schema for model prediction
25. schema_training.json: JSON file that contains database schema for model training


The following sections below explains the three main approaches that can be used for deployment in this project:
1. <b>Docker</b>
2. <b>Cloud Platform (Heroku with Docker)</b>
3. <b>Local environment</b>

## Project Instructions (Docker)
---
<img src="https://user-images.githubusercontent.com/34255556/195037066-21347c07-217e-4ecd-9fef-4e7f8cf3e098.png" width="600">

Deploying this project on Docker allows for portability between different environments and running instances without relying on host operating system.
  
<b>Note that docker image is created under Windows Operating system for this project, therefore these instructions will only work on other windows instances.</b>

<b> For deploying this project onto Docker, the following additional files are essential</b>:
- DockerFile
- requirements.txt
- setup.py

Docker Desktop needs to be installed into your local system (https://www.docker.com/products/docker-desktop/), before proceeding with the following steps:

1. Download and extract the zip file from this github repository into your local machine system.
<img src="https://user-images.githubusercontent.com/34255556/195367439-1dd10dd8-5e22-412e-8620-d4afb21176a0.png" width="600" height="200">

2. Copy Docker_env folder into a separate directory, before proceeding with subsequent steps which will use Docker_env folder as root directory.

3. Create the following volumes (mysql and mysql configuration) and network in Docker for connecting between database container and application container using the following syntax:
```
docker volume create mysql
docker volume create mysql_config
docker network create mysqlnet
```
- Note that the naming conventions for both volumes and network can be changed.

4. Run the following docker volumes and network for creating a new MySQL container in Docker:
```
docker run --rm -d -v mysql:/var/lib/mysql -v mysql_config:/etc/mysql -p 3306:3306 --network mysqlnet --name mysqldb -e MYSQL_ROOT_PASSWORD=custom_password mysql
```
Note that mysqldb refers to the name of the container, which will also be host name of database.

5. For checking if the MySQL container has been created successfully, the following command can be executed on a separate command prompt, which will prompt the user to enter root password defined in previous step:
```
docker exec -ti mysqldb mysql -u root -p
```
  
6. Add an additional Python file named as DBConnectionSetup.py that contains the following Python code structure: 
```
logins = {"host": <host_name>, 
          "user": <user_name>, 
          "password": <password>, 
          "dbname": <default_database_name>} 
```
- For security reasons, this file needs to be stored in private. (Default host is container name defined in step 4 and user is root for MySQL)

7. Build a new docker image on the project directory with the following command:
```
docker build -t api-name .
```

8. Run the docker image on the project directory with the following command: 
```
docker run --network mysqlnet -e PORT=8501 -p 8501:8501 api-name
```
Note that the command above creates a new docker app container with the given image "api-name". Adding network onto the docker app container will allow connection between two separate docker containers.

9. A new browser will open after successfully running the streamlit app with the following interface:
<img src = "https://user-images.githubusercontent.com/34255556/195365035-d2f9bc6e-76b6-45e8-ba25-db1b02e5d7a3.png" width="600">

Browser for the application can be opened from Docker Desktop by clicking on the specific button shown below:
![image](https://user-images.githubusercontent.com/34255556/195381876-b3377125-a9c1-46c0-aa4f-9734c430638d.png)

10. From the image above, click on Training Data Validation first for initializing data ingestion into MySQL, followed by subsequent steps from top to bottom in order to avoid potential errors with the model training/model prediction process. The image below shows an example of notification after the process is completed for Training Data Validation process:
<img src = "https://user-images.githubusercontent.com/34255556/195366117-9c65a3b6-b405-4967-9236-907f3b012439.png" width="600">

11. After running all steps of the pipeline, run the following command to extract files from a specific directory within the docker container to host machine for viewing:
```
docker cp <container-id>:<source-dir> <destination-dir>
```

## Project Instructions (Heroku with Docker)
---
<img src = "https://user-images.githubusercontent.com/34255556/195489080-3673ab77-833d-47f6-8151-0fed308b9eec.png" width="600">

A suitable alternative for deploying this project is to use docker images with cloud platforms like Heroku. 

<b> For deploying models onto Heroku platform, the following additional files are essential</b>:
- DockerFile
- requirements.txt
- setup.py

<b>Note that deploying this project onto other cloud platforms like GCP, AWS or Azure may have different additionnal files required.</b>

For replicating the steps required for running this project on your own Heroku account, the following steps are required:
1. Clone this github repository into your local machine system or your own Github account if available.
<img src="https://user-images.githubusercontent.com/34255556/195367439-1dd10dd8-5e22-412e-8620-d4afb21176a0.png" width="600" height="200">

2. Copy Docker_env folder into a separate directory, before proceeding with subsequent steps which will use Docker_env folder as root directory.

3. Go to your own Heroku account and create a new app with your own customized name.
<img src="https://user-images.githubusercontent.com/34255556/160223589-301262f6-6225-4962-a92f-fc7ca8a0eee9.png" width="600" height="400">

4. Go to "Resources" tab and search for ClearDB MySQL in the add-ons search bar.
<img src="https://user-images.githubusercontent.com/34255556/160224064-35295bf6-3170-447a-8eae-47c6721cf8f0.png" width="600" height="200">

5. Select the ClearDB MySQL add-on and select the relevant pricing plan. (Note that I select Punch plan, which currently cost about $9.99 per month to increase storage capacity for this project.)

6. Add an additional Python file named as DBConnectionSetup.py that contains the following Python code structure: 
```
  logins = {"host": <host_name>, 
            "user": <user_name>, 
            "password": <password>, 
            "dbname": <default_Heroku_database_name>}
```
- For security reasons, this file needs to be stored in private.

- Information related to host, user and dbname for ClearDB MySQL can be found in the settings tab under ConfigVars section as shown in the image below:

![image](https://user-images.githubusercontent.com/34255556/195486639-e1c94433-54c9-43ca-a134-c7501e84111f.png)

- <b>CLEARDB_DATABASE_URL has the following format: mysql://<user_name>:<pass_word>@<host_name>/<db_name>?reconnect=true</b>

7. From a new command prompt window, login to Heroku account and Container Registry by running the following commands:
```
heroku login
heroku container:login
```
Note that Docker needs to be installed on your local system before login to heroku's container registry.

8. Using the Dockerfile, push the docker image onto Heroku's container registry using the following command:
```
heroku container:push web -a app-name
```

9. Release the newly pushed docker images to deploy app using the following command:
```
heroku container:release web -a app-name
```

10. After successfully deploying docker image onto Heroku, open the app from the Heroku platform and you will see the following interface designed using Streamlit:
<img src = "https://user-images.githubusercontent.com/34255556/195365035-d2f9bc6e-76b6-45e8-ba25-db1b02e5d7a3.png" width="600">

11. From the image above, click on Training Data Validation first for initializing data ingestion into MySQL, followed by subsequent steps from top to bottom in order to avoid potential errors with the model training/model prediction process. The image below shows an example of notification after the process is completed for Training Data Validation process:
<img src = "https://user-images.githubusercontent.com/34255556/195366117-9c65a3b6-b405-4967-9236-907f3b012439.png" width="600">

<b>Important Note</b>: 
- Using "free" dynos on Heroku app only allows the app to run for a maximum of 30 minutes. Since the model training and prediction process takes a long time, consider changing the dynos type to "hobby" for unlimited time, which cost about $7 per month per dyno. You may also consider changing the dynos type to Standard 1X/2X for enhanced app performance.

- Unlike stand-alone Docker containers, Heroku uses an ephemeral hard drive, meaning that files stored locally from running apps on Heroku will not persist when apps are restarted (once every 24 hours). Any files stored on disk will not be visible from one-off dynos such as a heroku run bash instance or a scheduler task because these commands use new dynos. Best practice for having persistent object storage is to leverage a cloud file storage service such as Amazon’s S3 (not part of project scope but can be considered)

## Project Instructions (Local Environment)
---  
If you prefer to deploy this project on your local machine system, the steps for deploying this project has been simplified down to the following:

1. Download and extract the zip file from this github repository into your local machine system.
<img src="https://user-images.githubusercontent.com/34255556/195367439-1dd10dd8-5e22-412e-8620-d4afb21176a0.png" width="600" height="200">

2. Copy Docker_env folder into a separate directory, before proceeding with subsequent steps which will use Docker_env folder as root directory.

3. Open MySQL in your local machine system and create a new database name of your choice with the following syntax: 
```
CREATE DATABASE db_name;
```
- Note that you will need to install MySQL if not available in your local system: https://dev.mysql.com/downloads/windows/installer/8.0.html
  
4. Add an additional Python file named as DBConnectionSetup.py that contains the following Python code structure: 
```
logins = {"host": <host_name>, 
          "user": <user_name>, 
          "password": <password>, 
          "dbname": <new_local_database_name>} 
```
- For security reasons, this file needs to be stored in private. (Default host is localhost and user is root for MySQL)
  
5. Open anaconda prompt and create a new environment with the following syntax: 
```
conda create -n myenv python=3.10
```
- Note that you will need to install anaconda if not available in your local system: https://www.anaconda.com/

6. After creating a new anaconda environment, activate the environment using the following command: 
```
conda activate myenv
```

7. Go to the local directory in Command Prompt where Docker_env folder is located and run the following command to install all the python libraries : 
```
pip install -r requirements.txt
```

8. After installing all the required Python libraries, run the following command on your project directory: 
```
streamlit run main.py
```

9. A new browser will open after successfully running the streamlit app with the following interface::
<img src = "https://user-images.githubusercontent.com/34255556/195365035-d2f9bc6e-76b6-45e8-ba25-db1b02e5d7a3.png" width="600">

10. From the image above, click on Training Data Validation first for initializing data ingestion into MySQL, followed by subsequent steps from top to bottom in order to avoid potential errors with the model training/model prediction process. The image below shows an example of notification after the process is completed for Training Data Validation process:
<img src = "https://user-images.githubusercontent.com/34255556/195366117-9c65a3b6-b405-4967-9236-907f3b012439.png" width="600">

## Initial Data Cleaning and Feature Engineering
---
After performing Exploratory Data Analysis, the following steps are performed initially on the entire dataset before performing further data preprocessing and model training:

i) Removing "Wafer" column, which is an ID representation of a given row

ii) Checking for duplicated rows and remove if exist

iii) Split dataset into features and target labels with values of -1 replaced as 0 (non-faulty).

iv) Adding missing indicator (binary value) for all continuous features

v) Adding zero value indicator (binary value) for all continuous features with more than 1% having zero values

vi) Remove all features that have low variance (less than 2%)

vii) Save reduced set of features and target values into 2 different CSV files (X.csv and y.csv) for further data preprocessing with pipelines to reduce data leakage.

For more details of which features have been initially removed from the dataset, refer to the following CSV file: <b>Columns_Drop_from_Original.csv</b>

In addition, the following pickle files (with self-explanatory names) have been created inside Intermediate_Train_Results folder during this stage which will be used later on during data preprocessing on test data batch:
- <b>AddMissingIndicator.pkl</b>
- <b>Dropconstantfeatures.pkl</b>
- <b>ZeroIndicator.pkl</b>

## Machine Pipelines Configuration
---
While data preprocessing steps can be done on the entire dataset before model training, it is highly recommended to perform all data preprocessing steps within cross validation using pipelines to reduce the risk of data leakage, where information from training data is leaked to validation/test data.

The sections below summarizes the details of Machine Learning pipelines with various variations in steps:

#### i. Handling missing values
Most machine learning models do not automatically handle missing values (with the exception of XGBoost, LightGBM and CatBoost). Therefore, missing values need appropriate handling first before subsequent steps of the pipeline can be executed.

For this project, missing values are handled using the following methods:

- Simple Mean Imputation: For gaussian features that have data missing completely at random (MCAR)
- Simple Median Imputation: For non gaussian features that have data missing completely at random (MCAR)
- End Tail Mean Imputation: For gaussian features that have data missing at random (MAR)
- End Tail Median Imputation: For non gaussian features that have data missing at random (MAR)

Note that skewness of features is used to distinguish between gaussian and non-gaussian features. On the other hand, number of features with low spearman correlation (between -0.4 and 0.4) of missingness with other features is used to distinguish between MCAR (All features) and MAR (Not all features).

For XGBoost, LightGBM and CatBoost, the presence of this pipeline step is also tested as part of hyperparameter tuning within nested cross validation.

#### ii. Handling imbalanced data
While most machine learning models have hyperparameters that allow adjustment of <b>class weights</b> for classification, an alternative solution to handle imbalanced data is to use oversampling or undersampling or combination of both oversampling and undersampling methods.

For this project, the following methods of handling imbalanced data are tested:

- SMOTETomek: Combine over (SMOTEENC) and under sampling using SMOTE and Tomek links.
- SMOTEENN: Combine over (SMOTEENC) and under sampling using SMOTE and Edited Nearest Neighbours.
- SMOTEENC: Synthetic Minority Over-sampling Technique for Nominal and Continuous data.
- No oversampling or undersampling required

For XGBoost, LightGBM and CatBoost, if handling missing values are not included as part of the pipeline, this step of the pipeline will be discarded.

#### iii. Handling outliers by capping at extreme values
Machine learning models like Logistic Regression, Linear SVC and KNN are highly sensitive to outliers, which may impact model performance. For those 3 types of models, the presence of this step of the pipeline by capping outliers at extreme ends of gaussian/non-gaussian distribution will be tested accordingly using Winsorizer function from feature-engine library.

Note that Anderson test is used to identify gaussian vs non-gaussian distribution in this pipeline step.

#### iv. Gaussian transformation on non-gaussian variables
In Machine Learning, several machine learning models like Logistic Regression and Gaussian Naive Bayes tends to perform best when data follows the assumption of normal distribution. The following types of gaussian transformation are tested on non-gaussian features and the gaussian transformation that works best on given feature (the lowest test statistic that is smaller than 5% critical value) will be used for gaussian transformation: 

- Logarithmic
- Reciprocal
- Square Root
- Yeo Johnson
- Square
- Quantile (Normal distribution)

Note that Anderson test is used to identify whether a given gaussian transformation technique successfully converts a non-gaussian feature to a gaussian feature.

#### v. Feature Scaling
Feature scaling is only essential in some machine learning models like Logistic Regression, Linear SVC and KNN for faster convergence and to prevent misinterpretation of one feature significantly more important than other features.

For this project, the following methods of feature scaling are tested:

- Standard Scaler
- MinMax Scaler
- Robust Scaler
- Standard Scaler for gaussian features + MinMax Scaler for non-gaussian features

#### vi. Feature Selection
Given the current dataset has very large number of features, performing feature selection is essential for simplifying the machine learning model, reducing model training time and to reduce risk of model overfitting.

For this project, the presence of removing highly correlated variables (>0.8) based on spearman correlation is tested (except for FeatureWiz method) along with the following methods of feature selection with pipelines that handle missing values:

- Mutual Information
- ANOVA
- Feature Importance using Extra Trees Classifier
- Logistic Regression with Lasso Penalty (l1)
- BorutaShap (Default base learner: Random Forest Classifier)
- FeatureWiz (SULOV (Searching for Uncorrelated List of Variables) + Recursive Feature Elimination with XGBoost Classifier)

For pipelines (XGBoost, LightGBM and CatBoost) that do not involve handling missing values, only feature selection method tested is based on feature importance for respective base learners.

#### vii. Cluster Feature representation
After selecting the best features from feature selection, an additional step that can be tested involves representing distance between various points and identified cluster point as a feature (cluster_distance) for model training. From the following research paper (https://link.springer.com/content/pdf/10.1007/s10115-021-01572-6.pdf) written by Maciej Piernik and Tadeusz Morzy in 2021, both authors concluded the following points that will be applied to this project:

-  Adding cluster-generated features may improve quality of classification models (linear classifiers like Logistic Regression and Linear SVC), with extra caution required for non-linear classifiers like K Neighbors Classifier and random forest approaches.

- Encoding clusters as features based on distances between points and cluster representatives with feature scaling is significantly better than solely relying on cluster membership with One Hot encoding. 

- Adding generated cluster features to existing ones is safer option than replacing them altogether, which may yield model improvements without degrading model quality

- No single clustering approach (K-means vs Hierarchical vs DBScan vs Affinity Propagation) provide significantly better results in model performance. Thus, affinity propagation method is used for this project, which automatically determines the number of clusters to use. However, "damping" parameter requires hyperparameter tuning for using Affinity Propagation method.

## Legality
---
This is a personal project made for non-commercial uses ONLY. This project will not be used to generate any promotional or monetary value for me, the creator, or the user.
