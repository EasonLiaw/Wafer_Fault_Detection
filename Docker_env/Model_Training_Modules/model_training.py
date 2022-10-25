'''
Author: Liaw Yi Xian
Last Modified: 25th October 2022
'''

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import optuna
import joblib
import time
import shap
from BorutaShap import BorutaShap
from featurewiz import FeatureWiz
import scipy.stats as st
from yellowbrick.classifier import DiscriminationThreshold
from yellowbrick.model_selection import LearningCurve
import feature_engine.imputation as fei
import feature_engine.selection as fes
import feature_engine.outliers as feo
import feature_engine.transformation as fet
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_validate, StratifiedKFold, learning_curve
from sklearn.feature_selection import mutual_info_classif, f_classif, SelectKBest, SelectFromModel
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, precision_score, recall_score, make_scorer, f1_score,average_precision_score, ConfusionMatrixDisplay, classification_report, PrecisionRecallDisplay
from Application_Logger.logger import App_Logger
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import pairwise_distances

random_state=120

class model_trainer:


    def __init__(self, file_object):
        '''
            Method Name: __init__
            Description: This method initializes instance of model_trainer class
            Output: None

            Parameters:
            - file_object: String path of logging text file
        '''
        self.file_object = file_object
        self.log_writer = App_Logger()
        self.optuna_selectors = {
            'LogisticRegression': {'obj': model_trainer.lr_objective,'clf': LogisticRegression(random_state=random_state)},
            'LinearSVC': {'obj': model_trainer.svc_objective, 'clf': LinearSVC(random_state=random_state)},
            'KNeighborsClassifier': {'obj': model_trainer.knn_objective, 'clf': KNeighborsClassifier()},
            'GaussianNB': {'obj': model_trainer.gaussiannb_objective, 'clf': GaussianNB()},
            'DecisionTreeClassifier': {'obj': model_trainer.dt_objective, 'clf': DecisionTreeClassifier(random_state=random_state)},
            'RandomForestClassifier': {'obj': model_trainer.rf_objective, 'clf': RandomForestClassifier(random_state=random_state)},
            'ExtraTreesClassifier': {'obj': model_trainer.et_objective, 'clf': ExtraTreesClassifier(random_state=random_state)},
            'AdaBoostClassifier': {'obj': model_trainer.adaboost_objective, 'clf': AdaBoostClassifier(random_state=random_state)},
            'GradientBoostingClassifier': {'obj': model_trainer.gradientboost_objective, 'clf': GradientBoostingClassifier(random_state=random_state)},
            'XGBClassifier': {'obj': model_trainer.xgboost_objective, 'clf': XGBClassifier(random_state=random_state)},
            'LGBMClassifier': {'obj': model_trainer.lightgbm_objective, 'clf': LGBMClassifier(random_state=random_state)},
            'CatBoostClassifier': {'obj': model_trainer.catboost_objective,'clf': CatBoostClassifier(random_state=random_state)}
        }


    def setting_attributes(trial, cv_results):
        '''
            Method Name: setting_attributes
            Description: This method sets attributes of metric results for training and validation set from a given Optuna trial
            Output: None

            Parameters:
            - trial: Optuna trial object
            - cv_results: Dictionary object related to results from cross validate function
        '''
        trial.set_user_attr("train_balanced_accuracy", 
                            np.nanmean(cv_results['train_balanced_accuracy']))
        trial.set_user_attr("val_balanced_accuracy", 
                            cv_results['test_balanced_accuracy'].mean())
        trial.set_user_attr("train_precision_score", 
                            np.nanmean(cv_results['train_precision_score']))
        trial.set_user_attr("val_precision_score", 
                            cv_results['test_precision_score'].mean())
        trial.set_user_attr("train_recall_score", 
                            np.nanmean(cv_results['train_recall_score']))
        trial.set_user_attr("val_recall_score", 
                            cv_results['test_recall_score'].mean())
        trial.set_user_attr("train_f1_score", 
                            np.nanmean(cv_results['train_f1_score']))
        trial.set_user_attr("val_f1_score", 
                            cv_results['test_f1_score'].mean())
        trial.set_user_attr("train_matthews_corrcoef", 
                            np.nanmean(cv_results['train_matthews_corrcoef']))
        trial.set_user_attr("val_matthews_corrcoef", 
                            cv_results['test_matthews_corrcoef'].mean())
        trial.set_user_attr(
            "train_average_precision_score",
            np.nanmean(cv_results['train_average_precision_score']))
        trial.set_user_attr(
            "val_average_precision_score",
            cv_results['test_average_precision_score'].mean())


    def pipeline_missing_step(pipeline, continuous_columns, continuous_index):
        '''
            Method Name: pipeline_missing_step
            Description: This method adds custom transformer with MissingTransformer class into pipeline for handling missing data.
            Output: None

            Parameters:
            - pipeline: imblearn pipeline object
            - continuous_columns: List of continuous variable names
            - continuous_index: List of continuous variable index
        '''
        missing_transformer = ColumnTransformer(
            [('missingtransform',MissingTransformer(continuous_columns),continuous_index)],remainder='passthrough',n_jobs=2)
        pipeline.steps.append(('missing',missing_transformer))


    def pipeline_balancing_step(
            pipeline, balancing_indicator, categorical_index):
        '''
            Method Name: pipeline_balancing_step
            Description: This method adds SMOTETomek or SMOTEENN object with SMOTENC or nothing into pipeline for handling imbalanced data.
            Output: None

            Parameters:
            - pipeline: imblearn pipeline object
            - balancing_indicator: String name indicating method of handling imbalanced data
            - categorical_index: List of categorical variable index
        '''
        if balancing_indicator == 'smotetomek':
            pipeline.steps.append(('smote',SMOTETomek(random_state=random_state, smote=SMOTENC(categorical_index, random_state=random_state, n_jobs=2), n_jobs=2)))
        elif balancing_indicator == 'smoteenn':
            pipeline.steps.append(('smote',SMOTEENN(random_state=random_state, smote=SMOTENC(categorical_index, random_state=random_state, n_jobs=2), n_jobs=2)))
        elif balancing_indicator == 'smoteenc':
            pipeline.steps.append(('smote',SMOTENC(categorical_index, random_state=random_state, n_jobs=2)))


    def pipeline_outlier_step(
            pipeline, outlier_indicator, continuous_columns, continuous_index):
        '''
            Method Name: pipeline_outlier_step
            Description: This method adds custom transformer with OutlierCapTransformer class into pipeline for capping outliers if relevant.
            Output: None

            Parameters:
            - pipeline: imblearn pipeline object
            - outlier_indicator: String name indicating method of handling outliers
            - continuous_columns: List of continuous variable names
            - continuous_index: List of continuous variable index
        '''
        if outlier_indicator == 'capped':
            outlier_transformer = ColumnTransformer(
                [('outlier',OutlierCapTransformer(continuous_columns),continuous_index)],remainder='passthrough',n_jobs=2)
            pipeline.steps.append(('outlier',outlier_transformer))


    def pipeline_scaling_step(
            pipeline, scaling_indicator, continuous_columns, continuous_index):
        '''
            Method Name: pipeline_scaling_step
            Description: This method adds custom transformer with ScalingTransformer class into pipeline for scaling data.
            Output: None

            Parameters:
            - pipeline: imblearn pipeline object
            - scaling_indicator: String that represents method of performing feature scaling. (Accepted values are 'Standard', 'MinMax', 'Robust', 'Combine' and 'no').
            - continuous_columns: List of continuous variable names
            - continuous_index: List of continuous variable index
        '''
        scaling_transformer = ColumnTransformer(
            [('scaling',ScalingTransformer(scaling_indicator,continuous_columns),continuous_index)],remainder='passthrough',n_jobs=2)
        pipeline.steps.append(('scaling',scaling_transformer))


    def pipeline_feature_selection_step(
            pipeline, trial, fs_method, drop_correlated, continuous_columns, categorical_columns, clf, scaling_indicator='no', cluster_indicator='no', damping=None):
        '''
            Method Name: pipeline_feature_selection_step
            Description: This method adds custom transformer with FeatureSelectionTransformer class into pipeline for performing feature selection.
            Output: None
    
            Parameters:
            - pipeline: imblearn pipeline object
            - trial: Optuna trial object
            - fs_method: String name indicating method of feature selection
            - drop_correlated: String indicator of dropping highly correlated features (yes or no)
            - continuous_columns: List of continuous variable names
            - categorical_columns: List of categorical variable names
            - clf: Model object
            - scaling_indicator: String that represents method of performing feature scaling. (Accepted values are 'Standard', 'MinMax', 'Robust', 'Combine' and 'no'). Default value is 'no'
            - cluster_indicator: String indicator of including cluster-related feature (yes or no). Default value is 'no'
            - damping: Float value (range from 0.5 to 1 not inclusive) as an additional hyperparameter for Affinity Propagation clustering algorithm. Default value is None.
        '''
        if fs_method not in ['BorutaShap','FeatureWiz']:
            number_to_select = trial.suggest_int('number_features',1,30)
        else:
            number_to_select = None
        trial.set_user_attr("number_features", number_to_select)
        pipeline.steps.append(
            ('featureselection',FeatureSelectionTransformer(fs_method, drop_correlated, continuous_columns, categorical_columns, clf, scaling_indicator = scaling_indicator, cluster_indicator = cluster_indicator, damping=damping, number = number_to_select)))


    def pipeline_setup(
            pipeline, trial, continuous_columns, continuous_index,categorical_columns, categorical_index, clf):
        '''
            Method Name: pipeline_setup
            Description: This method configures pipeline for model training, which varies depending on model class and preprocessing related parameters selected by Optuna.
            Output: None
    
            Parameters:
            - pipeline: imblearn pipeline object
            - trial: Optuna trial object
            - continuous_columns: List of continuous variable names
            - continuous_index: List of continuous variable index
            - categorical_columns: List of categorical variable names
            - categorical_index: List of categorical variable index
            - clf: Model object
        '''
        missing_indicator = trial.suggest_categorical('missing',['yes','no']) if type(clf).__name__ in ['XGBClassifier','LGBMClassifier','CatBoostClassifier'] else 'no'
        trial.set_user_attr("missing_indicator", missing_indicator)
        if missing_indicator == 'no':
            model_trainer.pipeline_missing_step(
                pipeline, continuous_columns, continuous_index)
            balancing_indicator = trial.suggest_categorical(
                'balancing',['smoteenn','smotetomek','smoteenc','no'])
            model_trainer.pipeline_balancing_step(
                pipeline, balancing_indicator, categorical_index)
            outlier_indicator = trial.suggest_categorical('outlier',['capped','retained']) if type(clf).__name__ in ['LogisticRegression','LinearSVC','KNeighborsClassifier','GaussianNB'] else 'retained'
            model_trainer.pipeline_outlier_step(
                pipeline, outlier_indicator, continuous_columns, continuous_index)
            if type(clf).__name__ in ['LogisticRegression','GaussianNB']:
                gaussian_transformer = ColumnTransformer(
                    [('gaussiantransform',GaussianTransformer(continuous_columns),continuous_index)],remainder='passthrough',n_jobs=2)
                pipeline.steps.append(('gaussian',gaussian_transformer))
            if type(clf).__name__ in ['LogisticRegression','LinearSVC','KNeighborsClassifier']:
                scaling_indicator = trial.suggest_categorical(
                    'scaling',['Standard','MinMax','Robust','Combine'])
                model_trainer.pipeline_scaling_step(
                    pipeline, scaling_indicator, continuous_columns, continuous_index)
            else:
                scaling_indicator = 'no'
            fs_method = trial.suggest_categorical(
                'feature_selection',['BorutaShap','Lasso','FeatureImportance_ET','MutualInformation','ANOVA','FeatureWiz'])
            if fs_method != 'FeatureWiz':
                drop_correlated = trial.suggest_categorical(
                    'drop_correlated',['yes','no'])
            else:
                drop_correlated = 'no'
            cluster_indicator = trial.suggest_categorical('cluster_indicator',['yes','no']) if type(clf).__name__ in ['LogisticRegression','LinearSVC'] else 'no'
            damping = trial.suggest_float('damping',0.5,0.99,log=True) if cluster_indicator == 'yes' else None
        else:
            balancing_indicator = 'no'
            outlier_indicator = 'retained'
            scaling_indicator = 'no'
            drop_correlated = 'no'
            fs_method = 'FeatureImportance_self'
            cluster_indicator = 'no'
            damping = None
        model_trainer.pipeline_feature_selection_step(
            pipeline, trial, fs_method, drop_correlated, continuous_columns, categorical_columns, clf, scaling_indicator=scaling_indicator, cluster_indicator=cluster_indicator, damping = damping)
        trial.set_user_attr("balancing_indicator", balancing_indicator)
        trial.set_user_attr("outlier_indicator", outlier_indicator)
        trial.set_user_attr("scaling_indicator", scaling_indicator)
        trial.set_user_attr("drop_correlated", drop_correlated)
        trial.set_user_attr("feature_selection", fs_method)
        trial.set_user_attr("Pipeline", pipeline) 
        trial.set_user_attr("cluster_indicator", cluster_indicator)
        trial.set_user_attr("damping", damping)


    def lr_objective(
            trial,X_train_data,y_train_data, continuous_columns, continuous_index, categorical_columns, categorical_index):
        '''
            Method Name: lr_objective
            Description: This method sets the objective function for logistic regression model by setting various hyperparameters, including pipeline steps for different Optuna trials.
            Output: Single floating point value that represents f1 score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
            - continuous_columns: List of continuous variable names
            - continuous_index: List of continuous variable index
            - categorical_columns: List of categorical variable names
            - categorical_index: List of categorical variable index
        '''
        C = trial.suggest_float('C',0.0001,1,log=True)
        class_weight = trial.suggest_categorical(
            'class_weight',['balanced','None'])
        class_weight = None if class_weight == 'None' else class_weight
        penalty = trial.suggest_categorical('penalty',['l1','l2'])
        max_iter = trial.suggest_categorical('max_iter',[100000])
        solver = trial.suggest_categorical('solver',['saga'])
        dual = trial.suggest_categorical('dual',[False])
        n_jobs = trial.suggest_categorical('n_jobs',[2])
        clf = LogisticRegression(
            C=C, max_iter=max_iter, random_state=random_state, 
            class_weight = class_weight, penalty=penalty,dual=dual, solver=solver, n_jobs=n_jobs)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(
            pipeline, trial, continuous_columns, continuous_index,categorical_columns, categorical_index, clf)
        cv_results = model_trainer.classification_metrics(
            clf,pipeline,X_train_data,y_train_data,cv_jobs=3)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_f1_score'])


    def svc_objective(
            trial,X_train_data,y_train_data, continuous_columns, continuous_index, categorical_columns, categorical_index):
        '''
            Method Name: svc_objective
            Description: This method sets the objective function for linear support vector classifier model by setting various hyperparameters, including pipeline steps for different Optuna trials.
            Output: Single floating point value that represents f1 score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
            - continuous_columns: List of continuous variable names
            - continuous_index: List of continuous variable index
            - categorical_columns: List of categorical variable names
            - categorical_index: List of categorical variable index
        '''
        C = trial.suggest_float('C',0.0001,1,log=True)
        class_weight = trial.suggest_categorical(
            'class_weight',['balanced','None'])
        class_weight = None if class_weight == 'None' else class_weight
        penalty = trial.suggest_categorical('penalty',['l1','l2'])
        dual = trial.suggest_categorical('dual',[False])
        clf = LinearSVC(
            C=C, random_state=random_state, dual=dual,penalty=penalty, class_weight=class_weight)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(
            pipeline, trial, continuous_columns, continuous_index,categorical_columns, categorical_index, clf)
        cv_results = model_trainer.classification_metrics(
            clf,pipeline,X_train_data,y_train_data, cv_jobs=3)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_f1_score'])


    def knn_objective(
            trial,X_train_data,y_train_data, continuous_columns, continuous_index, categorical_columns, categorical_index):
        '''
            Method Name: knn_objective
            Description: This method sets the objective function for K-neighbors classifier model by setting various hyperparameters, including pipeline steps for different Optuna trials.
            Output: Single floating point value that represents f1 score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
            - continuous_columns: List of continuous variable names
            - continuous_index: List of continuous variable index
            - categorical_columns: List of categorical variable names
            - categorical_index: List of categorical variable index
        '''
        n_neighbors = trial.suggest_categorical('n_neighbors', [3, 5, 7, 9, 11])
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        leaf_size = trial.suggest_int('leaf_size',10,50)
        p = trial.suggest_int('p',1,4)
        n_jobs = trial.suggest_categorical('n_jobs', [2])
        clf = KNeighborsClassifier(
            n_neighbors=n_neighbors,weights=weights,leaf_size=leaf_size,p=p,n_jobs=n_jobs)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(
            pipeline, trial, continuous_columns, continuous_index,categorical_columns, categorical_index, clf)
        cv_results = model_trainer.classification_metrics(
            clf,pipeline,X_train_data,y_train_data, cv_jobs=3)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_f1_score'])


    def dt_objective(
            trial,X_train_data,y_train_data, continuous_columns, continuous_index, categorical_columns, categorical_index, path):
        '''
            Method Name: dt_objective
            Description: This method sets the objective function for Decision Tree classifier model by setting various hyperparameters, including pipeline steps for different Optuna trials using post pruning.
            Output: Single floating point value that represents f1 score of given model on validation set from using 3 fold cross validation
    
            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
            - continuous_columns: List of continuous variable names
            - continuous_index: List of continuous variable index
            - categorical_columns: List of categorical variable names
            - categorical_index: List of categorical variable index
            - path: Dictionary object, with attributes ccp_alphas and impurities
        '''
        ccp_alpha = trial.suggest_categorical('ccp_alpha',path.ccp_alphas[:-1])
        class_weight = trial.suggest_categorical(
            'class_weight',['balanced','None'])
        class_weight = None if class_weight == 'None' else class_weight
        clf = DecisionTreeClassifier(
            random_state=random_state, class_weight=class_weight, ccp_alpha=ccp_alpha)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(
            pipeline, trial, continuous_columns, continuous_index,categorical_columns, categorical_index, clf)
        cv_results = model_trainer.classification_metrics(
            clf,pipeline,X_train_data,y_train_data, cv_jobs=3)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_f1_score'])


    def rf_objective(
            trial,X_train_data,y_train_data, continuous_columns, continuous_index, categorical_columns, categorical_index, path):
        '''
            Method Name: rf_objective
            Description: This method sets the objective function for Random Forest classifier model by setting various hyperparameters, including pipeline steps for different Optuna trials using post pruning.
            Output: Single floating point value that represents f1 score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
            - continuous_columns: List of continuous variable names
            - continuous_index: List of continuous variable index
            - categorical_columns: List of categorical variable names
            - categorical_index: List of categorical variable index
            - path: Dictionary object, with attributes ccp_alphas and impurities
        '''
        ccp_alpha = trial.suggest_categorical('ccp_alpha',path.ccp_alphas[:-1])
        class_weight = trial.suggest_categorical(
            'class_weight', ['balanced', 'balanced_subsample','None'])
        class_weight = None if class_weight == 'None' else class_weight
        n_jobs = trial.suggest_categorical('n_jobs',[2])
        n_estimators = trial.suggest_categorical('n_estimators',[100])
        clf = RandomForestClassifier(
            random_state=random_state, class_weight=class_weight, ccp_alpha=ccp_alpha, n_jobs=n_jobs, n_estimators=n_estimators)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(
            pipeline, trial, continuous_columns, continuous_index,categorical_columns, categorical_index, clf)
        cv_results = model_trainer.classification_metrics(
            clf,pipeline,X_train_data,y_train_data, cv_jobs=3)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_f1_score'])


    def et_objective(
            trial,X_train_data,y_train_data, continuous_columns, continuous_index, categorical_columns, categorical_index, path):
        '''
            Method Name: et_objective
            Description: This method sets the objective function for Extra Trees classifier model by setting various hyperparameters, including pipeline steps for different Optuna trials using post pruning.
            Output: Single floating point value that represents f1 score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
            - continuous_columns: List of continuous variable names
            - continuous_index: List of continuous variable index
            - categorical_columns: List of categorical variable names
            - categorical_index: List of categorical variable index
            - path: Dictionary object, with attributes ccp_alphas and impurities
        '''
        ccp_alpha = trial.suggest_categorical('ccp_alpha',path.ccp_alphas[:-1])
        class_weight = trial.suggest_categorical('class_weight', 
            ['balanced', 'balanced_subsample','None'])
        class_weight = None if class_weight == 'None' else class_weight
        n_jobs = trial.suggest_categorical('n_jobs',[2])
        n_estimators = trial.suggest_categorical('n_estimators',[100])
        clf = ExtraTreesClassifier(
            random_state=random_state, class_weight=class_weight, ccp_alpha=ccp_alpha, n_jobs=n_jobs, n_estimators=n_estimators)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(
            pipeline, trial, continuous_columns, continuous_index,categorical_columns, categorical_index, clf)
        cv_results = model_trainer.classification_metrics(
            clf,pipeline,X_train_data,y_train_data, cv_jobs=3)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_f1_score'])


    def gaussiannb_objective(
            trial,X_train_data,y_train_data, continuous_columns, continuous_index, categorical_columns, categorical_index):
        '''
            Method Name: gaussiannb_objective
            Description: This method sets the objective function for Gaussian Naive Bayes model by setting various hyperparameters, including pipeline steps for different Optuna trials.
            Output: Single floating point value that represents f1 score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
            - continuous_columns: List of continuous variable names
            - continuous_index: List of continuous variable index
            - categorical_columns: List of categorical variable names
            - categorical_index: List of categorical variable index
        '''
        var_smoothing = trial.suggest_float(
            'var_smoothing',0.000000001,1,log=True)
        clf = GaussianNB(var_smoothing=var_smoothing)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(
            pipeline, trial, continuous_columns, continuous_index,categorical_columns, categorical_index, clf)
        cv_results = model_trainer.classification_metrics(
            clf,pipeline,X_train_data,y_train_data, cv_jobs=3)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_f1_score'])


    def adaboost_objective(
            trial,X_train_data,y_train_data, continuous_columns, continuous_index, categorical_columns, categorical_index):
        '''
            Method Name: adaboost_objective
            Description: This method sets the objective function for AdaBoost model by setting various hyperparameters, including pipeline steps for different Optuna trials.
            Output: Single floating point value that represents f1 score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
            - continuous_columns: List of continuous variable names
            - continuous_index: List of continuous variable index
            - categorical_columns: List of categorical variable names
            - categorical_index: List of categorical variable index
        '''
        learning_rate = trial.suggest_float('learning_rate',0.01,1,log=True)
        n_estimators = trial.suggest_categorical('n_estimators',[100])
        clf = AdaBoostClassifier(
            learning_rate=learning_rate, random_state=random_state, n_estimators=n_estimators)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(
            pipeline, trial, continuous_columns, continuous_index,categorical_columns, categorical_index, clf)
        cv_results = model_trainer.classification_metrics(
            clf,pipeline,X_train_data,y_train_data, cv_jobs=3)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_f1_score'])


    def gradientboost_objective(
            trial,X_train_data,y_train_data, continuous_columns, continuous_index, categorical_columns, categorical_index, path):
        '''
            Method Name: gradientboost_objective
            Description: This method sets the objective function for Gradient Boosting classifier model by setting various hyperparameters, including pipeline steps for different Optuna trials using post pruning.
            Output: Single floating point value that represents f1 score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
            - continuous_columns: List of continuous variable names
            - continuous_index: List of continuous variable index
            - categorical_columns: List of categorical variable names
            - categorical_index: List of categorical variable index
            - path: Dictionary object, with attributes ccp_alphas and impurities
        '''
        ccp_alpha = trial.suggest_categorical('ccp_alpha',path.ccp_alphas[:-1])
        loss = trial.suggest_categorical('loss',['log_loss','exponential'])
        learning_rate = trial.suggest_float('learning_rate',0.01,0.3,log=True)
        n_estimators = trial.suggest_categorical('n_estimators',[100])
        subsample = trial.suggest_float('subsample',0.5,1,log=True)
        max_features = trial.suggest_categorical('max_features',['sqrt'])      
        clf = GradientBoostingClassifier(
            random_state=random_state, loss=loss, ccp_alpha=ccp_alpha,
            n_estimators=n_estimators, subsample=subsample, learning_rate=learning_rate, max_features=max_features)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(
            pipeline, trial, continuous_columns, continuous_index,categorical_columns, categorical_index, clf)
        cv_results = model_trainer.classification_metrics(
            clf,pipeline,X_train_data,y_train_data, cv_jobs=3)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_f1_score'])


    def xgboost_objective(
            trial,X_train_data,y_train_data, continuous_columns, continuous_index, categorical_columns, categorical_index):
        '''
            Method Name: xgboost_objective
            Description: This method sets the objective function for XGBoost model by setting various hyperparameters, including pipeline steps for different Optuna trials.
            Output: Single floating point value that represents f1 score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
            - continuous_columns: List of continuous variable names
            - continuous_index: List of continuous variable index
            - categorical_columns: List of categorical variable names
            - categorical_index: List of categorical variable index
        '''
        smallest_class_count = y_train_data.values.sum()
        largest_class_count = len(y_train_data) - smallest_class_count
        spw = largest_class_count / smallest_class_count
        booster = trial.suggest_categorical('booster',['gbtree','dart'])
        rate_drop = trial.suggest_float('rate_drop',0.0001,1,log=True) if booster == 'dart' else None
        scale_pos_weight = trial.suggest_categorical('scale_pos_weight',[1,spw])
        eta = trial.suggest_float('eta',0.1,0.5,log=True)
        gamma = trial.suggest_float('gamma',0.1,20,log=True)
        min_child_weight = trial.suggest_float(
            'min_child_weight',0.1,1000,log=True)
        max_depth = trial.suggest_int('max_depth',1,10)
        lambdas = trial.suggest_float('lambda',0.1,1000,log=True)
        alpha = trial.suggest_float('alpha',0.1,100,log=True)
        subsample = trial.suggest_float('subsample',0.5,1,log=True)
        colsample_bytree = trial.suggest_float(
            'colsample_bytree',0.5,1,log=True)
        num_round = trial.suggest_categorical('num_round',[100])
        objective = trial.suggest_categorical('objective',['binary:logistic'])
        eval_metric = trial.suggest_categorical('eval_metric',['aucpr'])
        verbosity = trial.suggest_categorical('verbosity',[0])
        tree_method = trial.suggest_categorical('tree_method',['gpu_hist'])
        single_precision_histogram = trial.suggest_categorical(
            'single_precision_histogram',[True])
        clf = XGBClassifier(
            objective=objective, eval_metric=eval_metric, verbosity=verbosity,tree_method = tree_method, booster=booster, eta=eta, gamma=gamma,single_precision_histogram=single_precision_histogram,  min_child_weight=min_child_weight, max_depth=max_depth,scale_pos_weight=scale_pos_weight, subsample=subsample,colsample_bytree=colsample_bytree, lambdas=lambdas, alpha=alpha, random_state=random_state, num_round=num_round, rate_drop=rate_drop)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(
            pipeline, trial, continuous_columns, continuous_index,categorical_columns, categorical_index, clf)
        cv_results = model_trainer.classification_metrics(
            clf,pipeline,X_train_data,y_train_data, cv_jobs=1)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_f1_score'])


    def lightgbm_objective(
            trial,X_train_data,y_train_data, continuous_columns, continuous_index, categorical_columns, categorical_index):
        '''
            Method Name: lightgbm_objective
            Description: This method sets the objective function for LightGBM model by setting various hyperparameters, including pipeline steps for different Optuna trials.
            Output: Single floating point value that represents f1 score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
            - continuous_columns: List of continuous variable names
            - continuous_index: List of continuous variable index
            - categorical_columns: List of categorical variable names
            - categorical_index: List of categorical variable index
        '''
        is_unbalance = trial.suggest_categorical(
            'is_unbalance',['true', 'false'])
        learning_rate = trial.suggest_float('learning_rate',0.01,0.3,log=True)
        max_depth = trial.suggest_int('max_depth',3,12)
        num_leaves = trial.suggest_int('num_leaves',8,4096)
        min_child_samples = trial.suggest_int('min_child_samples',5,100)
        boosting_type = trial.suggest_categorical(
            'boosting_type',['gbdt','dart'])
        drop_rate = trial.suggest_float('drop_rate',0.0001,1,log=True) if boosting_type == 'dart' else None
        subsample = trial.suggest_float('subsample',0.5,1,log=True)
        subsample_freq = trial.suggest_int('subsample_freq',1,10)
        reg_alpha = trial.suggest_float('reg_alpha',0.1,100,log=True)
        reg_lambda = trial.suggest_float('reg_lambda',0.1,100,log=True)
        min_split_gain = trial.suggest_float('min_split_gain',0.1,15,log=True)
        max_bin = trial.suggest_categorical("max_bin", [63])
        n_estimators = trial.suggest_categorical('n_estimators',[100])
        device_type = trial.suggest_categorical('device_type',['gpu'])
        gpu_use_dp = trial.suggest_categorical('gpu_use_dp',[False])
        clf = LGBMClassifier(
            num_leaves=num_leaves, learning_rate=learning_rate, is_unbalance = is_unbalance, boosting_type=boosting_type, max_depth=max_depth, min_child_samples = min_child_samples, max_bin=max_bin, reg_alpha=reg_alpha, reg_lambda=reg_lambda, subsample = subsample, subsample_freq = subsample_freq, min_split_gain=min_split_gain, random_state=random_state, n_estimators=n_estimators, device_type=device_type,gpu_use_dp=gpu_use_dp, drop_rate=drop_rate, drop_seed = random_state)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(
            pipeline, trial, continuous_columns, continuous_index,categorical_columns, categorical_index, clf)
        cv_results = model_trainer.classification_metrics(
            clf,pipeline,X_train_data,y_train_data, cv_jobs=1)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_f1_score'])


    def catboost_objective(
            trial,X_train_data,y_train_data, continuous_columns, continuous_index, categorical_columns, categorical_index):
        '''
            Method Name: catboost_objective
            Description: This method sets the objective function for CatBoost model by setting various hyperparameters, including pipeline steps for different Optuna trials.
            Output: Single floating point value that represents f1 score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
            - continuous_columns: List of continuous variable names
            - continuous_index: List of continuous variable index
            - categorical_columns: List of categorical variable names
            - categorical_index: List of categorical variable index
        '''
        max_depth = trial.suggest_int('max_depth',4,10)
        l2_leaf_reg = trial.suggest_int('l2_leaf_reg',2,10)
        random_strength = trial.suggest_float('random_strength',0.1,10,log=True)
        auto_class_weights = trial.suggest_categorical(
            'auto_class_weights',['None', 'Balanced', 'SqrtBalanced'])
        auto_class_weights = None if auto_class_weights == 'None' else auto_class_weights
        learning_rate = trial.suggest_float('learning_rate',0.01,0.3,log=True)
        boosting_type = trial.suggest_categorical('boosting_type',['Plain'])
        loss_function = trial.suggest_categorical('loss_function',['Logloss'])
        nan_mode = trial.suggest_categorical('nan_mode',['Min'])
        task_type = trial.suggest_categorical('task_type',['GPU'])
        iterations = trial.suggest_categorical('iterations',[100])
        verbose = trial.suggest_categorical('verbose',[False])
        clf = CatBoostClassifier(
            max_depth = max_depth, l2_leaf_reg = l2_leaf_reg, learning_rate=learning_rate, random_strength=random_strength,  auto_class_weights=auto_class_weights, boosting_type = boosting_type, loss_function=loss_function,nan_mode=nan_mode,random_state=random_state,task_type=task_type, iterations=iterations, verbose=verbose)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(
            pipeline, trial, continuous_columns, continuous_index,categorical_columns, categorical_index, clf)
        cv_results = model_trainer.classification_metrics(
            clf,pipeline,X_train_data,y_train_data, cv_jobs=1)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_f1_score'])


    def classification_metrics(clf,pipeline,X_train_data,y_train_data, cv_jobs):
        '''
            Method Name: classification_metrics
            Description: This method performs 3-fold cross validation on the training set and performs model evaluation on the validation set.
            Output: Dictionary of metric scores from 3-fold cross validation.

            Parameters:
            - clf: Model object
            - pipeline: imblearn pipeline object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
            - cv_jobs: Number of cross validation jobs to run in parallel
        '''
        pipeline_copy = clone(pipeline)
        pipeline_copy.steps.append(('clf',clf))
        cv_results = cross_validate(
            pipeline_copy, X_train_data, y_train_data, cv=3,
            scoring={"balanced_accuracy": make_scorer(balanced_accuracy_score),
            "precision_score": make_scorer(precision_score),
            "recall_score": make_scorer(recall_score),
            "f1_score": make_scorer(f1_score),
            "matthews_corrcoef": make_scorer(matthews_corrcoef),
            "average_precision_score": make_scorer(average_precision_score)},n_jobs=cv_jobs,return_train_score=True)
        return cv_results


    def optuna_optimizer(self, obj, n_trials, fold):
        '''
            Method Name: optuna_optimizer
            Description: This method creates a new Optuna study object and optimizes the given objective function. In addition, the following plots and results are also created and saved:
            1. Hyperparameter Importance Plot
            2. Optimization History Plot
            3. Optuna study object
            4. Optimization Results (csv format)
            
            Output: Single best trial object
            On Failure: Logging error and raise exception

            Parameters:
            - obj: Optuna objective function
            - n_trials: Number of trials for Optuna hyperparameter tuning
            - fold: Fold number from nested cross-validation in outer loop
        '''
        try:
            sampler = optuna.samplers.TPESampler(
                multivariate=True, seed=random_state)
            study = optuna.create_study(direction='maximize',sampler=sampler)
            study.optimize(
                obj, n_trials=n_trials, gc_after_trial=True, show_progress_bar=True)
            trial = study.best_trial
            if trial.number !=0:
                param_imp_fig = optuna.visualization.plot_param_importances(study)
                opt_fig = optuna.visualization.plot_optimization_history(study)
                param_imp_fig.write_image(
                    self.folderpath+ obj.__name__ +f'/HP_Importances_{obj.__name__}_Fold_{fold}.png')
                opt_fig.write_image(
                    self.folderpath+ obj.__name__ +f'/Optimization_History_{obj.__name__}_Fold_{fold}.png')
            joblib.dump(
                study, self.folderpath + obj.__name__ + f'/OptStudy_{obj.__name__}_Fold_{fold}.pkl')
            study.trials_dataframe().to_csv(
                self.folderpath + obj.__name__ + f"/Hyperparameter_Tuning_Results_{obj.__name__}_Fold_{fold}.csv",index=False)
            del study
        except Exception as e:
            self.log_writer.log(
                self.file_object, f'Performing optuna hyperparameter tuning for {obj.__name__} model failed with the following error: {e}')
            raise Exception(
                f'Performing optuna hyperparameter tuning for {obj.__name__} model failed with the following error: {e}')
        return trial

    
    def confusion_matrix_plot(
            self, clf, figtitle, plotname, actual_labels, pred_labels):
        '''
            Method Name: confusion_matrix_plot
            Description: This method plots confusion matrix and saves plot within the given model class folder.
            Output: None

            Parameters:
            - clf: Model object
            - figtitle: String that represents part of title figure
            - plotname: String that represents part of image name
            - actual_labels: Actual target labels from dataset
            - pred_labels: Predicted target labels from model
        '''
        cmd = ConfusionMatrixDisplay.from_predictions(
            actual_labels, pred_labels)
        cmd.ax_.set_title(f"{type(clf).__name__} {figtitle}")
        plt.grid(False)
        cmd.figure_.savefig(
            self.folderpath+type(clf).__name__+f'/Confusion_Matrix_{type(clf).__name__}_{plotname}.png')
        plt.clf()


    def classification_report_plot(
            self, clf, figtitle, plotname, actual_labels, pred_labels):
        '''
            Method Name: classification_report_plot
            Description: This method plots classification report in heatmap form and saves plot within the given model class folder.
            Output: None

            Parameters:
            - clf: Model object
            - figtitle: String that represents part of title figure
            - plotname: String that represents part of image name
            - actual_labels: Actual target labels from dataset
            - pred_labels: Predicted target labels from model
        '''
        clf_report = classification_report(
            actual_labels,pred_labels,target_names=['0','1'],output_dict=True,digits=4)
        fig = plt.figure()
        sns.heatmap(
            pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, fmt=".4f")
        plt.title(f"{type(clf).__name__} {figtitle}")
        fig.savefig(
            self.folderpath+type(clf).__name__+f'/Classification_Report_{type(clf).__name__}_{plotname}.png')
        plt.clf()


    def precision_recall_plot(self, clf, actual_labels, pred_proba):
        '''
            Method Name: precision_recall_plot
            Description: This method plots precision recall curve and saves plot within the given model class folder.
            Output: None

            Parameters:
            - clf: Model object
            - actual_labels: Actual target labels from dataset
            - pred_proba: Predicted probability of target being positive (1) from model
        '''
        display = PrecisionRecallDisplay.from_predictions(
            actual_labels, pred_proba, name= type(clf).__name__)
        display.ax_.set_title(f"{type(clf).__name__} Precision-Recall curve")
        display.figure_.savefig(
            self.folderpath+type(clf).__name__+f'/PrecisionRecall_Curve_{type(clf).__name__}.png')
        plt.clf()


    def binary_threshold_plot(self, clf, input_data, output_data):
        '''
            Method Name: binary_threshold_plot
            Description: This method plots discrimination threshold for binary classification and saves plot within the given model class folder.
            Note that this function will not work for CatBoost, since DiscriminationThreshold function from yellowbricks.classifier module is not yet supported for this model class.
            Output: None

            Parameters:
            - clf: Model object
            - input_data: Features from dataset
            - output_data: Target column from dataset
        '''
        if type(clf).__name__ not in ['CatBoostClassifier']:
            visualizer = DiscriminationThreshold(
                clf, random_state=random_state, cv= StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state))
            visualizer.fit(input_data,output_data)
            visualizer.show(outpath=self.folderpath+type(clf).__name__+f'/Binary_Threshold_{type(clf).__name__}.png',clear_figure=True)
            joblib.dump(visualizer, 'Saved_Models/Binary_Threshold.pkl')


    def learning_curve_plot(self, clf, input_data, output_data):
        '''
            Method Name: learning_curve_plot
            Description: This method plots learning curve of 5 fold cross validation and saves plot within the given model class folder.
            Output: None

            Parameters:
            - clf: Model object
            - input_data: Features from dataset
            - output_data: Target column from dataset
        '''
        if type(clf).__name__ == 'CatBoostClassifier':
            train_sizes, train_scores, validation_scores = learning_curve(estimator = clf, X = input_data, y = output_data, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state), scoring='f1', train_sizes=np.linspace(0.3, 1.0, 10))
            plt.style.use('seaborn-whitegrid')
            plt.grid(True)
            plt.fill_between(train_sizes, train_scores.mean(axis = 1) - train_scores.std(axis = 1), train_scores.mean(axis = 1) + train_scores.std(axis = 1), alpha=0.25, color='blue')
            plt.plot(train_sizes, train_scores.mean(axis = 1), label = 'Training Score', marker='.',markersize=14)
            plt.fill_between(train_sizes, validation_scores.mean(axis = 1) - validation_scores.std(axis = 1), validation_scores.mean(axis = 1) + validation_scores.std(axis = 1), alpha=0.25, color='green')
            plt.plot(train_sizes, validation_scores.mean(axis = 1), label = 'Cross Validation Score', marker='.',markersize=14)
            plt.ylabel('Score')
            plt.xlabel('Training instances')
            plt.title(f'Learning Curve for {type(clf).__name__}')
            plt.legend(frameon=True, loc='best')
            plt.savefig(
                self.folderpath+type(clf).__name__+f'/LearningCurve_{type(clf).__name__}.png',bbox_inches='tight')
            plt.clf()
        else:
            visualizer = LearningCurve(
                clf, cv= StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state), scoring='f1', train_sizes=np.linspace(0.3, 1.0, 10))
            visualizer.fit(input_data,output_data)
            visualizer.show(
                outpath=self.folderpath+type(clf).__name__+f'/LearningCurve_{type(clf).__name__}.png',clear_figure=True)


    def shap_plot(self, clf, input_data):
        '''
            Method Name: shap_plot
            Description: This method plots feature importance and its summary using shap values and saves plot within the given model class folder. Note that this function will not work specifically for XGBoost models that use 'dart' booster. In addition, shap plots for KNeighbors and GaussianNB require use of shap's Kernel explainer that involves high computational time. Thus, this function excludes both KNeighbors and GaussianNB.
            Output: None

            Parameters:
            - clf: Model object
            - input_data: Features from dataset
        '''
        if (type(clf).__name__ not in ['KNeighborsClassifier','GaussianNB']):
            if type(clf).__name__ in ['LogisticRegression','LinearSVC']:
                explainer = shap.LinearExplainer(clf, input_data)
                explainer_obj = explainer(input_data)
                shap_values = explainer.shap_values(input_data)
            else:
                if ('dart' in clf.get_params().values()) and (type(clf).__name__ == 'XGBClassifier'):
                    return
                explainer = shap.TreeExplainer(clf)
                if type(clf).__name__ in ['GradientBoostingClassifier','XGBClassifier','CatBoostClassifier']:
                    explainer_obj = explainer(input_data)
                    shap_values = explainer.shap_values(input_data)
                else:
                    explainer_obj = explainer(input_data)[:,:,1]
                    shap_values = explainer.shap_values(input_data)[1]
            plt.figure()
            shap.summary_plot(
                shap_values, input_data, plot_type="bar", show=False, max_display=40)
            plt.title(f'Shap Feature Importances for {type(clf).__name__}')
            plt.savefig(
                self.folderpath+type(clf).__name__+f'/Shap_Feature_Importances_{type(clf).__name__}.png',bbox_inches='tight')
            plt.clf()
            plt.figure()
            shap.plots.beeswarm(explainer_obj, show=False, max_display=40)
            plt.title(f'Shap Summary Plot for {type(clf).__name__}')
            plt.savefig(
                self.folderpath+type(clf).__name__+f'/Shap_Summary_Plot_{type(clf).__name__}.png',bbox_inches='tight')
            plt.clf()


    def model_training(
            self, clf, obj, continuous_columns, continuous_index, categorical_columns, categorical_index, input_data, output_data, n_trials, fold_num):
        '''
            Method Name: model_training
            Description: This method performs Optuna hyperparameter tuning using 3 fold cross validation on given dataset. The best hyperparameters with the best pipeline identified is used for model training.
            
            Output: 
            - model_copy: Trained model object
            - best_trial: Optuna's best trial object from hyperparameter tuning
            - input_data_transformed: Transformed features from dataset
            - output_data_transformed: Transformed target column from dataset
            - best_pipeline: imblearn pipeline object

            On Failure: Logging error and raise exception

            Parameters:
            - clf: Model object
            - obj: Optuna objective function
            - continuous_columns: List of continuous variable names
            - continuous_index: List of continuous variable index
            - categorical_columns: List of categorical variable names
            - categorical_index: List of categorical variable index
            - input_data: Features from dataset
            - output_data: Target column from dataset
            - n_trials: Number of trials for Optuna hyperparameter tuning
            - fold_num: Indication of fold number for model training (can be integer or string "overall")
        '''
        if type(clf).__name__ in ['DecisionTreeClassifier', 'RandomForestClassifier','ExtraTreesClassifier','GradientBoostingClassifier']:
            temp_pipeline = Pipeline(steps=[], memory='Caching')
            model_trainer.pipeline_missing_step(
                temp_pipeline, continuous_columns, continuous_index)
            X_train_data_copy = temp_pipeline.steps[0][1].fit_transform(input_data,output_data)
            temp_clf = DecisionTreeClassifier(random_state=random_state)
            path = temp_clf.cost_complexity_pruning_path(X_train_data_copy, output_data)
            func = lambda trial: obj(
                trial, input_data, output_data, continuous_columns, continuous_index, categorical_columns, categorical_index, path)
        else:
            func = lambda trial: obj(
                trial, input_data, output_data, continuous_columns, continuous_index, categorical_columns, categorical_index)
        func.__name__ = type(clf).__name__
        self.log_writer.log(
            self.file_object, f"Start hyperparameter tuning for {type(clf).__name__} for fold {fold_num}")
        best_trial = self.optuna_optimizer(func, n_trials, fold_num)
        self.log_writer.log(
            self.file_object, f"Hyperparameter tuning for {type(clf).__name__} completed for fold {fold_num}")
        self.log_writer.log(
            self.file_object, f"Start using best pipeline for {type(clf).__name__} for transforming training and validation data for fold {fold_num}")
        best_pipeline = best_trial.user_attrs['Pipeline']
        input_data_transformed = best_pipeline.fit_transform(
            input_data, output_data)
        if 'smote' in best_pipeline.named_steps.keys():
            output_data_transformed = best_pipeline.steps[1][1].fit_resample(best_pipeline.steps[0][1].fit_transform(input_data, output_data), output_data)[1]
        else:
            output_data_transformed = output_data
        self.log_writer.log(
            self.file_object, f"Finish using best pipeline for {type(clf).__name__} for transforming training and validation data for fold {fold_num}")
        for parameter in ['missing','balancing','outlier','scaling','feature_selection','number_features','drop_correlated','drop_correlated_missing','feature_selection_missing','damping','cluster_indicator']:
            if parameter in best_trial.params.keys():
                best_trial.params.pop(parameter)
        for weight_param in ['class_weight','auto_class_weights']:
            if weight_param in best_trial.params.keys():
                if best_trial.params[weight_param] == 'None':
                    best_trial.params.pop(weight_param)
        self.log_writer.log(
            self.file_object, f"Start evaluating model performance for {type(clf).__name__} on validation set for fold {fold_num}")
        model_copy = clone(clf)
        model_copy = model_copy.set_params(**best_trial.params)
        model_copy.fit(
            input_data_transformed, output_data_transformed['Output'])
        return model_copy, best_trial, input_data_transformed, output_data_transformed, best_pipeline


    def hyperparameter_tuning(
            self, obj, clf, n_trials, input_data, output_data, continuous_columns, continuous_index, categorical_columns, categorical_index):
        '''
            Method Name: hyperparameter_tuning
            Description: This method performs Stratified Nested 3 Fold Cross Validation on the entire dataset, where the inner loop (3-fold) performs Optuna hyperparameter tuning and the outer loop (5-fold) performs model evaluation to obtain overall generalization error of model. The best hyperparameters with the best pipeline identified from inner loop is used for model training on the entire training set and model evaluation on the test set for the outer loop.
            In addition, the following intermediate results are saved for a given model class:
            1. Model_Performance_Results_by_Fold (csv file)
            2. Overall_Model_Performance_Results (csv file)
            3. Confusion Matrix with default threshold (0.5) image
            4. Classification Report heatmap with default threshold (0.5) image
            5. Precision-Recall curve image
            
            Output: None
            On Failure: Logging error and raise exception

            Parameters:
            - obj: Optuna objective function
            - clf: Model object
            - n_trials: Number of trials for Optuna hyperparameter tuning
            - input_data: Features from dataset
            - output_data: Target column from dataset
            - continuous_columns: List of continuous variable names
            - continuous_index: List of continuous variable index
            - categorical_columns: List of categorical variable names
            - categorical_index: List of categorical variable index
        '''
        try:
            num_folds = 5
            skfold = StratifiedKFold(
                n_splits=num_folds, shuffle=True, random_state=random_state)
            bal_accuracy_train_cv, precision_train_cv, ap_train_cv = [], [], []
            recall_train_cv, f1_train_cv, mc_train_cv = [], [], []
            bal_accuracy_val_cv, precision_val_cv, recall_val_cv = [], [], []
            mc_val_cv, ap_val_cv, f1_val_cv = [], [], []
            bal_accuracy_test_cv, precision_test_cv, recall_test_cv = [], [], []
            mc_test_cv, ap_test_cv, f1_test_cv = [], [], []
            actual_labels, pred_labels, pred_proba = [], [], []
            for fold, (outer_train_idx, outer_valid_idx) in enumerate(skfold.split(input_data, output_data)):
                input_sub_train_data = input_data.iloc[outer_train_idx,:].reset_index(drop=True)
                output_sub_train_data = output_data.iloc[outer_train_idx].reset_index(drop=True)
                model_copy, best_trial, input_train_data_transformed, output_train_data_transformed, best_pipeline = self.model_training(clf, obj, continuous_columns, continuous_index, categorical_columns, categorical_index, input_sub_train_data, output_sub_train_data, n_trials, fold+1)
                input_val_data = input_data.iloc[outer_valid_idx,:].reset_index(drop=True)
                input_val_data_transformed = best_pipeline.transform(input_val_data)
                val_pred = model_copy.predict(input_val_data_transformed)
                val_pred_proba = model_copy.predict_proba(input_val_data_transformed)[:,1] if type(model_copy).__name__ != 'LinearSVC' else model_copy._predict_proba_lr(input_val_data_transformed)[:,1]
                actual_labels.extend(output_data.iloc[outer_valid_idx]['Output'].tolist())
                pred_labels.extend(val_pred)
                pred_proba.extend(val_pred_proba)
                bal_acc_outer_val_value = balanced_accuracy_score(
                    np.array(output_data.iloc[outer_valid_idx]),val_pred)
                precision_outer_val_value = precision_score(
                    np.array(output_data.iloc[outer_valid_idx]),val_pred)
                f1_outer_val_value = f1_score(
                    np.array(output_data.iloc[outer_valid_idx]),val_pred)
                mc_outer_val_value = matthews_corrcoef(
                    np.array(output_data.iloc[outer_valid_idx]),val_pred)
                recall_outer_val_value = recall_score(
                    np.array(output_data.iloc[outer_valid_idx]),val_pred)
                ap_outer_val_value = average_precision_score(
                    np.array(output_data.iloc[outer_valid_idx]),val_pred_proba)
                bal_accuracy_train_cv.append(
                    best_trial.user_attrs['train_balanced_accuracy'])
                precision_train_cv.append(
                    best_trial.user_attrs['train_precision_score'])
                f1_train_cv.append(best_trial.user_attrs['train_f1_score'])
                mc_train_cv.append(
                    best_trial.user_attrs['train_matthews_corrcoef'])
                recall_train_cv.append(
                    best_trial.user_attrs['train_recall_score'])
                ap_train_cv.append(
                    best_trial.user_attrs['train_average_precision_score'])
                bal_accuracy_val_cv.append(
                    best_trial.user_attrs['val_balanced_accuracy'])
                precision_val_cv.append(
                    best_trial.user_attrs['val_precision_score'])
                f1_val_cv.append(best_trial.user_attrs['val_f1_score'])
                mc_val_cv.append(best_trial.user_attrs['val_matthews_corrcoef'])
                recall_val_cv.append(best_trial.user_attrs['val_recall_score'])
                ap_val_cv.append(
                    best_trial.user_attrs['val_average_precision_score'])
                bal_accuracy_test_cv.append(bal_acc_outer_val_value)
                precision_test_cv.append(precision_outer_val_value)
                f1_test_cv.append(f1_outer_val_value)
                mc_test_cv.append(mc_outer_val_value)
                recall_test_cv.append(recall_outer_val_value)
                ap_test_cv.append(ap_outer_val_value)
                self.log_writer.log(
                    self.file_object, f"Evaluating model performance for {type(clf).__name__} on validation set completed for fold {fold+1}")
                optimized_results = pd.DataFrame({
                    'Feature_selector':best_trial.user_attrs['feature_selection'], 'Drop_correlated_features': best_trial.user_attrs['drop_correlated'], 'Models': type(model_copy).__name__, 'Best_params': str(model_copy.get_params()), 'Cluster_Indicator': best_trial.user_attrs['cluster_indicator'], 'Damping_cluster_value': best_trial.user_attrs['damping'], 'Number_features': [len(input_train_data_transformed.columns.tolist())], 'Features': [input_train_data_transformed.columns.tolist()], 'Balancing_handled': best_trial.user_attrs['balancing_indicator'], 'Missing_values_handled': best_trial.user_attrs['missing_indicator'], 'Outlier_handling_method': best_trial.user_attrs['outlier_indicator'], 'Feature_scaling_handled': best_trial.user_attrs['scaling_indicator'], 'Outer_fold': fold+1,'bal_acc_inner_train_cv': best_trial.user_attrs['train_balanced_accuracy'],'bal_acc_inner_val_cv': best_trial.user_attrs['val_balanced_accuracy'],'bal_acc_outer_val_cv': [bal_acc_outer_val_value],'precision_inner_train_cv': best_trial.user_attrs['train_precision_score'],'precision_inner_val_cv': best_trial.user_attrs['val_precision_score'],'precision_outer_val_cv': [precision_outer_val_value],'recall_inner_train_cv': best_trial.user_attrs['train_recall_score'],'recall_inner_val_cv': best_trial.user_attrs['val_recall_score'],'recall_outer_val_cv': [recall_outer_val_value],'f1_inner_train_cv': best_trial.user_attrs['train_f1_score'],'f1_inner_val_cv': best_trial.user_attrs['val_f1_score'],'f1_outer_val_cv': [f1_outer_val_value],'mc_inner_train_cv': best_trial.user_attrs['train_matthews_corrcoef'],'mc_inner_val_cv': best_trial.user_attrs['val_matthews_corrcoef'],'mc_outer_val_cv': [mc_outer_val_value],'average_precision_inner_train_cv': best_trial.user_attrs['train_average_precision_score'],'average_precision_inner_val_cv': best_trial.user_attrs['val_average_precision_score'],'average_precision_outer_val_cv': [ap_outer_val_value]})
                optimized_results.to_csv(
                    self.folderpath+'Model_Performance_Results_by_Fold.csv', mode='a', index=False, header=not os.path.exists(self.folderpath+'Model_Performance_Results_by_Fold.csv'))
                self.log_writer.log(
                    self.file_object, f"Optimized results for {type(clf).__name__} model saved for fold {fold+1}")
                time.sleep(10)
            average_results = pd.DataFrame({
                'Models': type(model_copy).__name__, 'bal_acc_train_cv_avg': np.mean(bal_accuracy_train_cv), 'bal_acc_train_cv_std': np.std(bal_accuracy_train_cv), 'bal_acc_val_cv_avg': np.mean(bal_accuracy_val_cv), 'bal_acc_val_cv_std': np.std(bal_accuracy_val_cv), 'bal_acc_test_cv_avg': np.mean(bal_accuracy_test_cv), 'bal_acc_test_cv_std': np.std(bal_accuracy_test_cv), 'precision_train_cv_avg': np.mean(precision_train_cv), 'precision_train_cv_std': np.std(precision_train_cv), 'precision_val_cv_avg': np.mean(precision_val_cv), 'precision_val_cv_std': np.std(precision_val_cv), 'precision_test_cv_avg': np.mean(precision_test_cv), 'precision_test_cv_std': np.std(precision_test_cv), 'recall_train_cv_avg': np.mean(recall_train_cv), 'recall_train_cv_std': np.std(recall_train_cv), 'recall_val_cv_avg': np.mean(recall_val_cv),'recall_val_cv_std': np.std(recall_val_cv),'recall_test_cv_avg': np.mean(recall_test_cv),'recall_test_cv_std': np.std(recall_test_cv), 'f1_train_cv_avg': np.mean(f1_train_cv), 'f1_train_cv_std': np.std(f1_train_cv), 'f1_val_cv_avg': np.mean(f1_val_cv),'f1_val_cv_std': np.std(f1_val_cv), 'f1_test_cv_avg': np.mean(f1_test_cv), 'f1_test_cv_std': np.std(f1_test_cv),'mc_train_cv_avg': np.mean(mc_train_cv), 'mc_train_cv_std': np.std(mc_train_cv), 'mc_val_cv_avg': np.mean(mc_val_cv),'mc_val_cv_std': np.std(mc_val_cv), 'mc_test_cv_avg': np.mean(mc_test_cv), 'mc_test_cv_std': np.std(mc_test_cv),'average_precision_train_cv_avg': np.mean(ap_train_cv),'average_precision_train_cv_std': np.std(ap_train_cv),'average_precision_val_cv_avg': np.mean(ap_val_cv),'average_precision_val_cv_std': np.std(ap_val_cv),'average_precision_test_cv_avg': np.mean(ap_test_cv),'average_precision_test_cv_std': np.std(ap_test_cv)}, index=[0])
            average_results.to_csv(
                self.folderpath+'Overall_Model_Performance_Results.csv', mode='a', index=False, header=not os.path.exists(self.folderpath+'Overall_Model_Performance_Results.csv'))
            self.log_writer.log(
                self.file_object, f"Average optimized results for {type(clf).__name__} model saved")                
            self.confusion_matrix_plot(
                clf, 'Confusion Matrix (Threshold: 0.5)', 'Default_Threshold', actual_labels, pred_labels)
            self.classification_report_plot(
                clf, 'Classification Report (Threshold: 0.5)', 'Default_Threshold', actual_labels, pred_labels)
            self.precision_recall_plot(clf, actual_labels, pred_proba)
        except Exception as e:
            self.log_writer.log(
                self.file_object, f'Hyperparameter tuning on {type(clf).__name__} model failed with the following error: {e}')
            raise Exception(
                f'Hyperparameter tuning on {type(clf).__name__} model failed with the following error: {e}')


    def final_overall_model(
            self, obj, clf, input_data, output_data, n_trials, continuous_columns, continuous_index, categorical_columns, categorical_index):
        '''
            Method Name: final_overall_model
            Description: This method performs hyperparameter tuning on best model algorithm identified using stratified 3 fold cross validation on entire dataset. The best hyperparameters identified are then used to train the entire dataset before saving model for deployment.
            In addition, the following intermediate results are saved for a given model class:
            1. Confusion Matrix with default threshold (0.5) image
            2. Classification Report heatmap with default threshold (0.5) image
            3. Discrimination Threshold image
            4. Learning Curve image
            5. Shap Feature Importances (barplot image)
            6. Shap Summary Plot (beeswarm plot image)
            
            Output: None

            Parameters:
            - obj: Optuna objective function
            - clf: Model object
            - input_data: Features from dataset
            - output_data: Target column from dataset
            - n_trials: Number of trials for Optuna hyperparameter tuning
            - continuous_columns: List of continuous variable names
            - continuous_index: List of continuous variable index
            - categorical_columns: List of categorical variable names
            - categorical_index: List of categorical variable index
        '''
        self.log_writer.log(
            self.file_object, f"Start final model training on all data for {type(clf).__name__}")
        overall_model, best_trial, input_data_transformed, output_data_transformed, best_pipeline = self.model_training(clf, obj, continuous_columns, continuous_index, categorical_columns, categorical_index, input_data, output_data, n_trials, 'overall')
        joblib.dump(best_pipeline,'Saved_Models/Preprocessing_Pipeline.pkl')
        joblib.dump(overall_model,'Saved_Models/FinalModel.pkl')
        actual_labels = output_data_transformed['Output']
        pred_labels = overall_model.predict(input_data_transformed)
        self.confusion_matrix_plot(
            clf, 'Confusion Matrix (Threshold: 0.5) - Final Model', 'Default_Threshold_Final_Model', actual_labels, pred_labels)
        self.classification_report_plot(
            clf, 'Classification Report (Threshold: 0.5) - Final Model', 'Default_Threshold_Final_Model', actual_labels, pred_labels)
        self.binary_threshold_plot(
            overall_model, input_data_transformed, output_data_transformed['Output'])
        self.learning_curve_plot(
            overall_model, input_data_transformed, output_data_transformed['Output'])
        self.shap_plot(overall_model, input_data_transformed)
        self.log_writer.log(
            self.file_object, f"Finish final model training on all data for {type(clf).__name__}")
        

    def model_selection(self, input, output, num_trials, folderpath):
        '''
            Method Name: model_selection
            Description: This method performs model algorithm selection using Stratified Nested Cross Validation (5-fold cv outer loop for model evaluation and 3-fold cv inner loop for hyperparameter tuning)
            Output: None

            Parameters:
            - input: Features from dataset
            - output: Target column from dataset
            - num_trials: Number of Optuna trials for hyperparameter tuning
            - folderpath: String path name where all results generated from model training are stored.
        '''
        self.log_writer.log(
            self.file_object, 'Start process of model selection')
        self.input = input
        self.output = output
        self.num_trials = num_trials
        self.folderpath = folderpath
        optuna.logging.set_verbosity(optuna.logging.DEBUG)
        continuous_columns = [col for col in self.input.columns if col.find('_') == -1]
        categorical_columns = [col for col in self.input.columns if col not in continuous_columns]
        continuous_index = [self.input.columns.get_loc(col) for col in continuous_columns]
        categorical_index = [self.input.columns.get_loc(col) for col in categorical_columns]
        input_data = self.input.copy()
        output_data = self.output.copy()
        for selector in self.optuna_selectors.values():
            obj = selector['obj']
            clf = selector['clf']
            path = os.path.join(self.folderpath, type(clf).__name__)
            if not os.path.exists(path):
                os.mkdir(path)
            self.hyperparameter_tuning(
                obj = obj, clf = clf, n_trials = self.num_trials, input_data = input_data, output_data = output_data, continuous_columns = continuous_columns, continuous_index = continuous_index, categorical_columns = categorical_columns, categorical_index = categorical_index)
            time.sleep(10)
        overall_results = pd.read_csv(
            self.folderpath + 'Overall_Model_Performance_Results.csv')
        self.log_writer.log(
            self.file_object, f"Best model identified based on balanced accuracy score is {overall_results.iloc[overall_results['bal_acc_test_cv_avg'].idxmax()]['Models']} with the following test score: {np.round(overall_results.iloc[overall_results['bal_acc_test_cv_avg'].idxmax()]['bal_acc_test_cv_avg'],4)} ({np.round(overall_results.iloc[overall_results['bal_acc_test_cv_avg'].idxmax()]['bal_acc_test_cv_std'],4)})")
        self.log_writer.log(
            self.file_object, f"Best model identified based on precision score is {overall_results.iloc[overall_results['precision_test_cv_avg'].idxmax()]['Models']} with the following test score: {np.round(overall_results.iloc[overall_results['precision_test_cv_avg'].idxmax()]['precision_test_cv_avg'],4)} ({np.round(overall_results.iloc[overall_results['precision_test_cv_avg'].idxmax()]['precision_test_cv_std'],4)})")
        self.log_writer.log(
            self.file_object, f"Best model identified based on recall score is {overall_results.iloc[overall_results['recall_test_cv_avg'].idxmax()]['Models']} with the following test score: {np.round(overall_results.iloc[overall_results['recall_test_cv_avg'].idxmax()]['recall_test_cv_avg'],4)} ({np.round(overall_results.iloc[overall_results['recall_test_cv_avg'].idxmax()]['recall_test_cv_std'],4)})")
        self.log_writer.log(
            self.file_object, f"Best model identified based on f1 score is {overall_results.iloc[overall_results['f1_test_cv_avg'].idxmax()]['Models']} with the following test score: {np.round(overall_results.iloc[overall_results['f1_test_cv_avg'].idxmax()]['f1_test_cv_avg'],4)} ({np.round(overall_results.iloc[overall_results['f1_test_cv_avg'].idxmax()]['f1_test_cv_std'],4)})")
        self.log_writer.log(
            self.file_object, f"Best model identified based on matthews correlation coefficient is {overall_results.iloc[overall_results['mc_test_cv_avg'].idxmax()]['Models']} with the following test score: {np.round(overall_results.iloc[overall_results['mc_test_cv_avg'].idxmax()]['mc_test_cv_avg'],4)} ({np.round(overall_results.iloc[overall_results['mc_test_cv_avg'].idxmax()]['mc_test_cv_std'],4)})")
        self.log_writer.log(
            self.file_object, f"Best model identified based on average precision is {overall_results.iloc[overall_results['average_precision_test_cv_avg'].idxmax()]['Models']} with the following test score: {np.round(overall_results.iloc[overall_results['average_precision_test_cv_avg'].idxmax()]['average_precision_test_cv_avg'],4)} ({np.round(overall_results.iloc[overall_results['average_precision_test_cv_avg'].idxmax()]['average_precision_test_cv_std'],4)})")
        self.log_writer.log(
            self.file_object, 'Finish process of model selection')


    def final_model_tuning(
            self, input_data, output_data, num_trials, folderpath):
        '''
            Method Name: final_model_tuning
            Description: This method performs final model training from best model algorithm identified on entire dataset using Stratified 3-fold cross validation.
            Output: None

            Parameters:
            - input_data: Features from dataset
            - output_data: Target column from dataset
            - num_trials: Number of Optuna trials for hyperparameter tuning
            - folderpath: String path name where all results generated from model training are stored.
        '''
        self.input_data = input_data
        self.output_data = output_data
        self.num_trials = num_trials
        self.folderpath = folderpath
        optuna.logging.set_verbosity(optuna.logging.DEBUG)
        try:
            model_number = int(input("""
    Select one of the following models to use for model deployment: 
    [1] Logistic Regression
    [2] Linear SVC
    [3] K Neighbors Classifier
    [4] Gaussian Naive Bayes
    [5] Decision Tree Classifier
    [6] Random Forest Classifier
    [7] Extra Trees Classifier
    [8] Ada Boost Classifier
    [9] Gradient Boost Classifier
    [10] XGBoost Classifier
    [11] LGBM Classifier
    [12] CatBoost Classifier
            """))
            model_options = {1: 'LogisticRegression', 2: 'LinearSVC', 3: 'KNeighborsClassifier', 4: 'GaussianNB', 5: 'DecisionTreeClassifier', 6: 'RandomForestClassifier', 7: 'ExtraTreesClassifier', 8: 'AdaBoostClassifier', 9: 'GradientBoostingClassifier', 10: 'XGBClassifier', 11: 'LGBMClassifier', 12: 'CatBoostClassifier'}
            best_model_name = model_options[model_number]
        except:
            print(
                'Please insert a valid number of choice for model deployment.')
            return
        self.log_writer.log(
            self.file_object, f"Start performing hyperparameter tuning on best model identified overall: {best_model_name}")
        obj = self.optuna_selectors[best_model_name]['obj']
        clf = self.optuna_selectors[best_model_name]['clf']
        continuous_columns = [col for col in self.input_data.columns if col.find('_') == -1]
        categorical_columns = [col for col in self.input_data.columns if col not in continuous_columns]
        continuous_index = [self.input_data.columns.get_loc(col) for col in continuous_columns]
        categorical_index = [self.input_data.columns.get_loc(col) for col in categorical_columns]
        input_data = self.input_data.copy()
        output_data = self.output_data.copy()
        self.final_overall_model(
            obj = obj, clf = clf, input_data = input_data, output_data = output_data, n_trials = self.num_trials, continuous_columns = continuous_columns, continuous_index = continuous_index, categorical_columns = categorical_columns, categorical_index = categorical_index)
        self.log_writer.log(
            self.file_object, f"Finish performing hyperparameter tuning on best model identified overall: {best_model_name}")


class CheckGaussian():
    

    def __init__(self):
        '''
            Method Name: __init__
            Description: This method initializes instance of CheckGaussian class
            Output: None
        '''
        pass


    def check_gaussian(self, X):
        '''
            Method Name: check_gaussian
            Description: This method classifies features from dataset into gaussian vs non-gaussian columns.
            Output: self

            Parameters:
            - X: Features from dataset
        '''
        X_ = pd.DataFrame(X, columns = self.continuous)
        self.gaussian_columns = []
        self.non_gaussian_columns = []
        for column in X_.columns:
            result = st.anderson(X_[column])
            if result[0] > result[1][2]:
                self.non_gaussian_columns.append(column)
            else:
                self.gaussian_columns.append(column)
        return self


class MissingTransformer(BaseEstimator, TransformerMixin, CheckGaussian):
    
    
    def __init__(self, continuous):
        '''
            Method Name: __init__
            Description: This method initializes instance of MissingTransformer class
            Output: None

            Parameters:
            - continuous: Continuous features from dataset
        '''
        super(MissingTransformer, self).__init__()
        self.continuous = continuous
    

    def fit(self, X, y=None):
        '''
            Method Name: fit
            Description: This method classifies way of handling missing data, while fitting respective class methods from feature-engine.imputation library
            Output: self

            Parameters:
            - X: Features from dataset
        '''
        self.mean_imputed_column = []
        self.median_imputed_column = [] 
        self.endtail_iqr_imputed_column = []
        self.endtail_gaussian_imputed_column = []
        if X.isnull().sum().sum() != 0:
            for column in X.columns[X.isna().any()].tolist():
                if (np.abs(X.isnull().corr(method='spearman')[column]).dropna()>0.4).sum()-1 == 0:
                    if (X[column].skew()>-0.5) & (X[column].skew()<0.5):
                        self.mean_imputed_column.append(column)
                    else:
                        self.median_imputed_column.append(column)
                else:
                    if (X[column].skew()>-0.5) & (X[column].skew()<0.5):
                        self.endtail_gaussian_imputed_column.append(column)
                    else:
                        self.endtail_iqr_imputed_column.append(column)
        if self.mean_imputed_column != []:
            self.meanimputer = fei.MeanMedianImputer(
                'mean', variables=self.mean_imputed_column)
            self.meanimputer.fit(X)
        if self.median_imputed_column != []:
            self.medianimputer = fei.MeanMedianImputer(
                'median', variables=self.median_imputed_column)
            self.medianimputer.fit(X)
        if self.endtail_iqr_imputed_column != []:
            self.endtailiqrimputer = fei.EndTailImputer(imputation_method='iqr', fold=1.5, variables = self.endtail_iqr_imputed_column)
            self.endtailiqrimputer.fit(X)
        if self.endtail_gaussian_imputed_column != []:
            self.endtailgaussianmputer = fei.EndTailImputer(imputation_method='gaussian', fold=3, variables=self.endtail_gaussian_imputed_column)
            self.endtailgaussianmputer.fit(X)
        return self


    def transform(self, X, y=None):
        '''
            Method Name: transform
            Description: This method performs transformation on features using respective class methods from feature-engine.imputation library
            Output: Transformed features from dataset in dataframe format.

            Parameters:
            - X: Features from dataset
        '''
        X_ = X.copy()
        if self.mean_imputed_column != []:
            X_ = self.meanimputer.transform(X_)
        if self.median_imputed_column != []:
            X_ = self.medianimputer.transform(X_)
        if self.endtail_iqr_imputed_column != []:
            X_ = self.endtailiqrimputer.transform(X_)
        if self.endtail_gaussian_imputed_column != []:
            X_ = self.endtailgaussianmputer.transform(X_)
        if X_.isnull().sum().sum() != 0:
            for col in X_.columns[X_.isna().any()].tolist():
                if (X_[col].skew()>-0.5) & (X_[col].skew()<0.5):
                    X_[col] = X_[col].fillna(np.nanmean(X_[col]))
                else:
                    X_[col] = X_[col].fillna(np.nanmedian(X_[col]))
        return X_


class OutlierCapTransformer(BaseEstimator, TransformerMixin, CheckGaussian):
    
    
    def __init__(self, continuous):
        '''
            Method Name: __init__
            Description: This method initializes instance of OutlierCapTransformer class
            Output: None

            Parameters:
            - continuous: Continuous features from dataset
        '''
        super(OutlierCapTransformer, self).__init__()
        self.continuous = continuous


    def fit(self, X, y=None):
        '''
            Method Name: fit
            Description: This method classifies way of handling outliers based on whether features are gaussian or non-gaussian, while fitting respective class methods from feature-engine.outliers library
            Output: self

            Parameters:
            - X: Features from dataset
        '''
        X_ = pd.DataFrame(X.copy(), columns = self.continuous)
        self.check_gaussian(X_[self.continuous])
        if self.non_gaussian_columns!=[]:
            self.non_gaussian_winsorizer = feo.Winsorizer(
                capping_method='iqr', tail='both', fold=1.5, add_indicators=False,variables=self.non_gaussian_columns)
            self.non_gaussian_winsorizer.fit(X_)
        if self.gaussian_columns!=[]:
            self.gaussian_winsorizer = feo.Winsorizer(
                capping_method='gaussian', tail='both', fold=3, add_indicators=False,variables=self.gaussian_columns)
            self.gaussian_winsorizer.fit(X_)
        return self
    

    def transform(self, X, y=None):
        '''
            Method Name: transform
            Description: This method performs transformation on features using respective class methods from feature-engine.outliers library
            Output: Transformed features from dataset in dataframe format.

            Parameters:
            - X: Features from dataset
        '''
        X_ = pd.DataFrame(X.copy(), columns = self.continuous)
        if self.non_gaussian_columns != []:
            X_ = self.non_gaussian_winsorizer.transform(X_)
        if self.gaussian_columns != []:
            X_ = self.gaussian_winsorizer.transform(X_)
        return X_


class GaussianTransformer(BaseEstimator, TransformerMixin, CheckGaussian):
    
    
    def __init__(self, continuous):
        '''
            Method Name: __init__
            Description: This method initializes instance of GaussianTransformer class
            Output: None

            Parameters:
            - continuous: Continuous features from dataset
        '''
        super(GaussianTransformer, self).__init__()
        self.continuous = continuous


    def fit(self, X, y=None):
        '''
            Method Name: fit
            Description: This method tests for various gaussian transformation techniques on non-gaussian variables. Non-gaussian variables that best successfully transformed to gaussian variables based on Anderson test will be used for fitting on respective gaussian transformers.
            Output: self

            Parameters:
            - X: Features from dataset
        '''
        X_ = pd.DataFrame(X.copy(), columns = self.continuous)
        self.check_gaussian(X_[self.continuous])
        transformer_list = [
            fet.LogTransformer(), fet.ReciprocalTransformer(), fet.PowerTransformer(exp=0.5), fet.YeoJohnsonTransformer(), fet.PowerTransformer(exp=2),QuantileTransformer(output_distribution='normal')
        ]
        transformer_names = [
            'logarithmic','reciprocal','square-root','yeo-johnson','square','quantile'
        ]
        result_names, result_test_stats, result_columns, result_critical_value=[], [], [], []
        for transformer, name in zip(transformer_list, transformer_names):
            for column in self.non_gaussian_columns:
                try:
                    X_transformed = pd.DataFrame(
                        transformer.fit_transform(X_[[column]]), columns = [column])
                    result_columns.append(column)
                    result_names.append(name)
                    result_test_stats.append(
                        st.anderson(X_transformed[column])[0])
                    result_critical_value.append(
                        st.anderson(X_transformed[column])[1][2])
                except:
                    continue
        results = pd.DataFrame(
            [pd.Series(result_columns, name='Variable'), 
            pd.Series(result_names,name='Transformation_Type'),
            pd.Series(result_test_stats, name='Test-stats'), 
            pd.Series(result_critical_value, name='Critical value')]).T
        best_results = results[results['Test-stats']<results['Critical value']].groupby(by='Variable')[['Transformation_Type','Test-stats']].min()
        transformer_types = best_results['Transformation_Type'].unique()
        for type in transformer_types:
            variable_list = best_results[best_results['Transformation_Type'] == type].index.tolist()
            if type == 'logarithmic':
                self.logtransformer = fet.LogTransformer(variables=variable_list)
                self.logtransformer.fit(X_)
            elif type == 'reciprocal':
                self.reciprocaltransformer = fet.ReciprocalTransformer(variables=variable_list)
                self.reciprocaltransformer.fit(X_)
            elif type == 'square-root':
                self.sqrttransformer = fet.PowerTransformer(exp=0.5, variables=variable_list)
                self.sqrttransformer.fit(X_)
            elif type == 'yeo-johnson':
                self.yeojohnsontransformer = fet.YeoJohnsonTransformer(variables=variable_list)
                self.yeojohnsontransformer.fit(X_)
            elif type == 'square':
                self.squaretransformer = fet.PowerTransformer(exp=2, variables=variable_list)
                self.squaretransformer.fit(X_)
            elif type == 'quantile':
                self.quantiletransformer = QuantileTransformer(output_distribution='normal',random_state=random_state)
                self.quantiletransformer.fit(X_[variable_list])
                self.quantilevariables = variable_list
        return self
    

    def transform(self, X, y=None):
        '''
            Method Name: transform
            Description: This method performs gaussian transformation on features using respective gaussian transformers.
            Output: Transformed features from dataset in dataframe format.

            Parameters:
            - X: Features from dataset
        '''
        X_ = pd.DataFrame(X.copy(), columns = self.continuous)
        if hasattr(self, 'logtransformer'):
            try:
                X_ = self.logtransformer.transform(X_)
            except:
                old_variable_list = self.logtransformer.variables_.copy()
                for var in old_variable_list:
                    if (X_[var]<=0).sum()>0:
                        self.logtransformer.variables_.remove(var)
                X_ = self.logtransformer.transform(X_)
        if hasattr(self, 'reciprocaltransformer'):
            try:
                X_ = self.reciprocaltransformer.transform(X_)
            except:
                old_variable_list = self.reciprocaltransformer.variables_.copy()
                for var in old_variable_list:
                    if (X_[var]==0).sum()>0:
                        self.reciprocaltransformer.variables_.remove(var)
                X_ = self.reciprocaltransformer.transform(X_)
        if hasattr(self, 'sqrttransformer'):
            try:
                X_ = self.sqrttransformer.transform(X_)
            except:
                old_variable_list = self.sqrttransformer.variables_.copy()
                for var in old_variable_list:
                    if (X_[var]==0).sum()>0:
                        self.sqrttransformer.variables_.remove(var)
                X_ = self.sqrttransformer.transform(X_)
        if hasattr(self, 'yeojohnsontransformer'):
            X_ = self.yeojohnsontransformer.transform(X_)
        if hasattr(self, 'squaretransformer'):
            X_ = self.squaretransformer.transform(X_)
        if hasattr(self, 'quantiletransformer'):
            X_[self.quantilevariables] = pd.DataFrame(
                self.quantiletransformer.transform(X_[self.quantilevariables]), columns = self.quantilevariables)
        return X_
        

class ScalingTransformer(BaseEstimator, TransformerMixin, CheckGaussian):
    
    
    def __init__(self, scaler, continuous):
        '''
            Method Name: __init__
            Description: This method initializes instance of ScalingTransformer class
            Output: None

            Parameters:
            - scaler: String that represents method of performing feature scaling. (Accepted values are 'Standard', 'MinMax', 'Robust' and 'Combine')
            - continuous: Continuous features from dataset
        '''
        super(ScalingTransformer, self).__init__()
        self.scaler = scaler
        self.continuous = continuous


    def fit(self, X, y=None):
        '''
            Method Name: fit
            Description: This method fits dataset onto respective scalers selected.
            Output: self

            Parameters:
            - X: Features from dataset
        '''
        X_ = pd.DataFrame(X.copy(), columns = self.continuous)
        if self.scaler == 'Standard':
            self.copyscaler = StandardScaler()
            self.copyscaler.fit(X_)
        elif self.scaler == 'MinMax':
            self.copyscaler = MinMaxScaler()
            self.copyscaler.fit(X_)
        elif self.scaler == 'Robust':
            self.copyscaler = RobustScaler()
            self.copyscaler.fit(X_)
        elif self.scaler == 'Combine':
            self.check_gaussian(X_[self.continuous])
            self.copyscaler = ColumnTransformer(
                [('std_scaler',StandardScaler(),self.gaussian_columns),('minmax_scaler',MinMaxScaler(),self.non_gaussian_columns)],remainder='passthrough',n_jobs=2)
            self.copyscaler.fit(X_)
        return self
    

    def transform(self, X, y=None):
        '''
            Method Name: transform
            Description: This method performs transformation on features using respective scalers.
            Output: Transformed features from dataset in dataframe format.

            Parameters:
            - X: Features from dataset
        '''
        X_ = pd.DataFrame(X.copy(), columns = self.continuous)
        if self.scaler != 'Combine':
            X_ = pd.DataFrame(
                self.copyscaler.transform(X_), columns = self.continuous)
        else:
            X_ = pd.DataFrame(
                self.copyscaler.transform(X_), columns = self.gaussian_columns + self.non_gaussian_columns)
            X_ = X_[self.continuous]
        return X_


class FeatureSelectionTransformer(BaseEstimator, TransformerMixin):
    
    
    def __init__(
            self, method, drop_correlated, continuous, categorical, model, scaling_indicator= 'no', cluster_indicator= 'no', damping = None, number=None):
        '''
            Method Name: __init__
            Description: This method initializes instance of FeatureSelectionTransformer class
            Output: None

            Parameters:
            - method: String that represents method of feature selection (Accepted values are 'BorutaShap', 'Lasso', 'FeatureImportance_ET', 'FeatureImportance_self', 'MutualInformation', 'ANOVA', 'FeatureWiz')
            - drop_correlated: String indicator of dropping highly correlated features (yes or no)
            - continuous: Continuous features from dataset
            - categorical: Categorical features from dataset
            - model: Model object
            - scaling_indicator: String that represents method of performing feature scaling. (Accepted values are 'Standard', 'MinMax', 'Robust', 'Combine' and 'no'). Default value is 'no'
            - cluster_indicator: String indicator of including cluster-related feature (yes or no). Default value is 'no'
            - damping: Float value (range from 0.5 to 1 not inclusive) as an additional hyperparameter for Affinity Propagation clustering algorithm. Default value is None.
            - number: Integer that represents number of features to select. Minimum value required is 1. Default value is None.

        '''
        self.method = method
        self.drop_correlated = drop_correlated
        self.continuous = continuous
        self.categorical = categorical
        self.model = model
        self.scaling_indicator = scaling_indicator
        self.cluster_indicator = cluster_indicator
        self.damping = damping
        self.number = number


    def fit(self, X, y=None):
        '''
            Method Name: fit
            Description: This method finds list of correlated features if drop_correlated indicator is 'yes', identifies subset of columns from respective feature selection techniques and fits clustering from features using Affinity Propagation if clustering indicator is 'yes'.
            Output: self

            Parameters:
            - X: Features from dataset
        '''
        X_ = pd.DataFrame(X.copy(), columns = self.continuous + self.categorical)
        y = y.reset_index(drop=True)
        if self.drop_correlated == 'yes':
            self.correlated_selector = fes.DropCorrelatedFeatures(method='spearman')
            self.correlated_selector.fit(X_)
            X_ = self.correlated_selector.transform(X_)
        if self.method == 'BorutaShap':
            borutashap = BorutaShap(
                importance_measure = 'shap', classification = True)
            borutashap.fit(
                X = X_, y = y['Output'], verbose = False, stratify = y['Output'])
            self.sub_columns = borutashap.Subset().columns.to_list()
        elif self.method == 'Lasso':
            imp_model = LogisticRegression(
                random_state=random_state,penalty='l1',max_iter=1000, solver='saga')
            imp_model.fit(X_,y)
            if self.scaling_indicator == 'no':
                result = pd.DataFrame(
                    [pd.Series(X_.columns),pd.Series(np.abs(imp_model.coef_[0])*np.array(X_).std(axis=0))], index=['Variable','Value']).T
            else:
                result = pd.DataFrame(
                    [pd.Series(X_.columns),pd.Series(np.abs(imp_model.coef_[0]))], index=['Variable','Value']).T
            result['Value'] = result['Value'].astype('float64')
            self.sub_columns =  result.loc[result['Value'].nlargest(self.number).index.tolist()]['Variable'].tolist()
        elif self.method == 'FeatureImportance_ET':
            fimp_model = ExtraTreesClassifier(random_state=random_state)
            fimportance_selector = SelectFromModel(
                fimp_model,max_features=self.number,threshold=0.0)
            fimportance_selector.fit(X_,y)
            self.sub_columns = X_.columns[fimportance_selector.get_support()].to_list()
        elif self.method == 'FeatureImportance_self':
            fimp_model = clone(self.model)
            fimportance_selector = SelectFromModel(
                fimp_model,max_features=self.number,threshold=0.0)
            fimportance_selector.fit(X_,y)
            self.sub_columns = X_.columns[fimportance_selector.get_support()].to_list()
        elif self.method == 'MutualInformation':
            values = mutual_info_classif(X_,y,random_state=random_state)
            result = pd.DataFrame(
                [pd.Series(X_.columns),pd.Series(values)], index=['Variable','Value']).T
            result['Value'] = result['Value'].astype('float64')
            self.sub_columns =  result.loc[result['Value'].nlargest(self.number).index.tolist()]['Variable'].tolist()
        elif self.method == 'ANOVA':
            fclassif_selector = SelectKBest(f_classif,k=self.number)
            fclassif_selector.fit(X_,y)
            self.sub_columns =  X_.columns[fclassif_selector.get_support()].to_list()
        elif self.method == 'FeatureWiz':
            selector = FeatureWiz(verbose=0)
            selector.fit(X_, y)
            self.sub_columns = selector.features
        if self.cluster_indicator == 'yes':
            self.affinitycluster = AffinityPropagation(random_state=random_state, damping = self.damping)
            if self.sub_columns != []:
                self.affinitycluster.fit(X_[self.sub_columns])
                dist = pairwise_distances(
                    X_[self.sub_columns], self.affinitycluster.cluster_centers_).min(axis=1)
            else:
                self.affinitycluster.fit(X_)
                dist = pairwise_distances(
                    X_, self.affinitycluster.cluster_centers_).min(axis=1)
            if self.scaling_indicator == 'Standard':
                self.scaler = StandardScaler()
                self.scaler.fit(dist.reshape(-1, 1))
            elif self.scaling_indicator == 'MinMax':
                self.scaler = MinMaxScaler()
                self.scaler.fit(dist.reshape(-1, 1))
            elif self.scaling_indicator == 'Robust':
                self.scaler = RobustScaler()
                self.scaler.fit(dist.reshape(-1, 1))  
            elif self.scaling_indicator == 'Combine': 
                result = st.anderson(dist)
                if result[0] > result[1][2]:
                    self.scaler = MinMaxScaler()
                    self.scaler.fit(dist.reshape(-1, 1))
                else:
                    self.scaler = StandardScaler()
                    self.scaler.fit(dist.reshape(-1, 1))
        return self
    

    def transform(self, X, y=None):
        '''
            Method Name: transform
            Description: This method extracts non-correlated columns if drop correlated indicator is "yes", followed by subset of columns identified from feature selection and add additional feature named "cluster_distance" if cluster inndicator is "yes" 
            Output: Transformed features from dataset in dataframe format.

            Parameters:
            - X: Features from dataset
        '''
        X_ = pd.DataFrame(X.copy(), columns = self.continuous + self.categorical)
        if self.drop_correlated == 'yes':
            X_ = self.correlated_selector.transform(X_)
        if self.sub_columns != []:
            X_ = X_[self.sub_columns]
        if self.cluster_indicator == 'yes':
            X_['cluster_distance'] = pairwise_distances(X_, self.affinitycluster.cluster_centers_).min(axis=1)
            X_['cluster_distance'] = self.scaler.transform(np.array(X_['cluster_distance']).reshape(-1, 1))
        return X_