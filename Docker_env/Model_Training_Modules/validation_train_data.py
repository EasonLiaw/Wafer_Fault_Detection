'''
Author: Liaw Yi Xian
Last Modified: 11th October 2022
'''

import os, shutil, json
import pandas as pd
import mysql.connector
import csv
from Application_Logger.logger import App_Logger
import DBConnectionSetup as login

class DBOperations:


    def __init__(self, tablename, file_object):
        '''
            Method Name: __init__
            Description: This method initializes instance of DBOperations class
            Output: None
        '''
        self.tablename = tablename
        self.file_object = file_object
        self.log_writer = App_Logger()
        self.host = login.logins['host']
        self.user = login.logins['user']
        self.password = login.logins['password']
        self.dbname = login.logins['dbname']


    def newDB(self, schema):
        '''
            Method Name: newDB
            Description: This method creates a new database and table in MySQL database based on a given schema file object.
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(
            self.file_object, f"Start creating new table({self.tablename}) in SQL database ({self.dbname})")
        self.schema = schema
        try:
            conn = mysql.connector.connect(
                host=self.host,user=self.user,password=self.password)
            mycursor = conn.cursor()
            mycursor.execute(f"""CREATE DATABASE IF NOT EXISTS {self.dbname}""")
            mycursor.execute(f"""USE {self.dbname}""")
            for name, type in zip(self.schema['ColName'].keys(),self.schema['ColName'].values()):
                mycursor.execute(
                    f"""SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{self.tablename}' AND table_schema = '{self.dbname}'""")
                if mycursor.fetchone()[0] == 1:
                    try:
                        mycursor.execute(
                            f"ALTER TABLE {self.tablename} ADD {name} {type}")
                        self.log_writer.log(
                            self.file_object, f"Column {name} added into {self.tablename} table")
                    except:
                        self.log_writer.log(
                            self.file_object, f"Column {name} already exists in {self.tablename} table")
                else:
                    try:
                        mycursor.execute(
                            f"CREATE TABLE {self.tablename} ({name} {type})")
                        self.log_writer.log(
                            self.file_object, f"{self.tablename} table created with column {name}")
                    except Exception as e:
                        self.log_writer.log(
                            self.file_object, f"SQL server has error of creating new table {self.tablename} with the following error: {e}")
        except ConnectionError:
            self.log_writer.log(
                self.file_object, "Error connecting to SQL database")
            raise Exception("Error connecting to SQL database")
        except Exception as e:
            self.log_writer.log(
                self.file_object, f"The following error occured when creating new SQL database: {e}")
            raise Exception(
                f"The following error occured when creating new SQL database: {e}")
        conn.close()
        self.log_writer.log(
            self.file_object, f"Finish creating new table({self.tablename}) in SQL database ({self.dbname})")
    

    def data_insert(self, gooddir):
        '''
            Method Name: data_insert
            Description: This method inserts data from existing csv file into MySQL database
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(
            self.file_object, "Start inserting new good training data into SQL database")
        self.gooddir = gooddir
        try:
            conn = mysql.connector.connect(
                host=self.host,user=self.user,password=self.password,database = self.dbname)
            mycursor = conn.cursor()
            for file in os.listdir(self.gooddir[:-1]):
                with open(self.gooddir+file, "r") as f:
                    next(f)
                    filename = csv.reader(f)
                    for line in enumerate(filename):
                        try:
                            line[1][0] = f"\"{line[1][0]}\""
                            mycursor.execute(
                                f"INSERT INTO {self.tablename} VALUES ({','.join(line[1])})")
                            conn.commit()
                        except Exception as e:
                            self.log_writer.log(
                                self.file_object, f'Row {line[0]} could not be inserted into database for {file} file with the following error: {e}')
                            conn.rollback()
                    self.log_writer.log(
                        self.file_object, f"{file} file added into database")
        except ConnectionError:
            self.log_writer.log(
                self.file_object, "Error connecting to SQL database")
            raise Exception("Error connecting to SQL database")
        except Exception as e:
            self.log_writer.log(
                self.file_object, f"The following error occured when inserting good training data into SQL database: {e}")
            raise Exception(
                f"The following error occured when inserting good training data into SQL database: {e}")
        conn.close()
        self.log_writer.log(
            self.file_object, "Finish inserting new good training data into SQL database")


    def compile_data_from_DB(self,compiledir):
        '''
            Method Name: compile_data_from_DB
            Description: This method compiles data from MySQL table into csv file for further data preprocessing.
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(
            self.file_object, "Start writing compiled good training data into a new CSV file")
        self.compiledir = compiledir
        try:
            conn = mysql.connector.connect(
                host=self.host,user=self.user,password=self.password,database = self.dbname)
            data = pd.read_sql(
                f'''SELECT DISTINCT * FROM {self.tablename};''', conn)
            data.to_csv(self.compiledir, index=False)
        except ConnectionError:
            self.log_writer.log(
                self.file_object, "Error connecting to SQL database")
            raise Exception("Error connecting to SQL database")
        except Exception as e:
            self.log_writer.log(
                self.file_object, f"The following error occured when compiling good training data into a new CSV file: {e}")
            raise Exception(
                f"The following error occured when compiling good training data into a new CSV file: {e}")
        conn.close()
        self.log_writer.log(
            self.file_object, "Finish writing compiled good training data into a new CSV file")


class rawtraindatavalidation(DBOperations):


    def __init__(self, tablename, file_object, gooddir, baddir):
        '''
            Method Name: __init__
            Description: This method initializes instance of rawtraindatavalidation class, while inheriting methods from DBOperations class
            Output: None
            On Failure: Logging error and raise exception
        '''
        super().__init__(tablename, file_object)
        self.gooddir = gooddir
        self.baddir = baddir
        self.log_writer = App_Logger()


    def load_train_schema(self, filename):
        '''
            Method Name: load_train_schema
            Description: This method loads the schema of the training data from a given JSON file for creating tables in MySQL database.
            Output: JSON object
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, "Start loading train schema")
        self.filename = filename
        try:
            with open(filename, 'r') as f:
                schema = json.load(f)
        except Exception as e:
            self.log_writer.log(
                self.file_object, f"Training schema fail to load with the following error: {e}")
            raise Exception(
                f"Training schema fail to load with the following error: {e}")
        self.log_writer.log(self.file_object, "Finish loading train schema")
        return schema
    

    def file_initialize(self, filelist):
        '''
            Method Name: file_initialize
            Description: This method creates the list of folders mentioned in the filelist if not exist. If exist, this method deletes the existing folders and creates new ones. Note that manual archiving will be required if backup of existing files is required.
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(
            self.file_object, "Start initializing folder structure")
        self.filelist = filelist
        for folder in self.filelist:
            try:
                if os.path.exists(folder):
                    shutil.rmtree(folder)
                os.makedirs(os.path.dirname(folder), exist_ok=True)
                self.log_writer.log(
                    self.file_object, f"Folder {folder} has been initialized")
            except Exception as e:
                self.log_writer.log(
                    self.file_object, f"Folder {folder} could not be initialized with the following error: {e}")
                raise Exception(
                    f"Folder {folder} could not be initialized with the following error: {e}")
        self.log_writer.log(
            self.file_object, "Finish initializing folder structure")
    

    def file_namecheck(self,sourcedir,schema):
        '''
            Method Name: file_namecheck
            Description: This method checks for the validity of file names of CSV files. If the CSV file does not follow specified name format, the CSV file is moved to bad data folder.
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(
            self.file_object, "Start checking for valid name of files")
        self.sourcedir = sourcedir
        self.schema = schema
        for file in os.listdir(self.sourcedir):
            filename = file.split(".csv")[0].split('_')
            if len(filename)!=3 or filename[0] != 'wafer' or len(filename[1])!=self.schema['LengthOfDateStampInFile'] or len(filename[2])!=self.schema['LengthOfTimeStampInFile']:
                try:
                    shutil.copyfile(self.sourcedir+"/"+file, self.baddir+file)
                    self.log_writer.log(
                        self.file_object, f"{file} moved to bad data folder due to invalid file name")
                except Exception as e:
                    self.log_writer.log(
                        self.file_object, f"{file} could not be moved to bad data folder with the following error: {e}")
                    raise Exception(
                        f"{file} could not be moved to bad data folder with the following error: {e}")
            else:
                try:
                    shutil.copyfile(self.sourcedir+"/"+file, self.gooddir+file)
                    self.log_writer.log(
                        self.file_object, f"{file} moved to good data folder")
                except Exception as e:
                    self.log_writer.log(
                        self.file_object, f"{file} could not be moved to good data folder with the following error: {e}")
                    raise Exception(
                        f"{file} could not be moved to good data folder with the following error: {e}")
        self.log_writer.log(
            self.file_object, "Finish checking for valid name of files")


    def column_count(self, schema):
        '''
            Method Name: column_count
            Description: This method checks for the number of columns in a given CSV file based on number of columns defined in schema object.If the CSV file does not contain specified number of columns, the CSV file is moved to bad data folder.
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(
            self.file_object, "Start checking for number of columns in file")
        self.schema = schema
        for file in os.listdir(self.gooddir[:-1]):
            try:
                filename = pd.read_csv(os.path.join(self.gooddir,file))
            except Exception as e:
                self.log_writer.log(
                    self.file_object, f"{file} could not be read with the following error: {e}")
                raise Exception(
                    f"{file} could not be read with the following error: {e}")
            if filename.shape[1] != self.schema['NumberofColumns']:
                try:
                    shutil.move(self.gooddir+file, self.baddir+file)
                    self.log_writer.log(
                        self.file_object, f"{file} moved to bad data folder due to mismatch of number of columns")
                except PermissionError:
                    self.log_writer.log(
                        self.file_object, f"{file} is open, please close and try again")
                    raise Exception(
                        f"{file} is open, please close and try again")
                except Exception as e:
                    self.log_writer.log(
                        self.file_object, f"{file} file fail to move to bad data folder with the following error: {e}")
                    raise Exception(
                        f"{file} file fail to move to bad data folder with the following error: {e}")
        self.log_writer.log(
            self.file_object, "Finish checking for number of columns in file")
    

    def all_null_column_check(self):
        '''
            Method Name: all_null_column_check
            Description: This method checks for the existence of columns having all null values in a given CSV file. If the CSV file has any columns that contains all null values, the CSV file is moved to bad data folder.
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(
            self.file_object, "Start checking for columns with all missing values in file")
        for file in os.listdir(self.gooddir[:-1]):
            try:
                filename = pd.read_csv(os.path.join(self.gooddir,file))
            except Exception as e:
                self.log_writer.log(
                    self.file_object, f"{file} could not be read with the following error: {e}")
                raise Exception(
                    f"{file} could not be read with the following error: {e}")      
            for column in filename.columns:
                if filename[column].isnull().all():
                    try:
                        shutil.move(self.gooddir+file, self.baddir+file)
                        self.log_writer.log(
                            self.file_object, f"{file} moved to bad data folder due to having columns with all missing values")
                    except PermissionError:
                        self.log_writer.log(
                            self.file_object, f"{file} is open, please close and try again")
                        raise Exception(
                            f"{file} is open, please close and try again")
                    except Exception as e:
                        self.log_writer.log(
                            self.file_object, f"{file} file fail to move to bad data folder with the following error: {e}")
                        raise Exception(
                            f"{file} file fail to move to bad data folder with the following error: {e}")
                    break
        self.log_writer.log(
            self.file_object, "Finish checking for columns with all missing values in file")
    

    def blank_with_null_replacement(self):
        '''
            Method Name: blank_with_null_replacement
            Description: This method replaces blank cells with null keyword in a given CSV file.
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(
            self.file_object, "Start replacing missing values with null keyword")
        for file in os.listdir(self.gooddir[:-1]):
            try:
                filename = pd.read_csv(os.path.join(self.gooddir,file))
                filename.fillna('null', inplace=True)
                filename.to_csv(self.gooddir+file, index=False)
            except Exception as e:
                self.log_writer.log(
                    self.file_object, f"Replacing missing values with null keyword for file {file} fail with the following error: {e}")
                raise Exception(
                    f"Replacing missing values with null keyword for file {file} fail with the following error: {e}")
        self.log_writer.log(
            self.file_object, "Finish replacing missing values with null keyword")
    

    def remove_temp_good_train_data(self):
        '''
            Method Name: remove_temp_good_train_data
            Description: This method removes files contained in good_training_data folder after successfully extract compiled data from MySQL database. Note that good data folder only stores CSV files temporarily during this pipeline execution.
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(
            self.file_object, "Start deleting all good_training_data files")
        for file in os.listdir(self.gooddir[:-1]):
            try:
                os.remove(self.gooddir+file)
                self.log_writer.log(
                    self.file_object, f"{file} file deleted from Good_Training_Data folder")
            except PermissionError:
                self.log_writer.log(
                    self.file_object, f"{file} file is open, please close and try again")
                raise Exception(
                    f"{file} file is open, please close and try again")
            except Exception as e:
                self.log_writer.log(
                    self.file_object, f"{file} file fail to delete with the following error: {e}")
                raise Exception(
                    f"{file} file fail to delete with the following error: {e}")
        self.log_writer.log(
            self.file_object, "Finish deleting all good_training_data files")
    

    def bad_to_archive_data(self, archivedir):
        '''
            Method Name: bad_to_archive_data
            Description: This method transfers files contained in bad data folder to archive data folder after successfully extract compiled data from MySQL database. Note that bad data folder only stores CSV files temporarily during this pipeline execution.
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(
            self.file_object, "Start moving all bad data files into archive folder")
        self.archivedir = archivedir
        for file in os.listdir(self.baddir[:-1]):
            try:
                shutil.move(self.baddir+file, self.archivedir+file)
                self.log_writer.log(
                    self.file_object, f"{file} moved to archive data folder")
            except PermissionError:
                self.log_writer.log(
                    self.file_object, f"{file} is open, please close and try again")
                raise Exception(f"{file} is open, please close and try again")
            except Exception as e:
                self.log_writer.log(
                    self.file_object, f"{file} file fail to move to archive with the following error: {e}")
                raise Exception(
                    f"{file} file fail to move to archive with the following error: {e}")
        self.log_writer.log(
            self.file_object, "Finish moving all bad data files into archive folder")
    

    def initial_data_preparation(
            self, schemapath, folders, batchfilepath, goodfilepath, archivefilepath, finalfilepath):
        '''
            Method Name: initial_data_preparation
            Description: This method performs all the preparation tasks for the data to be ingested into MySQL database.
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, "Start initial data preparation")
        self.schemapath = schemapath
        self.folders = folders
        self.batchfilepath = batchfilepath
        self.goodfilepath = goodfilepath
        self.archivefilepath = archivefilepath
        self.finalfilepath = finalfilepath
        schema = self.load_train_schema(self.schemapath)
        self.file_initialize(self.folders)
        self.file_namecheck(self.batchfilepath,schema)
        self.column_count(schema)
        self.all_null_column_check()
        self.blank_with_null_replacement()
        self.newDB(schema)
        self.data_insert(self.goodfilepath)
        self.compile_data_from_DB(self.finalfilepath)
        self.remove_temp_good_train_data()
        self.bad_to_archive_data(self.archivefilepath)
        self.log_writer.log(self.file_object, "Finish initial data preparation")