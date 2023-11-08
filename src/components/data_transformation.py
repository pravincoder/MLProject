import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
from src.exception import CustomException
from src.logger import logging


from src.utils import save_object


@dataclass
class DataTransformationConfig:
    """Configure Transformed Data Path"""
    preprocessor_obj_file_path = os.path.join('aritifact', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self) -> None:
        logging.info('Data Transformation Started')
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """Data Transformation based different types of Data like Numerical,Categorical,......"""
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course',
            ]
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )
            logging.info(f'Numerical columns standard scaling completed , where Numerical Colm={numerical_columns}')
            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('OneHotEncode', OneHotEncoder()),
                    ("scaler", StandardScaler())
                ]
            )
            logging.info(f'Categorical columns encoding completed ,where Categorical Colm={categorical_columns}')

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("categorical_pipeline", categorical_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info('Reading train and test Data.....')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Obtaining preprocessing Object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ['writing_score', 'reading_score']

            input_feature_train_df = train_df.drop(
                columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                "Applying Preprocessing Object on training dataframe and testing dataframe. ")

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_train_df)
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_test_df)
            ]
            test_arr = np.c_[input_feature_test_arr,
                             np.array(target_feature_test_df)]

            logging.info("Saved Preprocessing Object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr.
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,

            )

        except Exception as e:
            CustomException(e, sys)
