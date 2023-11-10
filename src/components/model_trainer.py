import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingClassifier,RandomForestRegressor)
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("aritifacts","model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Spliting training and testing input data...")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],train_array[:,-1],
                test_array[:,:-1],test_array[:,-1],
            )
            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Linear Regession" : LinearRegression(),
                "K-Neighbors Classifier":KNeighborsRegressor(),
                "XGBClassifier":XGBRegressor(),
                "CatBoosting Classifier":CatBoostRegressor(),
                "AdaBoost Classifier":AdaBoostRegressor(),
                }
            model_report = evaluate_model(X_train,y_train,X_test,y_test,models=models)
        
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get  best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            best_model = models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("NO BEST MODEL FOUND!")
            logging.info("Best Model on both training and test Dataset is Found")

            # load preprocessor.pkl file if data is changing

            save_obj(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)
            
            # Check Best model Prediction on test data
            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test,predicted)
            
            return r2_square
        
        except Exception as e:
            CustomException(e,sys)
