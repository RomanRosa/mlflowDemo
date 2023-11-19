# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import VotingRegressor
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Replace with your tracking server URL

#mlflow.set_experiment(experiment_id="0")
mlflow.autolog()

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def evaluate_models(train_x,train_y,test_x,test_y,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]

            # Train model
            model.fit(train_x,train_y)

            # Predict Training data
            y_train_pred = model.predict(train_x)

            # Predict Testing data
            y_test_pred =model.predict(test_x)

            # Get R2 scores for train and test data
            train_model_score = r2_score(train_y,y_train_pred)
            test_model_score = r2_score(test_y,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report
    
    except Exception as e:
        logger.exception('Exception Occured During Model Training')



def print_evaluated_results(train_x,train_y,test_x,test_y,model):
    try:
        ytrain_pred = model.predict(train_x)
        ytest_pred = model.predict(test_x)

        # Evaluate Train and Test dataset
        model_train_mae , model_train_rmse, model_train_r2 = eval_metrics(train_y, ytrain_pred)
        model_test_mae , model_test_rmse, model_test_r2 = eval_metrics(test_y, ytest_pred)

        # Printing results
        print('Model Performance For Training Set')
        print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
        print("- R2 Score: {:.4f}".format(model_train_r2))

        print('----------------------------------')
    
        print('Model Performance For Test Set')
        print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
        print("- R2 Score: {:.4f}".format(model_test_r2))
    
    except Exception as e:
        logger.info('Exception Occured During Printing Of Evaluated Results')

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        models= {
            'Linear Regression': LinearRegression(),
            "Lasso": Lasso(),
            "Ridge": Ridge(),
            #'ElasticNet': ElasticNet(),
            'XGBRFRegressor': XGBRegressor(),
            'CatBoostRegressor': CatBoostRegressor(),
            'KNeighborsRegressor':KNeighborsRegressor(),
            'RandomForestRegressor': RandomForestRegressor(),
            'AdaBoostRegressor': AdaBoostRegressor(),
           'GradientBoostingRegressor': GradientBoostingRegressor(),
            'SVR': SVR()
        }
        
        model_report:dict = evaluate_models(train_x,train_y,test_x,test_y,models)
        print(model_report)
        print('\n====================================================================================\n')
        logger.info(f'Model Report : {model_report}')
        
        # To get best model score from dictionary 
        best_model_score = max(sorted(model_report.values()))

        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
            ]
        best_model = models[best_model_name]

        if best_model_score < 0.6 :
            logger.info('Best Model Has R2 Score Less Than 60%')
            logger.exception('No Best Model Found!')
            
            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logger.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            logger.info('Hyperparameter Tuning Started For Catboost')

    # Hyperparameter tuning on Catboost
        # Initializing catboost
        cbr = CatBoostRegressor(verbose=False)

        # Creating the hyperparameter grid
        param_dist = {'depth'          : [4,5,6,7,8,9, 10],
                        'learning_rate' : [0.01,0.02,0.03,0.04],
                        'iterations'    : [300,400,500,600]}

        #Instantiate RandomSearchCV object
        rscv = RandomizedSearchCV(cbr , param_dist, scoring='r2', cv =5, n_jobs=-1)

        # Fit the model
        rscv.fit(train_x, train_y)

        # Print the tuned parameters and score
        print(f'Best Catboost Parameters : {rscv.best_params_}')
        print(f'Best Catboost Score : {rscv.best_score_}')
        print('\n====================================================================================\n')

        best_cbr = rscv.best_estimator_

        logger.info('Hyperparameter Tuning Complete for Catboost')

        logger.info('Hyperparameter Tuning Started for KNN')

        # Initialize knn
        knn = KNeighborsRegressor()

        # parameters
        k_range = list(range(2, 31))
        param_grid = dict(n_neighbors=k_range)

        # Fitting the cvmodel
        grid = GridSearchCV(knn, param_grid, cv=5, scoring='r2',n_jobs=-1)
        grid.fit(train_x, train_y)

        # Print the tuned parameters and score
        print(f'Best KNN Parameters : {grid.best_params_}')
        print(f'Best KNN Score : {grid.best_score_}')
        print('\n====================================================================================\n')

        best_knn = grid.best_estimator_

        logger.info('Hyperparameter Tuning Complete for KNN')

        logger.info('Voting Regressor Model Training Started')

        # Creating final Voting regressor
        er = VotingRegressor([('cbr',best_cbr),('xgb',XGBRegressor()),('knn',best_knn)], weights=[3,2,1])
        er.fit(train_x, train_y)
        print('Final Model Evaluation :\n')
        print_evaluated_results(train_x,train_y,train_x,train_y,er)
        logger.info('Voting Regressor Training Completed')

        # Evaluating Ensemble Regressor (Voting Classifier on test data)
        ytest_pred = er.predict(test_x)

        mae, rmse, r2 = eval_metrics(test_y, ytest_pred)
        logger.info(f'Test MAE : {mae}')
        logger.info(f'Test RMSE : {rmse}')
        logger.info(f'Test R2 Score : {r2}')
        logger.info('Final Model Training Completed')


        #print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        
        # # For remote server only (Dagshub)
        # remote_server_uri = "https://dagshub.com/entbappy/MLflow-Basic-Demo.mlflow"
        # mlflow.set_tracking_uri(remote_server_uri)


        # For remote server only (AWS)
        #remote_server_uri = "http://ec2-54-147-36-34.compute-1.amazonaws.com:5000/"
        #mlflow.set_tracking_uri(remote_server_uri)



        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            for model_name, model in models.item():
                mlflow.sklearn.load_model(
                    model,
                    'model',
                    registered_model_name=f'{model_name}WineModel')

        else:
            mlflow.sklearn.log_model(models, "model")