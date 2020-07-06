"""
Predictions are made here.
@AugustSemrau
"""

from dataLoader import dataLoader
from models import Models
from dataLoader import csvSaver
import xgboost as xgb



if __name__ == '__main__':

    # Import test data
    testData = dataLoader(test=True, ageNAN='median')

    # Initiate models
    Models = Models(agenan='median')


    # Logistic Regression model predictions
    logisticModel = Models.build_model_logisticReg()
    logistic_predictions = logisticModel.predict(testData)
    # Saving predictions
    csvSaver(predictions=logistic_predictions, type="logistic")

    # Naive Bayes model predictions
    naiveBayes_model = Models.build_model_naiveBayes()
    naiveBayes_predictions = naiveBayes_model.predict(testData)
    # Saving predictions
    csvSaver(predictions=naiveBayes_predictions, type="naiveBayes")

    # Stochastic Gradient Descent model predictions
    SGD_model = Models.build_model_SGD()
    SGD_predictions = SGD_model.predict(testData)
    # Saving predictions
    csvSaver(predictions=SGD_predictions, type="SGD")


    # Random Forest model predictions
    RF_model = Models.build_model_RF()
    RF_predictions = RF_model.predict(testData)
    # Saving predictions
    csvSaver(predictions=RF_predictions, type="RF")



    # XGBoost model predictions
    XGB_model = Models.build_model_XGBoost()
    XGB_testData = xgb.DMatrix(testData.to_numpy())
    XGB_predictions = XGB_model.predict(testData.to_numpy())
    # Saving predictions
    csvSaver(predictions=XGB_predictions, type="XGBoost")



