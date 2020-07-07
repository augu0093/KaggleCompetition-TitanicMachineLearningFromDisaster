"""
Predictions are made here.
@AugustSemrau
"""

from data_loader import dataLoader
from models import Models
from data_loader import csvSaver
import xgboost as xgb



if __name__ == '__main__':

    # Import test data
    testData = dataLoader(test=True, ageNAN='median')

    # Initiate models
    Models = Models(agenan='median')


    # Logistic Regression model predictions
    logisticModel = Models.build_model_LR()
    logistic_predictions = logisticModel.predict(testData)
    # Saving predictions
    csvSaver(predictions=logistic_predictions, type="logistic")

    # Naive Bayes model predictions
    naiveBayes_model = Models.build_model_NB()
    naiveBayes_predictions = naiveBayes_model.predict(testData)
    # Saving predictions
    csvSaver(predictions=naiveBayes_predictions, type="naiveBayes")

    # Stochastic Gradient Descent model predictions
    SGD_model = Models.build_model_SGD()
    SGD_predictions = SGD_model.predict(testData)
    # Saving predictions
    csvSaver(predictions=SGD_predictions, type="SGD")

    # Decision model predictions
    KNN_model = Models.build_model_KNN()
    KNN_predictions = KNN_model.predict(testData)
    # Saving predictions
    csvSaver(predictions=KNN_predictions, type="KNN")

    # Decision model predictions
    DT_model = Models.build_model_DT()
    DT_predictions = DT_model.predict(testData)
    # Saving predictions
    csvSaver(predictions=DT_predictions, type="DT")

    # Default and tuned Random Forest model predictions
    RF_model = Models.build_model_RF()
    RF_tuned = Models.build_optimized_RF()
    RF_predictions = RF_model.predict(testData)
    RF_tuned_predictions = RF_tuned.predict(testData)
    # Saving predictions
    csvSaver(predictions=RF_predictions, type="RF")
    csvSaver(predictions=RF_tuned_predictions, type="RF_tuned")

    # Decision model predictions
    SVC_model = Models.build_model_SVC()
    SVC_predictions = SVC_model.predict(testData)
    # Saving predictions
    csvSaver(predictions=SVC_predictions, type="SVC")

    # XGBoost model predictions
    XGB_model = Models.build_model_XGB()
    XGB_predictions = XGB_model.predict(testData)
    # Saving predictions
    csvSaver(predictions=XGB_predictions, type="XGBoost")



