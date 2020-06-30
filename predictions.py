"""
Predictions are made here.
@AugustSemrau
"""

from dataLoader import dataLoader
from models import Models
from dataLoader import csvSaver




if __name__ == '__main__':

    # Import test data
    testData = dataLoader(test=True, ageNAN='median')

    # Initiate models
    Models = Models(agenan='median')


    # Logistic regression Model
    logisticModel = Models.build_model_logisticReg()
    logistic_predictions = logisticModel.predict(testData)
    # Saving predictions
    csvSaver(predictions=logistic_predictions, type="Logistic")

    # print(list(logistic_predictions))
    # print(len(logistic_predictions))



