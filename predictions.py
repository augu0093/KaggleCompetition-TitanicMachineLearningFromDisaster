"""
Predictions are made here.
@AugustSemrau
"""

from dataLoader import dataLoader
from models import Models





if __name__ == '__main__':

    ageReplacement = 'median'
    # Import test data
    testData = dataLoader(test=True, ageNAN=ageReplacement)


    # Initiate models
    Models = Models(agenan='median')


    logisticModel = Models.build_model_logisticReg()

    print(logisticModel.predict(testData))

