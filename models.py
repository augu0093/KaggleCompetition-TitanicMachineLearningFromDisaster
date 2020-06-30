"""
Building predictive model on regression principles.
@AugustSemrau
"""
from dataLoader import dataLoader
from sklearn.linear_model import LogisticRegression



class Models():
    """Class containing all (sklearn-built) models"""

    def __init__(self, agenan="median"):
        self.X, self.y = dataLoader(test=False, return_X_y=True, ageNAN=agenan)

    # Logistic Regression Model
    def build_model_logisticReg(self):
        model = LogisticRegression(random_state=0)
        model.fit(self.X, self.y)
        return model







