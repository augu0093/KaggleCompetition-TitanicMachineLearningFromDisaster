"""
Building predictive model on regression principles.
@AugustSemrau
"""
import numpy as np

from data_loader import dataLoader

# SciKit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import xgboost as xgb
from xgboost import XGBClassifier


class Models:
    """Class containing all classification models"""
    # Init
    def __init__(self, agenan="median"):
        self.X, self.y = dataLoader(test=False, optimize_set=False, ageNAN=agenan)
        self.y = self.y.values.ravel()

    # Logistic Regression Model
    def build_model_LR(self):
        model = LogisticRegression(max_iter=1000, random_state=0)
        model.fit(self.X, self.y)
        return model

    # Naive Bayes Model
    def build_model_NB(self):
        model = GaussianNB()
        model.fit(self.X, self.y)
        return model

    # Stochastic Gradient Descent Model
    def build_model_SGD(self):
        model = SGDClassifier(loss='squared_loss', shuffle=True, random_state=0)
        model.fit(self.X, self.y)
        return model

    # K-Nearest Neighbors Model
    def build_model_KNN(self):
        model = KNeighborsClassifier(n_neighbors=10)
        model.fit(self.X, self.y)
        return model

    # Decision Tree Model
    def build_model_DT(self):
        model = DecisionTreeClassifier(random_state=0)
        model.fit(self.X, self.y)
        return model

    # Random Forest Model, no tuning
    def build_model_RF(self):
        model = RandomForestClassifier(oob_score=True, random_state=0)
        model.fit(self.X, self.y)
        return model
    # Random Forest Model, Bayesian Optimization tuned
    def build_optimized_RF(self):
        model = RandomForestClassifier(n_estimators=146, max_depth=20, max_features='sqrt', criterion='entropy',
                                       oob_score=True, random_state=0)
        # model = RandomForestClassifier(n_estimators=141, max_depth=10, max_features='sqrt', criterion='entropy',
        #                                oob_score=True, random_state=0)
        model.fit(self.X, self.y)
        return model

    # Support Vector Machine Model
    def build_model_SVC(self):
        model = SVC(kernel='linear', C=0.025, random_state=0)
        model.fit(self.X, self.y)
        return model


    # Extreme Gradient Boosting Model
    def build_model_XGB(self):
        model = XGBClassifier(n_estimators=1000, learning_rate=0.5)
        model.fit(self.X, self.y)
        return model
    # # Setting model parameters
    # # param = {
    # #     'eta': 0.3,  # Learning rate
    # #     'max_depth': 3,
    # #     'objective': 'multi:softprob',
    # #     'num_class': 3}
    # # Train model
    # model = xgb.sklearn.XGBClassifier(eta=0.3, max_depth=3, objective='multi:softprob', num_class=3, steps=20)
