"""
Building predictive model on regression principles.
@AugustSemrau
"""
import numpy as np

from dataLoader import dataLoader

# SciKit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import xgboost as xgb

class Models():
    """Class containing all classification models"""

    # Init
    def __init__(self, agenan="median"):
        self.X, self.y = dataLoader(test=False, optimize_set=False, ageNAN=agenan)
        self.y = np.ravel(self.y)

    # Logistic Regression Model
    def build_model_logisticReg(self):
        model = LogisticRegression(random_state=0)
        model.fit(self.X, self.y)
        return model

    # Naive Bayes Model
    def build_model_naiveBayes(self):
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
    def build_model_decisionTree(self):
        model = DecisionTreeClassifier(max_depth=10, random_state=0, max_features=None, min_samples_leaf=15)
        model.fit(self.X, self.y)
        return model

    # Random Forest Model
    def build_model_randomForest(self):
        model = RandomForestClassifier(n_estimators=50, oob_score=True, n_jobs=1, random_state=0, max_features=None, min_samples_leaf=30)
        model.fit(self.X, self.y)
        return model

    # Support Vector Machine Model
    def build_model_SVC(self):
        model = SVC(kernel='linear', C=0.025, random_state=0)
        model.fit(self.X, self.y)
        return model


    # Extreme Gradient Boosting Model
    def build_model_XGBoost(self):
        # Redefining data for use in XGB
        xgb_train_data = xgb.DMatrix(self.X.to_numpy(), label=self.y)
        # Setting model parameters
        # param = {
        #     'eta': 0.3,  # Learning rate
        #     'max_depth': 3,
        #     'objective': 'multi:softprob',
        #     'num_class': 3}
        # Train model
        model = xgb.sklearn.XGBClassifier(eta=0.3, max_depth=3, objective='multi:softprob', num_class=3, steps=20)
        model.fit(self.X.to_numpy(), self.y)
        # model.train(xgb_train_data)
        return model




