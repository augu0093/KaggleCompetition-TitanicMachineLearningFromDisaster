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

from xgboost import XGBClassifier

class Models():
    """Class containing all (sklearn-built) models"""

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

    # Extreme Gradient Boosting Model
    def build_model_XGBoost(self):
        model = XGBClassifier()
        model.fit(self.X, self.y)
        return model