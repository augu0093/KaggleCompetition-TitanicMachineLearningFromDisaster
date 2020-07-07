"""
This script is experimenting with the XGBoost model.
@AugustSemrau
"""

from data_loader import dataLoader
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


if __name__ == '__main__':

    # Get data
    X_train, X_val, y_train, y_val = dataLoader(test=False, optimize_set=True)
    y_train, y_val = y_train.values.ravel(), y_val.values.ravel()

    # Define baseline XGB classifier model with default parameters
    defualt_xgb = XGBClassifier()
    defualt_xgb.fit(X_train, y_train)
    default_predictions = defualt_xgb.predict(X_val)
    print('Default XGB model test accuracy', accuracy_score(y_val, default_predictions))








