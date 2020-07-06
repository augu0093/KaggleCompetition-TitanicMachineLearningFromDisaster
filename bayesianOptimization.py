"""
This script tunes hyper-parameter settings of some of the utilized classification models.
@AugustSemrau
"""

from dataLoader import dataLoader
from sklearn.ensemble import RandomForestClassifier
import GPyOpt


# Baseline default-setting Random Forest Model for comparison with optimized model
def build_baseline_randomForest(X, y):
    model = RandomForestClassifier(oob_score=True)
    model.fit(X, y)
    return model

# Random Forest Model for optimization
def build_model_randomForest(self):
    model = RandomForestClassifier(n_estimators=50, max_depth=20, oob_score=True, n_jobs=1, random_state=0, max_features=None, min_samples_leaf=30)
    model.fit(self.X, self.y)
    return model



if __name__ == '__main__':

    X_train, X_val, y_train, y_val = dataLoader(test=False, optimize_set=True, ageNAN="median")

    baselineRF = build_baseline_randomForest(X_train, y_train)
    print('Default model out-of-the-bag error', baselineRF.oob_score_)
    print('Default model test accuracy', baselineRF.score(X_val, y_val))











