"""
This script tunes hyper-parameter settings of some of the utilized classification models.
@AugustSemrau
"""

from dataLoader import dataLoader
from sklearn.ensemble import RandomForestClassifier
import GPyOpt
import numpy as np

# Baseline default-setting Random Forest Model for comparison with optimized model
def build_baseline_RF(X, y):
    model = RandomForestClassifier(oob_score=True, random_state=0)
    model.fit(X, y)
    return model


""" Bayesian Optimization
For Random Forest we are looking to optimize:
n_estimators
max_depth
max_features
criterion
"""

# First the domain of considered parameters is defined
n_estimators = tuple(np.arange(1, 151, 1, dtype=np.int))
max_depth = tuple(np.arange(10, 110, 10, dtype=np.int))
max_features = (0, 1)  # max_features = ('log2', 'sqrt', None)
criterion = (0, 1)  # criterion = ('gini', 'entropy')

# Defining the dictionary of domain for GPyOpt
domain_RF = [{'name': 'n_estimators', 'type': 'discrete', 'domain': n_estimators},
          {'name': 'max_depth', 'type': 'discrete', 'domain': max_depth},
          {'name': 'max_features', 'type': 'categorical', 'domain': max_features},
          {'name': 'criterion', 'type': 'categorical', 'domain': criterion}]

# Defining objective function for RF which is the oob.score, which we want to maximize
def objective_function(domain):
    # The two paramteters max_features and criterion are converted from binary 0/1 to labels log2/sqrt and gini/entropy
    params = domain[0]
    # print(params)

    # Defining max_features from params
    if params[2] == 0:
        max_f = 'log2'
    elif params[2] == 1:
        max_f = 'sqrt'
    else:
        max_f = None

    # Define criterion from binary params
    # crit = ['gini', 'entropy'][params[3]]
    if params[3] == 0:  # criterion
        crit = 'gini'
    else:
        crit = 'entropy'


    # Create the RF model
    model = RandomForestClassifier(n_estimators=int(params[0]),
                                   max_depth=int(params[1]),
                                   max_features=max_f,
                                   criterion=crit,
                                   oob_score=True,
                                   n_jobs=-1)

    # fit the model
    model.fit(X_train, y_train)
    # print(model.oob_score_)
    return - model.oob_score_

# Random Forest Model for optimization
def build_optimized_RF(X, y, d0, d1, d2, d3):
    model = RandomForestClassifier(n_estimators=d0, max_depth=d1, max_features=d2, criterion=d3, oob_score=True, random_state=0)
    model.fit(X, y)
    return model





if __name__ == '__main__':

    # Defining data
    X_train, X_val, y_train, y_val = dataLoader(test=False, optimize_set=True, ageNAN="median")

    # Building baseline RF model
    baselineRF = build_baseline_RF(X=X_train, y=y_train)
    print('Default RF model out-of-the-bag error', baselineRF.oob_score_)
    print('Default RF model test accuracy', baselineRF.score(X_val, y_val))

    # Performing Bayesian optimization to optimize RF model parameters
    # Acquisition functions can be MPI, EI or LCB
    opt = GPyOpt.methods.BayesianOptimization(f=objective_function, domain=domain_RF, acquisition_type='EI')
    opt.acquisition.exploration_weight = 0.5
    opt.run_optimization(max_iter=15)
    domain_best = opt.X[np.argmin(opt.Y)]

    # Printing best model parameters
    best_n_estimators = int(domain_best[0])
    best_max_depth = int(domain_best[1])
    best_max_features = ['log2', 'sqrt'][int(domain_best[2])]
    best_criterion = ['gini', 'entropy'][int(domain_best[3])]
    print("The best parameters obtained: n_estimators=" + str(best_n_estimators) + ", max_depth=" + str(best_max_depth)
          + ", max_features=" + best_max_features + ", criterion=" + best_criterion)

    # Printing model score with these parameters
    optimizedRF = build_optimized_RF(X=X_train, y=y_train,
                                     d0=best_n_estimators, d1=best_max_depth, d2=best_max_features, d3=best_criterion)
    print('Default RF model out-of-the-bag error', optimizedRF.oob_score_)
    print('Default RF model test accuracy', optimizedRF.score(X_val, y_val))





