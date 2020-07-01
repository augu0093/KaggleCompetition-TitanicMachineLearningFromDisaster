"""
This script tunes hyper-parameter settings of some of the utilized classification models.
@AugustSemrau
"""

from dataLoader import dataLoader








if __name__ == '__main__':

    X_train, X_val, y_train, y_val = dataLoader(test=False, optimize_set=True, ageNAN="median")

    print(X_train)



