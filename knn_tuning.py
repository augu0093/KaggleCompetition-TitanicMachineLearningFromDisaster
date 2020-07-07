"""
This script is a low-effort tuning of K-Nearest-Neighbor model.
@AugustSemrau
"""

from data_loader import dataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd



if __name__ == '__main__':

    # Get data
    X, y = dataLoader(test=False, optimize_set=False)
    y = y.values.ravel()
    # First make list of k's
    neighbors = list(range(1, 25, 1))
    # Second make empty list for scores
    cval_scores = []

    # Now do 10-fold cross-validation
    for K in neighbors:
        model_KNN = KNeighborsClassifier(n_neighbors=K)
        scores = cross_val_score(model_KNN, X, y, cv=10, scoring='accuracy')
        cval_scores.append(scores.mean())

    # Plotting these score to se how different number of neighbors affect performance
    def plot_acc(knn_scores):
        pd.DataFrame({"K": [i for i in neighbors], "Accuracy": knn_scores}).set_index("K").plot.bar(rot=0)
        plt.show()
    plot_acc(cval_scores)
