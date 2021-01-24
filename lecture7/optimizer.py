import warnings

import numpy as np
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.datasets import fetch_openml

if __name__ == "__main__":
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

    X = X / 255.
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    mlp_sgd = MLPClassifier(hidden_layer_sizes=(50, ), max_iter=10, alpha=1e-4,
    solver='sgd', verbose=10, random_state=1, learning_rate_init=.1)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module='sklearn')
        mlp_sgd.fit(X_train, y_train)