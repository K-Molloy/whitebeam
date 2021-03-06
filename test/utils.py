from sklearn.datasets import make_hastie_10_2, make_classification, make_multilabel_classification
from sklearn.model_selection import train_test_split

import numpy as np

def generate_hastie_10_2():

    n_samples = 100000
    test_size = 0.2

    # Hastie_10_2
    # X_i ~ Gaussian
    # sum of X_i^2 > Chi-squire(10, 0.5) 9.34, then 1, otherwise -1
    X, y_org = make_hastie_10_2(n_samples=n_samples) 
    z = np.random.randn(n_samples)
    y = y_org * z
    y[y > 0] = 1
    y[y <= 0] = 0
    X = np.hstack((X, z.reshape(n_samples,1)))
    n, m = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test


def generate_binary_classification():

    n_samples = 1000
    n_features = 50
    test_size = 0.2

    X, y = make_classification(n_samples=n_samples,
                               n_features=n_features,
                               shuffle=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test


def generate_multilabel_classification():

    n_samples = 1000
    n_classes = 3
    n_labels = 1
    test_size = 0.2

    X, y = make_classification(n_samples=n_samples, 
                               n_classes=n_classes,
                               n_features=30,
                               n_informative=4)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test