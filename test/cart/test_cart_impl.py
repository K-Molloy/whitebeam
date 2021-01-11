from whitebeam.base import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

import numpy as np
import pytest

from test.utils import generate_binary_classification


# Check that model.predict outputs an array y_hat is length 1000
def test_cart_output():

    X_train, X_test, y_train, y_test = generate_binary_classification()

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)

    # 200 is size of y_test because using 80/20 split
    assert len(y_hat) == 200


def test_cart_labels():

    X_train, X_test, y_train, y_test = generate_binary_classification()

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)

    # 200 is size of y_test because using 80/20 split
    assert set([0, 1]) == set(y_hat)


# Vary the depth of the alpha tree search, 1, 5, and 10 determined as
# significant intervals. Initially 20 and 50 were considered too, but
# computation took 26s and 106s. Depth 10 continues to be the 3rd most
# expensive test (2.05s) and may be removed soon
@pytest.mark.parametrize("param_depth", [1, 5, 10])
def test_cart_depth(param_depth):

    X_train, X_test, y_train, y_test = generate_binary_classification()

    model = DecisionTreeClassifier(max_depth=param_depth)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    score = accuracy_score(y_test, y_hat)

    assert score > 0.8

# Test common split and leaf values
@pytest.mark.parametrize("param_split, param_leaf, ", [(1, 1), (1, 2), (2, 1), (2, 2), (3, 3)])
def test_cart_split_leaf(param_split, param_leaf):

    X_train, X_test, y_train, y_test = generate_binary_classification()
    model = DecisionTreeClassifier(max_depth=4,
                      min_samples_split=param_split,
                      min_samples_leaf= param_leaf)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    score = accuracy_score(y_test, y_hat)

    assert score > 0.8
