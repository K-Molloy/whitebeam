from whitebeam.base import *

from sklearn.metrics import accuracy_score
import numpy as np
import pytest

from test.utils import generate_binary_classification

def test_c45_auc():

    X_train, X_test, y_train, y_test = generate_binary_classification()

    max_depth = 4
    model = C45TreeClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    score = accuracy_score(y_test, y_hat)

    assert score > 0.8

def test_cart_auc():

    X_train, X_test, y_train, y_test = generate_binary_classification()

    max_depth = 4
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    score = accuracy_score(y_test, y_hat)

    print(y_test)
    print(y_hat)

    assert score > 0.8
    
def test_alpha_auc():

    X_train, X_test, y_train, y_test = generate_binary_classification()

    max_depth = 4
    model = CCPTreeClassifier(alpha=3.0, max_depth=max_depth)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    score = accuracy_score(y_test, y_hat)

    assert score > 0.8

def test_regression_auc():

    X_train, X_test, y_train, y_test = generate_binary_classification()

    max_depth = 4
    model = DecisionTreeRegressor(max_depth=max_depth)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    score = accuracy_score(y_test, y_hat)

    assert score > 0.8

def test_xgboost_auc():

    X_train, X_test, y_train, y_test = generate_binary_classification()

    max_depth = 4
    model = XGBoostedClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    score = accuracy_score(y_test, y_hat)

    assert score > 0.8




    


