from whitebeam.base.alpha import AlphaTree
from whitebeam.base.c45 import C45Tree
from whitebeam.base.gini import GiniTree
from whitebeam.base.regr import RegTree
from whitebeam.base.xgb import XGBTree

from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import pytest

def test_c45_auc():

    X_train, X_test, y_train, y_test = createSomeData()

    max_depth = 4
    model = C45Tree(max_depth=max_depth)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    auc = roc_auc_score(y_test, y_hat)
    
    assert auc > 0.5

def test_gini_auc():

    X_train, X_test, y_train, y_test = createSomeData()

    max_depth = 4
    model = GiniTree(max_depth=max_depth)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    auc = roc_auc_score(y_test, y_hat)

    assert auc > 0.5
    
def test_alpha_auc():

    X_train, X_test, y_train, y_test = createSomeData()

    max_depth = 4
    model = AlphaTree(alpha=3.0, max_depth=max_depth)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    auc = roc_auc_score(y_test, y_hat)

    assert auc > 0.5

def test_regression_auc():

    X_train, X_test, y_train, y_test = createSomeData()

    max_depth = 4
    model = RegTree(max_depth=max_depth)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    auc = roc_auc_score(y_test, y_hat)

    assert auc > 0.5

def test_xgboost_auc():

    X_train, X_test, y_train, y_test = createSomeData()

    max_depth = 4
    model = XGBTree(max_depth=max_depth)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    auc = roc_auc_score(y_test, y_hat)

    assert auc > 0.5

def createSomeData():

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

                                            


    


