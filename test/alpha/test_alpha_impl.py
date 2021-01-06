from whitebeam.base.alpha import AlphaTree
from whitebeam.base.c45 import C45Tree
from whitebeam.base.gini import GiniTree

from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import pytest

from test.utils import createSomeData

# Vary the depth of the alpha tree search, 1, 5, and 10 determined as
# significant intervals. Initially 20 and 50 were considered too, but
# computation took 26s and 106s. Depth 10 continues to be the 3rd most
# expensive test (2.05s) and may be removed soon
@pytest.mark.parametrize("param_depth", [1, 5, 10])
def test_alpha_depth(param_depth):

    X_train, X_test, y_train, y_test = createSomeData()

    model = AlphaTree(alpha=3.0, max_depth=param_depth)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    auc = roc_auc_score(y_test, y_hat)

    assert auc != 0

# Varying the alpha value for the alpha tree search
# 1 is a regression tree
# 2 is a CART tree
# 3 is a common alpha tree
# 4 is a ?
@pytest.mark.parametrize("param_alpha", [1, 2, 3, 4])
def test_alpha_alpha(param_alpha):

    X_train, X_test, y_train, y_test = createSomeData()

    model = AlphaTree(alpha=param_alpha)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    auc = roc_auc_score(y_test, y_hat)

    assert auc != 0

# Test common split and leaf values
@pytest.mark.parametrize("param_split, param_leaf, ", [(1, 1), (1, 2), (2, 1), (2, 2), (3, 3)])
def test_alpha_split_leaf(param_split, param_leaf):

    X_train, X_test, y_train, y_test = createSomeData()
    model = AlphaTree(alpha=0.3,
                      max_depth=4,
                      min_samples_split=param_split,
                      min_samples_leaf= param_leaf)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    auc = roc_auc_score(y_test, y_hat)

    assert auc != 0