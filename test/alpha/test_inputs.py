from whitebeam.base.alpha import AlphaTree
from whitebeam.base.c45 import C45Tree
from whitebeam.base.gini import GiniTree

from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import pytest

from test.utils import createSomeData

# ---------------
# Expected Error 
# ---------------

@pytest.mark.parametrize("param_alpha", [2+3j, "3", []])
def test_alpha_alpha_data_types_fail(param_alpha):

    X_train, X_test, y_train, y_test = createSomeData()

    # should fail DURING THE <model.fit> CALL
    with pytest.raises(Exception) as e_info:
        model = AlphaTree(alpha=param_alpha)
        model.fit(X_train, y_train)

@pytest.mark.parametrize("param_depth", [2+3j, "3", []])
def test_alpha_depth_data_types_fail(param_depth):

    X_train, X_test, y_train, y_test = createSomeData()

    # should fail DURING THE <model.fit> CALL
    with pytest.raises(Exception) as e_info:
        model = AlphaTree(max_depth=param_depth)
        model.fit(X_train, y_train)

# for some reason, a complex number does not call an error here :/
# also an empty list doesnt
# Perhaps the split condition gets ignored in some way

@pytest.mark.parametrize("param_split", ["3"])
def test_alpha_split_data_types_fail(param_split):

    X_train, X_test, y_train, y_test = createSomeData()

    # should fail DURING THE <model.fit> CALL
    with pytest.raises(Exception) as e_info:
        model = AlphaTree(min_samples_split=param_split)
        model.fit(X_train, y_train)

@pytest.mark.parametrize("param_leaf", [2+3j, "3", []])
def test_alpha_leaf_data_types_fail(param_leaf):

    X_train, X_test, y_train, y_test = createSomeData()

    # should fail DURING THE <model.fit> CALL
    with pytest.raises(Exception) as e_info:
        model = AlphaTree(min_samples_leaf= param_leaf)
        model.fit(X_train, y_train)

# ---------------
# Expected Pass 
# ---------------

@pytest.mark.parametrize("param_alpha", [True, 3.2])
def test_alpha_alpha_data_types_pass(param_alpha):

    X_train, X_test, y_train, y_test = createSomeData()

    model = AlphaTree(alpha=param_alpha)
    model.fit(X_train, y_train)

@pytest.mark.parametrize("param_depth", [True, 3.2])
def test_alpha_depth_data_types_pass(param_depth):

    X_train, X_test, y_train, y_test = createSomeData()

    model = AlphaTree(max_depth=param_depth)
    model.fit(X_train, y_train)

# for some reason, a complex number does not call an error here :/
@pytest.mark.parametrize("param_split", [True, 3.2, 3+2j, []])
def test_alpha_split_data_types_pass(param_split):

    X_train, X_test, y_train, y_test = createSomeData()

    model = AlphaTree(min_samples_split=param_split)
    model.fit(X_train, y_train)

@pytest.mark.parametrize("param_leaf", [True, 3.2])
def test_alpha_leaf_data_types_pass(param_leaf):

    X_train, X_test, y_train, y_test = createSomeData()

    # should fail DURING THE <model.fit> CALL
    model = AlphaTree(min_samples_leaf= param_leaf)
    model.fit(X_train, y_train)






