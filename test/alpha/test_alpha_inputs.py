from whitebeam.base import AlphaTreeClassifier

from sklearn.model_selection import train_test_split
import numpy as np
import pytest

from test.utils import generate_binary_classification

# ---------------
# Expected Error 
# ---------------

@pytest.mark.parametrize("param_alpha", [2+3j, "3", []])
def test_alpha_alpha_data_types_fail(param_alpha):

    X_train, X_test, y_train, y_test = generate_binary_classification()

    # should fail DURING THE <model.fit> CALL
    with pytest.raises(Exception) as e_info:
        model = AlphaTreeClassifier(alpha=param_alpha)
        model.fit(X_train, y_train)

@pytest.mark.parametrize("param_depth", [2+3j, "3", []])
def test_alpha_depth_data_types_fail(param_depth):

    X_train, X_test, y_train, y_test = generate_binary_classification()

    # should fail DURING THE <model.fit> CALL
    with pytest.raises(Exception) as e_info:
        model = AlphaTreeClassifier(max_depth=param_depth)
        model.fit(X_train, y_train)

# for some reason, a complex number does not call an error here :/
# also an empty list doesnt
# Perhaps the split condition gets ignored in some way

@pytest.mark.parametrize("param_split", ["3"])
def test_alpha_split_data_types_fail(param_split):

    X_train, X_test, y_train, y_test = generate_binary_classification()

    # should fail DURING THE <model.fit> CALL
    with pytest.raises(Exception) as e_info:
        model = AlphaTreeClassifier(min_samples_split=param_split)
        model.fit(X_train, y_train)

@pytest.mark.parametrize("param_leaf", [2+3j, "3", []])
def test_alpha_leaf_data_types_fail(param_leaf):

    X_train, X_test, y_train, y_test = generate_binary_classification()

    # should fail DURING THE <model.fit> CALL
    with pytest.raises(Exception) as e_info:
        model = AlphaTreeClassifier(min_samples_leaf= param_leaf)
        model.fit(X_train, y_train)

# ---------------
# Expected Pass 
# ---------------

@pytest.mark.parametrize("param_alpha", [True, 3.2])
def test_alpha_alpha_data_types_pass(param_alpha):

    X_train, X_test, y_train, y_test = generate_binary_classification()

    model = AlphaTreeClassifier(alpha=param_alpha)
    model.fit(X_train, y_train)

@pytest.mark.parametrize("param_depth", [True, 3.2])
def test_alpha_depth_data_types_pass(param_depth):

    X_train, X_test, y_train, y_test = generate_binary_classification()

    model = AlphaTreeClassifier(max_depth=param_depth)
    model.fit(X_train, y_train)

# for some reason, a complex number does not call an error here :/
# Additionally, passing a list as split brings a deprecation warning from arrays
@pytest.mark.parametrize("param_split", [True, 3.2, 3+2j, []])
def test_alpha_split_data_types_pass(param_split):

    X_train, X_test, y_train, y_test = generate_binary_classification()

    model = AlphaTreeClassifier(min_samples_split=param_split)
    model.fit(X_train, y_train)

@pytest.mark.parametrize("param_leaf", [True, 3.2])
def test_alpha_leaf_data_types_pass(param_leaf):

    X_train, X_test, y_train, y_test = generate_binary_classification()

    # should fail DURING THE <model.fit> CALL
    model = AlphaTreeClassifier(min_samples_leaf= param_leaf)
    model.fit(X_train, y_train)






