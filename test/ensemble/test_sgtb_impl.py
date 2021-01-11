from whitebeam.ensemble import SGTB
from whitebeam.base import DecisionTreeClassifier

import numpy as np
import pytest

from sklearn.metrics import roc_auc_score


from test.utils import generate_binary_classification


# Check that model.predict outputs an array y_hat is length 1000
def test_sgtb_output():

    X_train, X_test, y_train, y_test = generate_binary_classification()

    n_est = 100
    max_depth = 4
    learning_rate = 0.1

    model = SGTB(distribution="bernoulli",
                n_estimators=n_est, 
                learning_rate=learning_rate,
                max_depth=max_depth)
    y_hat = model.predict(X_test)

    # 200 is size of y_test because using 80/20 split
    assert len(y_hat) == 200


def test_sgtb_labels():

    X_train, X_test, y_train, y_test = generate_binary_classification()

    n_est = 100
    max_depth = 4
    learning_rate = 0.1

    model = SGTB(distribution="bernoulli",
                n_estimators=n_est, 
                learning_rate=learning_rate,
                max_depth=max_depth)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)

    # 200 is size of y_test because using 80/20 split
    assert set([0, 1]).issuperset(set(y_hat))
    # assert any(label in method([0, 1]) for label in y_hat)


# Vary the depth of the alpha tree search, 1, 5, and 10 determined as
# significant intervals. Initially 20 and 50 were considered too, but
# computation took 26s and 106s. Depth 10 continues to be the 3rd most
# expensive test (2.05s) and may be removed soon
@pytest.mark.parametrize("param_estimators", [50, 100, 150])
def test_sgtb_estimators(param_estimators):

    X_train, X_test, y_train, y_test = generate_binary_classification()

    max_depth = 4
    learning_rate = 0.1
    

    model = SGTB(distribution="bernoulli",
                n_estimators=param_estimators, 
                learning_rate=learning_rate,
                max_depth=max_depth)

    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    auc = roc_auc_score(y_test, y_hat)

    assert auc != 0
