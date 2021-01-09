from whitebeam.ensemble.randomforest import RandomForests
from whitebeam.base.gini import GiniTree

from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import pytest

from test.utils import createSomeData


def test_rf_gini():

    test_size = 0.2
    n_est = 10
    params = {"max_depth": 4}

    X_train, X_test, y_train, y_test = createSomeData()

    model = RandomForests(base_estimator=GiniTree,
                            base_params=params,
                            n_estimators=n_est)

    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    mse_rf = np.mean((y_test - y_hat)**2)
    mse_baseline = np.mean((y_test - np.mean(y_train))**2)

    assert(mse_rf < mse_baseline)