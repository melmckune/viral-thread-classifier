"""Sanity checks: each from-scratch model learns a simple problem and agrees
with its scikit-learn counterpart where one exists."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from virality.metrics import evaluate_model
from virality.models import KNN, DecisionTree, GaussianNaiveBayes, LinearSVM, LogisticRegression


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=600, n_features=8, n_informative=5,
                               random_state=0, class_sep=1.5)
    return X[:400], y[:400], X[400:500], y[400:500], X[500:], y[500:]


def test_naive_bayes_matches_sklearn(data):
    X_tr, y_tr, _, _, X_te, _ = data
    ours = GaussianNaiveBayes().fit(X_tr, y_tr)
    ref = GaussianNB(var_smoothing=0).fit(X_tr, y_tr)
    np.testing.assert_array_equal(ours.predict(X_te), ref.predict(X_te))
    np.testing.assert_allclose(ours.predict_proba(X_te), ref.predict_proba(X_te), atol=1e-6)


def test_knn_matches_sklearn(data):
    X_tr, y_tr, _, _, X_te, _ = data
    ours = KNN(k=7).fit(X_tr, y_tr)
    ref = KNeighborsClassifier(n_neighbors=7).fit(X_tr, y_tr)
    agreement = np.mean(ours.predict(X_te) == ref.predict(X_te))
    assert agreement > 0.98  # float32 distances can flip exact ties


@pytest.mark.parametrize("make_model", [
    lambda: LogisticRegression(lr=0.1, epochs=300),
    lambda: LinearSVM(lr=1e-2, lambda_reg=1e-3, epochs=30),
])
def test_gradient_models_learn(data, make_model):
    X_tr, y_tr, X_val, y_val, X_te, y_te = data
    model = make_model().fit(X_tr, y_tr, X_val, y_val)
    assert np.mean(model.predict(X_te) == y_te) > 0.8


def test_decision_tree_fits_training_data(data):
    X_tr, y_tr, _, _, X_te, y_te = data
    tree = DecisionTree(max_depth=50, min_split=2).fit(X_tr, y_tr)
    assert np.mean(tree.predict(X_tr) == y_tr) == 1.0
    shallow = DecisionTree(max_depth=4).fit(X_tr, y_tr)
    assert np.mean(shallow.predict(X_te) == y_te) > 0.75
    np.testing.assert_allclose(shallow.predict_proba(X_te).sum(axis=1), 1.0)


def test_evaluate_model():
    y = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    m = evaluate_model(y, y_pred)
    assert m["accuracy"] == pytest.approx(0.6)
    np.testing.assert_array_equal(m["confusion_matrix"], [[1, 1], [1, 2]])
    # class 0: P=0.5 R=0.5 F1=0.5 ; class 1: P=2/3 R=2/3 F1=2/3
    assert m["macro_f1"] == pytest.approx((0.5 + 2 / 3) / 2)
