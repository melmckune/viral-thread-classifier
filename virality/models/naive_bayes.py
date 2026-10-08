import numpy as np


class GaussianNaiveBayes:
    """Gaussian Naive Bayes with per-class feature means/variances."""

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.classes = None
        self.log_priors = None
        self.means = None
        self.vars = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.log_priors = np.log([np.mean(y == c) for c in self.classes])
        self.means = np.array([X[y == c].mean(axis=0) for c in self.classes])
        self.vars = np.array([X[y == c].var(axis=0) for c in self.classes]) + self.var_smoothing
        return self

    def _joint_log_likelihood(self, X):
        # (n_samples, n_classes): log P(c) + sum_j log N(x_j | mu_cj, var_cj)
        diff = X[:, None, :] - self.means[None, :, :]
        log_pdf = -0.5 * (np.log(2 * np.pi * self.vars)[None] + diff ** 2 / self.vars[None])
        return self.log_priors[None, :] + log_pdf.sum(axis=2)

    def predict_proba(self, X):
        scores = self._joint_log_likelihood(X)
        scores -= scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes[np.argmax(self._joint_log_likelihood(X), axis=1)]
