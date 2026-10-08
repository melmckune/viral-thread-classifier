import numpy as np


class KNN:
    """k-nearest neighbours with chunked, vectorised Euclidean distances.

    Distances are computed in chunks so memory stays bounded at roughly
    chunk_size x n_train floats regardless of the test-set size.
    """

    def __init__(self, k=5, chunk_size=500):
        self.k = k
        self.chunk_size = chunk_size
        self.X_train = None
        self.y_train = None
        self.classes = None

    def fit(self, X, y):
        # float32 halves memory for the distance matrix
        self.X_train = X.astype(np.float32)
        self.y_train = np.asarray(y)
        self.classes = np.unique(self.y_train)
        self._train_sq = np.sum(self.X_train ** 2, axis=1)
        return self

    def neighbor_labels(self, X, k=None):
        """Labels of the k nearest training points, sorted nearest-first."""
        k = k or self.k
        X = X.astype(np.float32)
        out = []
        for start in range(0, X.shape[0], self.chunk_size):
            chunk = X[start:start + self.chunk_size]
            sq_dist = (np.sum(chunk ** 2, axis=1, keepdims=True)
                       + self._train_sq[None, :]
                       - 2 * chunk @ self.X_train.T)
            idx = np.argpartition(sq_dist, k - 1, axis=1)[:, :k]
            order = np.argsort(np.take_along_axis(sq_dist, idx, axis=1), axis=1)
            out.append(self.y_train[np.take_along_axis(idx, order, axis=1)])
        return np.vstack(out)

    def _vote(self, labels):
        counts = np.stack([(labels == c).sum(axis=1) for c in self.classes], axis=1)
        return counts / labels.shape[1]

    def predict_proba(self, X, labels=None):
        labels = self.neighbor_labels(X) if labels is None else labels[:, :self.k]
        return self._vote(labels)

    def predict(self, X, labels=None):
        return self.classes[np.argmax(self.predict_proba(X, labels), axis=1)]
