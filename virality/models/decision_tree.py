import numpy as np


class Node:
    def __init__(self, feat_idx, threshold, left, right):
        self.feat_idx = feat_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.is_leaf = False


class LeafNode:
    def __init__(self, label, probs):
        self.label = label
        self.probs = probs
        self.is_leaf = True


class DecisionTree:
    """CART classification tree using Gini impurity.

    Split search is vectorised: for each feature the samples are sorted once
    and cumulative class counts give the Gini impurity of every candidate
    threshold in O(n log n), instead of re-partitioning the data per threshold.
    """

    def __init__(self, max_depth=10, min_split=2):
        self.max_depth = max_depth
        self.min_split = min_split
        self.classes = None
        self.root = None

    def _leaf(self, y_idx):
        counts = np.bincount(y_idx, minlength=len(self.classes))
        probs = counts / counts.sum()
        return LeafNode(self.classes[np.argmax(counts)], probs)

    def _best_split(self, X, y_idx):
        n, d = X.shape
        k = len(self.classes)
        best_gini, best_feat, best_thresh = np.inf, None, None

        for feat in range(d):
            order = np.argsort(X[:, feat], kind="stable")
            x_sorted = X[order, feat]
            onehot = np.zeros((n, k))
            onehot[np.arange(n), y_idx[order]] = 1

            left_counts = np.cumsum(onehot, axis=0)[:-1]   # split after position i
            right_counts = left_counts[-1] + onehot[-1] - left_counts
            n_left = np.arange(1, n)
            n_right = n - n_left

            gini_left = 1 - np.sum((left_counts / n_left[:, None]) ** 2, axis=1)
            gini_right = 1 - np.sum((right_counts / n_right[:, None]) ** 2, axis=1)
            weighted = (n_left * gini_left + n_right * gini_right) / n

            # Only split between distinct feature values
            valid = x_sorted[:-1] < x_sorted[1:]
            if not valid.any():
                continue
            weighted[~valid] = np.inf
            i = np.argmin(weighted)
            if weighted[i] < best_gini:
                best_gini = weighted[i]
                best_feat = feat
                best_thresh = (x_sorted[i] + x_sorted[i + 1]) / 2

        return best_feat, best_thresh

    def _build(self, X, y_idx, depth):
        if depth >= self.max_depth or len(y_idx) < self.min_split or np.all(y_idx == y_idx[0]):
            return self._leaf(y_idx)

        feat, thresh = self._best_split(X, y_idx)
        if feat is None:
            return self._leaf(y_idx)

        mask = X[:, feat] <= thresh
        return Node(
            feat, thresh,
            self._build(X[mask], y_idx[mask], depth + 1),
            self._build(X[~mask], y_idx[~mask], depth + 1),
        )

    def fit(self, X, y):
        self.classes, y_idx = np.unique(y, return_inverse=True)
        self.root = self._build(X, y_idx, depth=0)
        return self

    def _leaf_for(self, x):
        node = self.root
        while not node.is_leaf:
            node = node.left if x[node.feat_idx] <= node.threshold else node.right
        return node

    def predict(self, X):
        return np.array([self._leaf_for(x).label for x in X])

    def predict_proba(self, X):
        return np.array([self._leaf_for(x).probs for x in X])
