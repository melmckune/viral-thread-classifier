import numpy as np


class LinearSVM:
    """Soft-margin linear SVM trained on the primal hinge loss with mini-batch SGD."""

    def __init__(self, lr=1e-3, lambda_reg=1e-2, epochs=1000, batch_size=256,
                 seed=42, verbose=False):
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose
        self.W = None
        self.b = 0.0
        self.train_losses, self.val_losses = [], []
        self.train_accs, self.val_accs = [], []

    @staticmethod
    def _to_bipolar(y):
        return np.where(y == 1, 1.0, -1.0)

    def _hinge_loss(self, X, y_bip):
        margins = y_bip * self.decision_function(X)
        return 0.5 * self.lambda_reg * self.W @ self.W + np.mean(np.maximum(0.0, 1.0 - margins))

    def decision_function(self, X):
        return X @ self.W + self.b

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)

    def fit(self, X, y, X_val, y_val):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.W = np.zeros(d)
        self.b = 0.0
        y_bip = self._to_bipolar(y)
        y_val_bip = self._to_bipolar(y_val)

        for epoch in range(self.epochs):
            idx = rng.permutation(n)
            X_s, y_s = X[idx], y_bip[idx]
            for start in range(0, n, self.batch_size):
                Xb = X_s[start:start + self.batch_size]
                yb = y_s[start:start + self.batch_size]
                active = yb * (Xb @ self.W + self.b) < 1  # samples inside the margin

                dW = self.lambda_reg * self.W
                db = 0.0
                if active.any():
                    dW = dW - (yb[active, None] * Xb[active]).sum(axis=0) / len(yb)
                    db = -yb[active].sum() / len(yb)
                self.W -= self.lr * dW
                self.b -= self.lr * db

            self.train_losses.append(self._hinge_loss(X, y_bip))
            self.val_losses.append(self._hinge_loss(X_val, y_val_bip))
            self.train_accs.append(np.mean(self.predict(X) == y))
            self.val_accs.append(np.mean(self.predict(X_val) == y_val))

            if self.verbose and (epoch % 100 == 0 or epoch == self.epochs - 1):
                print(f"    epoch {epoch:4d}  train_loss={self.train_losses[-1]:.4f}  "
                      f"val_loss={self.val_losses[-1]:.4f}  val_acc={self.val_accs[-1]:.4f}")
        return self
