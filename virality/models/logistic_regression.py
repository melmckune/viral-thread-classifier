import numpy as np


class LogisticRegression:
    """Softmax regression trained with full-batch gradient descent.

    Supports L1/L2 regularisation and early stopping on validation loss.
    """

    def __init__(self, lr=0.1, epochs=1000, lambda_val=0.0, reg_type="l2",
                 patience=50, verbose=False):
        self.lr = lr
        self.epochs = epochs
        self.lambda_val = lambda_val
        self.reg_type = reg_type
        self.patience = patience
        self.verbose = verbose
        self.W = None
        self.b = None
        self.losses, self.accuracies = [], []
        self.val_losses, self.val_accuracies = [], []

    @staticmethod
    def softmax(Z):
        Z = Z - np.max(Z, axis=1, keepdims=True)
        expZ = np.exp(Z)
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    @staticmethod
    def one_hot(y, num_classes):
        out = np.zeros((len(y), num_classes))
        out[np.arange(len(y)), y] = 1
        return out

    @staticmethod
    def cross_entropy(probs, y_one_hot):
        probs = np.clip(probs, 1e-15, 1.0)
        return -np.mean(np.sum(y_one_hot * np.log(probs), axis=1))

    def _reg_loss(self, n):
        if self.reg_type == "l2":
            return (self.lambda_val / (2 * n)) * np.sum(self.W ** 2)
        if self.reg_type == "l1":
            return (self.lambda_val / n) * np.sum(np.abs(self.W))
        return 0.0

    def _reg_grad(self, n):
        if self.reg_type == "l2":
            return (self.lambda_val / n) * self.W
        if self.reg_type == "l1":
            return (self.lambda_val / n) * np.sign(self.W)
        return 0.0

    def predict_proba(self, X):
        return self.softmax(X @ self.W + self.b)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def fit(self, X, y, X_val, y_val):
        y = y.astype(int)
        y_val = y_val.astype(int)
        n, d = X.shape
        k = len(np.unique(y))
        self.W = np.zeros((d, k))
        self.b = np.zeros(k)
        Y = self.one_hot(y, k)
        Y_val = self.one_hot(y_val, k)

        best_val_loss, wait = np.inf, 0
        best_W, best_b = self.W.copy(), self.b.copy()
        for epoch in range(self.epochs):
            probs = self.predict_proba(X)
            val_probs = self.predict_proba(X_val)
            reg = self._reg_loss(n)
            loss = self.cross_entropy(probs, Y) + reg
            val_loss = self.cross_entropy(val_probs, Y_val) + reg

            self.losses.append(loss)
            self.val_losses.append(val_loss)
            self.accuracies.append(np.mean(np.argmax(probs, axis=1) == y))
            self.val_accuracies.append(np.mean(np.argmax(val_probs, axis=1) == y_val))

            if self.verbose and epoch % 50 == 0:
                print(f"    epoch {epoch:4d}  loss={loss:.4f}  val_loss={val_loss:.4f}  "
                      f"val_acc={self.val_accuracies[-1]:.4f}")

            if val_loss < best_val_loss:
                best_val_loss, wait = val_loss, 0
                best_W, best_b = self.W.copy(), self.b.copy()
            else:
                wait += 1
                if wait >= self.patience:
                    if self.verbose:
                        print(f"    early stopping at epoch {epoch}")
                    break

            self.W -= self.lr * (X.T @ (probs - Y) / n + self._reg_grad(n))
            self.b -= self.lr * np.mean(probs - Y, axis=0)

        # Restore the weights with the lowest validation loss
        self.W, self.b = best_W, best_b
        return self
