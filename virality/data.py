"""Dataset loading and leakage-free train/validation/test splitting."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "viral-db.csv"
FEATURE_PREFIXES = ("rf", "ef")
LABEL_COL = "class_label"
GROUP_COL = "topic_id"


@dataclass
class DataSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list

    def summary(self) -> str:
        rows = [
            ("train", self.X_train, self.y_train),
            ("val", self.X_val, self.y_val),
            ("test", self.X_test, self.y_test),
        ]
        lines = [f"{len(self.feature_names)} features"]
        for name, X, y in rows:
            lines.append(f"  {name:<5} {X.shape[0]:>6} rows  ({np.mean(y == 1):.1%} viral)")
        return "\n".join(lines)


def load_dataset(path=DEFAULT_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Run `python scripts/download_data.py` or see data/README.md."
        )
    return pd.read_csv(path)


def select_features(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]


def make_splits(df: pd.DataFrame, seed: int = 42) -> DataSplits:
    """Split 60/20/20 by topic_id so no thread appears in more than one split.

    Each thread appears up to ~200 times under different virality-definition
    parameters (alpha, beta, sigma_*), so a plain row-level split would leak
    near-duplicate samples of the same thread into the test set.
    """
    topics = df[GROUP_COL].unique()
    train_topics, rest = train_test_split(topics, test_size=0.4, random_state=seed)
    val_topics, test_topics = train_test_split(rest, test_size=0.5, random_state=seed)

    feature_cols = select_features(df)
    parts = {}
    for name, ids in (("train", train_topics), ("val", val_topics), ("test", test_topics)):
        part = df[df[GROUP_COL].isin(ids)]
        parts[name] = (part[feature_cols].to_numpy(dtype=float), part[LABEL_COL].to_numpy())

    # Fit the scaler on training data only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(parts["train"][0])
    X_val = scaler.transform(parts["val"][0])
    X_test = scaler.transform(parts["test"][0])
    X_train, y_train = shuffle(X_train, parts["train"][1], random_state=seed)

    return DataSplits(
        X_train, y_train,
        X_val, parts["val"][1],
        X_test, parts["test"][1],
        feature_cols,
    )
