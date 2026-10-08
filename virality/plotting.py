"""Figures for the README / results folder (static PNGs, light background)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import auc, precision_recall_curve, roc_curve  # noqa: E402

# Fixed categorical order: a model keeps its colour in every figure.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.titlecolor": TEXT,
    "axes.titleweight": "semibold",
    "axes.titlesize": 13,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "legend.frameon": False,
    "legend.labelcolor": TEXT,
    "font.size": 10.5,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
})


def color_for(name, names):
    return SERIES_COLORS[list(names).index(name) % len(SERIES_COLORS)]


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_roc_curves(results, y_true, path):
    names = list(results)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for name in names:
        fpr, tpr, _ = roc_curve(y_true, results[name]["scores"])
        ax.plot(fpr, tpr, lw=2, color=color_for(name, names),
                label=f"{name}  (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=TEXT_SECONDARY, label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="ROC curves - held-out test threads", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="lower right")
    _save(fig, path)


def plot_pr_curves(results, y_true, path):
    names = list(results)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for name in names:
        precision, recall, _ = precision_recall_curve(y_true, results[name]["scores"])
        ax.plot(recall, precision, lw=2, color=color_for(name, names),
                label=f"{name}  (AUC {auc(recall, precision):.3f})")
    base = np.mean(y_true == 1)
    ax.axhline(base, ls="--", lw=1, color=TEXT_SECONDARY, label=f"Base rate ({base:.2f})")
    ax.set(xlabel="Recall", ylabel="Precision",
           title="Precision-recall curves - held-out test threads", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="upper right")
    _save(fig, path)


def plot_model_comparison(results, path, metric="macro_f1", label="Macro F1"):
    names = sorted(results, key=lambda n: results[n]["metrics"][metric])
    values = [results[n]["metrics"][metric] for n in names]
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(names) + 1.2))
    ax.barh(names, values, color=SERIES_COLORS[0], height=0.6)
    for i, v in enumerate(values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", color=TEXT, fontsize=10)
    ax.set_xlim(0, max(values) * 1.12)
    ax.set_xlabel(f"{label} on test set")
    ax.set_title(f"Model comparison - {label}")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def plot_confusion_matrices(results, path):
    names = list(results)
    cols = min(3, len(names))
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.2 * rows), squeeze=False)
    for ax, name in zip(axes.flat, names):
        cm = results[name]["metrics"]["confusion_matrix"]
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v}\n{norm[i, j]:.0%}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.55 else TEXT, fontsize=10)
        ax.set_xticks([0, 1], ["Not viral", "Viral"])
        ax.set_yticks([0, 1], ["Not viral", "Viral"])
        ax.set_title(name, fontsize=11)
        ax.grid(False)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    fig.supxlabel("Predicted", color=TEXT_SECONDARY)
    fig.supylabel("Actual", color=TEXT_SECONDARY)
    fig.tight_layout()
    _save(fig, path)


def plot_training_curves(histories, path):
    """histories: {model name: (train_loss, val_loss, train_acc, val_acc)}"""
    fig, axes = plt.subplots(len(histories), 2, figsize=(10, 3.4 * len(histories)), squeeze=False)
    for row, (name, (tl, vl, ta, va)) in zip(axes, histories.items()):
        for ax, train, val, ylabel in ((row[0], tl, vl, "Loss"), (row[1], ta, va, "Accuracy")):
            ax.plot(train, lw=2, color=SERIES_COLORS[0], label="Train")
            ax.plot(val, lw=2, color=SERIES_COLORS[1], label="Validation")
            ax.set(xlabel="Epoch", ylabel=ylabel, title=f"{name} - {ylabel.lower()}")
            ax.legend()
    fig.tight_layout()
    _save(fig, path)


def plot_feature_importance(importances, feature_names, path, top_n=15):
    order = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(order) + 1.2))
    ax.barh(np.array(feature_names)[order], importances[order], color=SERIES_COLORS[0], height=0.6)
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title(f"Random Forest - top {len(order)} features")
    ax.grid(axis="y", visible=False)
    _save(fig, path)
