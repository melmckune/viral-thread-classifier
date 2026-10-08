"""Classification metrics implemented with NumPy."""

import numpy as np


def evaluate_model(y, y_pred) -> dict:
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    for true_c, pred_c in zip(y, y_pred):
        confusion[class_to_idx[true_c], class_to_idx[pred_c]] += 1

    precisions, recalls, f1_scores, class_sizes = [], [], [], []
    for c in classes:
        tp = np.sum((y == c) & (y_pred == c))
        fp = np.sum((y != c) & (y_pred == c))
        fn = np.sum((y == c) & (y_pred != c))
        precisions.append(tp / (tp + fp) if tp + fp else 0.0)
        recalls.append(tp / (tp + fn) if tp + fn else 0.0)
        f1_scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        class_sizes.append(np.sum(y == c))

    f1_scores = np.array(f1_scores)
    class_sizes = np.array(class_sizes)
    return {
        "accuracy": float(np.mean(y == y_pred)),
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "weighted_f1": float(np.sum(f1_scores * class_sizes) / np.sum(class_sizes)),
        "confusion_matrix": confusion,
        "classes": classes,
    }


def print_metrics(metrics: dict, name: str = "Model") -> None:
    print(f"{name} - test set")
    print(f"  Accuracy     {metrics['accuracy']:.4f}")
    print(f"  Precision    {metrics['precision']:.4f}  (macro)")
    print(f"  Recall       {metrics['recall']:.4f}  (macro)")
    print(f"  Macro F1     {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1  {metrics['weighted_f1']:.4f}")
    if "roc_auc" in metrics:
        print(f"  ROC AUC      {metrics['roc_auc']:.4f}")
    print("  Confusion matrix (rows = true, cols = predicted):")
    for row in metrics["confusion_matrix"]:
        print("    " + "  ".join(f"{v:>6}" for v in row))
