"""Train, tune and evaluate every model end to end.

    python -m virality.train                 # full run (all models, full grids)
    python -m virality.train --quick         # smaller grids, a few minutes
    python -m virality.train --models nb logreg rf

Hyperparameters are always selected on the validation split; the test split
is touched exactly once per model, for the final numbers.
"""

import argparse
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from . import plotting
from .data import DEFAULT_DATA_PATH, load_dataset, make_splits
from .metrics import evaluate_model, print_metrics
from .models import KNN, DecisionTree, GaussianNaiveBayes, LinearSVM, LogisticRegression

ROOT = Path(__file__).resolve().parent.parent


def accuracy(model, X, y):
    return float(np.mean(model.predict(X) == y))


def tune_naive_bayes(d, quick):
    return GaussianNaiveBayes().fit(d.X_train, d.y_train), {}


def tune_logistic_regression(d, quick):
    best = (-1, None, None)
    for lam in ([0, 10] if quick else [0, 0.1, 1, 10]):
        model = LogisticRegression(lr=0.1, epochs=1000, lambda_val=lam, reg_type="l2")
        model.fit(d.X_train, d.y_train, d.X_val, d.y_val)
        acc = accuracy(model, d.X_val, d.y_val)
        print(f"    lambda={lam:<5} val_acc={acc:.4f}  ({len(model.losses)} epochs)")
        if acc > best[0]:
            best = (acc, model, {"lambda": lam, "reg": "l2"})
    return best[1], best[2]


def tune_knn(d, quick):
    ks = [5, 30] if quick else [1, 5, 10, 15, 30, 50]
    model = KNN(k=max(ks)).fit(d.X_train, d.y_train)
    # Find neighbours once at the largest k, then evaluate each smaller k for free
    val_labels = model.neighbor_labels(d.X_val)
    best = (-1, None)
    for k in ks:
        model.k = k
        acc = float(np.mean(model.predict(d.X_val, val_labels) == d.y_val))
        print(f"    k={k:<3} val_acc={acc:.4f}")
        if acc > best[0]:
            best = (acc, k)
    model.k = best[1]
    return model, {"k": best[1]}


def tune_decision_tree(d, quick):
    grid = product([5, 10] if quick else [3, 5, 10, 15], [20] if quick else [2, 5, 10, 20])
    best = (-1, None, None)
    for depth, split in grid:
        model = DecisionTree(max_depth=depth, min_split=split).fit(d.X_train, d.y_train)
        acc = accuracy(model, d.X_val, d.y_val)
        print(f"    max_depth={depth:<3} min_split={split:<3} val_acc={acc:.4f}")
        if acc > best[0]:
            best = (acc, model, {"max_depth": depth, "min_split": split})
    return best[1], best[2]


def tune_svm(d, quick):
    lrs = [1e-2] if quick else [1e-2, 1e-3]
    lams = [1e-2, 1e-3] if quick else [1e-2, 1e-3, 1e-4]
    epochs = 100 if quick else 1000
    best = (-1, None, None)
    for lr, lam in product(lrs, lams):
        model = LinearSVM(lr=lr, lambda_reg=lam, epochs=epochs, batch_size=256)
        model.fit(d.X_train, d.y_train, d.X_val, d.y_val)
        acc = accuracy(model, d.X_val, d.y_val)
        print(f"    lr={lr:<6} lambda={lam:<6} val_acc={acc:.4f}")
        if acc > best[0]:
            best = (acc, model, {"lr": lr, "lambda": lam, "epochs": epochs})
    return best[1], best[2]


def tune_random_forest(d, quick):
    grid = product([50] if quick else [25, 50, 100], [10, 15] if quick else [5, 10, 15])
    best = (-1, None, None)
    for n_est, depth in grid:
        model = RandomForestClassifier(n_estimators=n_est, max_depth=depth,
                                       random_state=42, n_jobs=-1)
        model.fit(d.X_train, d.y_train)
        acc = accuracy(model, d.X_val, d.y_val)
        print(f"    n_estimators={n_est:<4} max_depth={depth:<3} val_acc={acc:.4f}")
        if acc > best[0]:
            best = (acc, model, {"n_estimators": n_est, "max_depth": depth})
    return best[1], best[2]


# key: (display name, tuning function, implemented from scratch?)
MODELS = {
    "nb": ("Naive Bayes", tune_naive_bayes, True),
    "logreg": ("Logistic Regression", tune_logistic_regression, True),
    "knn": ("k-NN", tune_knn, True),
    "tree": ("Decision Tree", tune_decision_tree, True),
    "svm": ("Linear SVM", tune_svm, True),
    "rf": ("Random Forest", tune_random_forest, False),
}


def positive_scores(model, X):
    if hasattr(model, "decision_function") and not hasattr(model, "predict_proba"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def write_results_table(results, path):
    lines = [
        "| Model | From scratch | Accuracy | Macro F1 | Weighted F1 | ROC AUC | Best params |",
        "|---|:---:|---:|---:|---:|---:|---|",
    ]
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["metrics"]["macro_f1"]):
        m = r["metrics"]
        params = ", ".join(f"{k}={v}" for k, v in r["params"].items()) or "-"
        lines.append(
            f"| {name} | {'yes' if r['scratch'] else 'sklearn'} | {m['accuracy']:.3f} | "
            f"{m['macro_f1']:.3f} | {m['weighted_f1']:.3f} | {m['roc_auc']:.3f} | {params} |"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="path to viral-db.csv")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--quick", action="store_true", help="smaller grids / fewer epochs")
    parser.add_argument("--out", default=ROOT / "results", help="output directory")
    parser.add_argument("--figures", default=ROOT / "assets" / "figures",
                        help="where to write PNG figures")
    args = parser.parse_args(argv)

    out, fig_dir = Path(args.out), Path(args.figures)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    d = make_splits(load_dataset(args.data))
    print(d.summary())

    results = {}
    for key in args.models:
        name, tune, scratch = MODELS[key]
        print(f"\n== {name} ==")
        start = time.perf_counter()
        model, params = tune(d, args.quick)
        y_pred = model.predict(d.X_test)
        scores = positive_scores(model, d.X_test)
        metrics = evaluate_model(d.y_test, y_pred)
        metrics["roc_auc"] = float(roc_auc_score(d.y_test, scores))
        elapsed = time.perf_counter() - start
        print(f"  best params: {params or '-'}   ({elapsed:.1f}s)")
        print_metrics(metrics, name)
        results[name] = {"model": model, "params": params, "metrics": metrics,
                         "scores": scores, "scratch": scratch, "seconds": elapsed}

    # Persist numbers
    serialisable = {
        name: {
            "params": r["params"],
            "from_scratch": r["scratch"],
            "train_seconds": round(r["seconds"], 1),
            **{k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k, v in r["metrics"].items()},
        }
        for name, r in results.items()
    }
    (out / "metrics.json").write_text(json.dumps(serialisable, indent=2, default=int))
    write_results_table(results, out / "results.md")

    # Figures
    plotting.plot_roc_curves(results, d.y_test, fig_dir / "roc_curves.png")
    plotting.plot_pr_curves(results, d.y_test, fig_dir / "pr_curves.png")
    plotting.plot_model_comparison(results, fig_dir / "model_comparison.png")
    plotting.plot_confusion_matrices(results, fig_dir / "confusion_matrices.png")

    histories = {}
    if "Logistic Regression" in results:
        m = results["Logistic Regression"]["model"]
        histories["Logistic Regression"] = (m.losses, m.val_losses, m.accuracies, m.val_accuracies)
    if "Linear SVM" in results:
        m = results["Linear SVM"]["model"]
        histories["Linear SVM"] = (m.train_losses, m.val_losses, m.train_accs, m.val_accs)
    if histories:
        plotting.plot_training_curves(histories, fig_dir / "training_curves.png")
    if "Random Forest" in results:
        rf = results["Random Forest"]["model"]
        plotting.plot_feature_importance(rf.feature_importances_, d.feature_names,
                                         fig_dir / "feature_importance.png")

    print(f"\nWrote {out / 'results.md'}, {out / 'metrics.json'} and figures in {fig_dir}/")
    print("\n" + (out / "results.md").read_text())


if __name__ == "__main__":
    main()
