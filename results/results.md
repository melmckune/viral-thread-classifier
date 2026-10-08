| Model | From scratch | Accuracy | Macro F1 | Weighted F1 | ROC AUC | Best params |
|---|:---:|---:|---:|---:|---:|---|
| Random Forest | sklearn | 0.703 | 0.692 | 0.703 | 0.761 | n_estimators=25, max_depth=15 |
| k-NN | yes | 0.680 | 0.665 | 0.679 | 0.741 | k=50 |
| Decision Tree | yes | 0.648 | 0.642 | 0.651 | 0.709 | max_depth=5, min_split=2 |
| Linear SVM | yes | 0.644 | 0.629 | 0.643 | 0.712 | lr=0.01, lambda=0.01, epochs=1000 |
| Logistic Regression | yes | 0.639 | 0.624 | 0.639 | 0.712 | lambda=0, reg=l2 |
| Naive Bayes | yes | 0.624 | 0.623 | 0.626 | 0.704 | - |
