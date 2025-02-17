import mlflow
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_and_log_metrics(y_true, y_pred, y_prob=None, prefix=""):
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    mlflow.log_metric(f"{prefix}precision", prec)
    mlflow.log_metric(f"{prefix}recall", rec)
    mlflow.log_metric(f"{prefix}f1", f1)
    mlflow.log_metric(f"{prefix}accuracy", acc)

    print(f"{prefix}Precision: {prec:.4f}")
    print(f"{prefix}Recall:    {rec:.4f}")
    print(f"{prefix}F1 Score:  {f1:.4f}")
    print(f"{prefix}Accuracy: {acc:.4f}")

    if y_prob is not None:
        roc_auc = roc_auc_score(y_true, y_prob)
        mlflow.log_metric(f"{prefix}roc_auc", roc_auc)
        print(f"{prefix}ROC AUC:   {roc_auc:.4f}")
