import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, classification_report, top_k_accuracy_score
import pandas as pd

def test(model, X, y, threshold, binary = False):
    if binary:
        y_val_prob = model.predict_proba(X)[:, 1]
        y_val_pred = (y_val_prob > threshold).astype(int)

        acc = accuracy_score(y, y_val_pred)
        auc = roc_auc_score(y, y_val_prob)
        print(f"Validation Accuracy: {acc:.4f}")
        print(f"Validation ROC AUC : {auc:.4f}")
        num_move_preds = np.sum(y_val_pred)
        total_preds = len(y_val_pred)
        move_ratio = num_move_preds / total_preds
        print(f"switch ratio: {1 - move_ratio}")
        print(classification_report(y, y_val_pred, target_names=["switch", "move"]))
    else:
        y_val_pred = model.predict(X)
        y_val_prob = model.predict_proba(X)
        labels = np.arange(y_val_prob.shape[1])

        acc = accuracy_score(y, y_val_pred)
        print(f"Top-1 Accuracy: {acc:.4f}")
        top3 = top_k_accuracy_score(y, y_val_prob, k=3, labels = labels)
        print(f"Top-3 Accuracy: {top3:.4f}")
