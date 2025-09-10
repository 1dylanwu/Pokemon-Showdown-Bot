import numpy as np
import joblib
from sklearn.metrics import accuracy_score
import pandas as pd

model_path = "models/stage1_type/final/type_clf_2.0.pkl"
data_dir = "data/processed/general/"

model = joblib.load(model_path)

X_tr = np.load(data_dir + "X_train.npy", allow_pickle=True)
y_tr = np.load(data_dir + "y_train.npy", allow_pickle=True)

X_va = np.load(data_dir + "X_val.npy", allow_pickle=True)
y_va = np.load(data_dir + "y_val.npy", allow_pickle=True)

X_te = np.load(data_dir + "X_test.npy", allow_pickle=True)
y_te = np.load(data_dir + "y_test.npy", allow_pickle=True)


def evaluate(model, X, y, name):
    y_pred = model.predict(X)
    if y_pred.ndim == 2:  # probability output
        y_pred = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y, y_pred)
    print(f"{name} accuracy: {acc:.4f}")

evaluate(model, X_tr, y_tr, "Train")
evaluate(model, X_va, y_va, "Validation")
evaluate(model, X_te, y_te, "Test")
