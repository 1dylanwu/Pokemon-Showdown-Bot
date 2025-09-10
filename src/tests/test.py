import numpy as np
import joblib
from sklearn.metrics import accuracy_score
import pandas as pd

model_path = "models/stage2_switch/final/switch_clf_1.2.pkl"
data_dir = "data/processed/switch/"

"""
model = joblib.load(model_path)

X_tr = np.load(data_dir + "X_tr_sw.npy")
y_tr = np.load(data_dir + "y_tr_sw.npy").astype(np.int32)

X_va = np.load(data_dir + "X_va_sw.npy")
y_va = np.load(data_dir + "y_va_sw.npy").astype(np.int32)

X_te = np.load(data_dir + "X_te_sw.npy")
y_te = np.load(data_dir + "y_te_sw.npy").astype(np.int32)


def evaluate(model, X, y, name):
    y_pred = model.predict(X)
    if y_pred.ndim == 2:  # probability output
        y_pred = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y, y_pred)
    print(f"{name} accuracy: {acc:.4f}")

evaluate(model, X_tr, y_tr, "Train")
evaluate(model, X_va, y_va, "Validation")
evaluate(model, X_te, y_te, "Test")
"""
pre = "data/processed/switch/"
X_test, y_test = np.load(pre+"X_te_sw.npy").astype(np.float32), np.load(pre+"y_te_sw.npy", allow_pickle=True)
print(X_test.shape)
print(y_test.shape)
df = pd.read_csv("data/parsed/test.csv")
n_forced = (df["action_type"] == "forced_switch").sum()
print(f"Forced switch rows: {n_forced}")
