import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from train_type import split_action_type

pre = "data/processed/"
X_train, y_train = np.load(pre + "X_train.npy", ).astype(np.float32), np.load(pre + "y_train.npy", allow_pickle=True)
X_val,   y_val   = np.load(pre+"X_val.npy").astype(np.float32),   np.load(pre+"y_val.npy", allow_pickle=True)
X_test,  y_test  = np.load(pre+"X_test.npy").astype(np.float32),  np.load(pre+"y_test.npy", allow_pickle=True)

y_tr_type = split_action_type(y_train)
y_va_type = split_action_type(y_val)
y_te_type = split_action_type(y_test)

switch_idx_tr = np.where(y_tr_type == "switch")[0]
switch_idx_va = np.where(y_va_type == "switch")[0]

X_tr_sw = X_train[switch_idx_tr]
y_tr_sw = y_train[switch_idx_tr]

X_va_sw = X_val[switch_idx_va]
y_va_sw = y_val[switch_idx_va]

le_sw = LabelEncoder().fit(y_tr_sw)
y_tr_sw_enc = le_sw.transform(y_tr_sw)
y_va_sw_enc = le_sw.transform(y_va_sw)

sw_clf = RandomForestClassifier(
    class_weight="balanced_subsample", n_estimators=200, n_jobs=-1
)
sw_clf.fit(X_tr_sw, y_tr_sw_enc)

print("Stage2b sw  train acc:", sw_clf.score(X_tr_sw, y_tr_sw_enc))
print("Stage2b sw  val   acc:", sw_clf.score(X_va_sw,  y_va_sw_enc))

joblib.dump((sw_clf, le_sw), "models/switch_clf.pkl")