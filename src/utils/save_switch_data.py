import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
from src.utils.utils import split_action_type

pre = "data/processed/general/"
X_train, y_train = np.load(pre + "X_train.npy").astype(np.float32), np.load(pre + "y_train.npy", allow_pickle=True)
X_val, y_val = np.load(pre+"X_val.npy").astype(np.float32), np.load(pre+"y_val.npy", allow_pickle=True)
X_test, y_test = np.load(pre+"X_test.npy").astype(np.float32), np.load(pre+"y_test.npy", allow_pickle=True)

y_tr_type = split_action_type(y_train)
y_va_type = split_action_type(y_val)
y_te_type = split_action_type(y_test)

switch_idx_tr = np.where(y_tr_type == "switch")[0]
switch_idx_va = np.where(y_va_type == "switch")[0]
switch_idx_te = np.where(y_te_type == "switch")[0]

X_tr_sw = X_train[switch_idx_tr]
y_tr_sw = y_train[switch_idx_tr]

X_va_sw = X_val[switch_idx_va]
y_va_sw = y_val[switch_idx_va]

X_te_sw = X_test[switch_idx_te]
y_te_sw = y_test[switch_idx_te]

all_switch_labels = np.concatenate([y_tr_sw, y_va_sw, y_te_sw])

le_sw = LabelEncoder().fit(all_switch_labels)

y_te_sw_enc = le_sw.transform(y_te_sw)
y_tr_sw_enc = le_sw.transform(y_tr_sw)
y_va_sw_enc = le_sw.transform(y_va_sw)


joblib.dump(le_sw, "models/stage2_switch/util/label_encoder.pkl")
np.save("data/processed/switch/X_tr_sw.npy", X_tr_sw)
np.save("data/processed/switch/y_tr_sw.npy", y_tr_sw_enc)
np.save("data/processed/switch/X_va_sw.npy", X_va_sw)
np.save("data/processed/switch/y_va_sw.npy", y_va_sw_enc)
np.save("data/processed/switch/X_te_sw.npy", X_te_sw)
np.save("data/processed/switch/y_te_sw.npy", y_te_sw_enc)