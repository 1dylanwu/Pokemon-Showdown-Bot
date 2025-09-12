import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
from src.utils.utils import split_action_type

def save_switch_data(type: str, out: str, le: LabelEncoder):
    pre = "data/processed/general/"
    X_train, y_train = np.load(pre + "X_train.npy").astype(np.float32), np.load(pre + "y_train.npy", allow_pickle=True)
    X_val, y_val = np.load(pre+"X_val.npy").astype(np.float32), np.load(pre+"y_val.npy", allow_pickle=True)
    X_test, y_test = np.load(pre+"X_test.npy").astype(np.float32), np.load(pre+"y_test.npy", allow_pickle=True)

    y_tr_type = split_action_type(y_train)
    y_va_type = split_action_type(y_val)
    y_te_type = split_action_type(y_test)

    switch_idx_tr = np.where(y_tr_type == type)[0]
    switch_idx_va = np.where(y_va_type == type)[0]
    switch_idx_te = np.where(y_te_type == type)[0]
    
    X_tr_sw = X_train[switch_idx_tr]
    y_tr_sw = y_train[switch_idx_tr]

    X_va_sw = X_val[switch_idx_va]
    y_va_sw = y_val[switch_idx_va]

    X_te_sw = X_test[switch_idx_te]
    y_te_sw = y_test[switch_idx_te]

    if type == "forced":
        def extract_species(label: str) -> str:
            if label.startswith("forced_switch_"):
                label = label.replace("forced_switch_", "")
            elif label.startswith("switch_"):
                label = label.replace("switch_", "")
            return label

        y_tr_sw = np.array([extract_species(l) for l in y_tr_sw])
        y_va_sw = np.array([extract_species(l) for l in y_va_sw])
        y_te_sw = np.array([extract_species(l) for l in y_te_sw])

    y_te_sw_enc = le.transform(y_te_sw)
    y_tr_sw_enc = le.transform(y_tr_sw)
    y_va_sw_enc = le.transform(y_va_sw)

    np.save(out + "X_tr_sw.npy", X_tr_sw)
    np.save(out + "y_tr_sw.npy", y_tr_sw_enc)
    np.save(out + "X_va_sw.npy", X_va_sw)
    np.save(out + "y_va_sw.npy", y_va_sw_enc)
    np.save(out + "X_te_sw.npy", X_te_sw)
    np.save(out + "y_te_sw.npy", y_te_sw_enc)


save_switch_data("forced", "data/processed/forced/", joblib.load("models/stage2_switch/util/label_encoder.pkl"))