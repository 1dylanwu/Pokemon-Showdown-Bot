import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from src.utils.utils import split_action_type

def extract(label: str) -> str:
    if label.startswith("forced_switch_"):
        label = label.replace("forced_switch_", "")
    elif label.startswith("switch_"):
        label = label.replace("switch_", "")
    elif label.startswith("move_"):
        label = label.replace("move_", "")
    return label

def save_data(X_train, y_train, X_val, y_val, X_test, y_test, action_type, le = None, out = "data/processed/"):
    y_tr_type = split_action_type(y_train)
    y_va_type = split_action_type(y_val)
    y_te_type = split_action_type(y_test)

    if action_type == "move":
        idx_tr = np.where([act.startswith("move_") for act in y_tr_type])[0]
        idx_va = np.where([act.startswith("move_") for act in y_va_type])[0]
        idx_te = np.where([act.startswith("move_") for act in y_te_type])[0]
    else:
        idx_tr = np.where(y_tr_type == action_type)[0]
        idx_va = np.where(y_va_type == action_type)[0]
        idx_te = np.where(y_te_type == action_type)[0]
    
    X_tr_new = X_train[idx_tr]
    y_tr_new = y_train[idx_tr]

    X_va_new = X_val[idx_va]
    y_va_new = y_val[idx_va]

    X_te_new = X_test[idx_te]
    y_te_new = y_test[idx_te]

    y_tr_new = np.array([extract(l) for l in y_tr_new])
    y_va_new = np.array([extract(l) for l in y_va_new])
    y_te_new = np.array([extract(l) for l in y_te_new])

    if not le:
        le = LabelEncoder()
        all_labels = np.unique(np.concatenate([
            y_tr_new, y_va_new, y_te_new
            #,["zoroark", "zoroarkhisui"]
        ]))
        le.fit(all_labels)
        joblib.dump(le, f"models/{action_type}/util/label_encoder.pkl")

    y_te_enc = le.transform(y_te_new)
    y_tr_enc = le.transform(y_tr_new)
    y_va_enc = le.transform(y_va_new)

    np.save(out + action_type + "/X_tr.npy", X_tr_new)
    np.save(out + action_type + "/y_tr.npy", y_tr_enc)
    np.save(out + action_type + "/X_va.npy", X_va_new)
    np.save(out + action_type + "/y_va.npy", y_va_enc)
    np.save(out + action_type + "/X_te.npy", X_te_new)
    np.save(out + action_type + "/y_te.npy", y_te_enc)

if __name__ == "__main__":
    pre = "data/processed/general/"
    X_train, y_train = np.load(pre + "X_train.npy", ).astype(np.float32), np.load(pre + "y_train.npy", allow_pickle=True)
    X_val, y_val = np.load(pre+"X_val.npy").astype(np.float32), np.load(pre+"y_val.npy", allow_pickle=True)
    X_test, y_test = np.load(pre+"X_test.npy").astype(np.float32), np.load(pre+"y_test.npy", allow_pickle=True)

    #save_data(X_train, y_train, X_val, y_val, X_test, y_test, "move", joblib.load("models/move/util/label_encoder.pkl"))
    #save_data(X_train, y_train, X_val, y_val, X_test, y_test, "forced", joblib.load("models/forced/util/label_encoder.pkl"))
    #save_data(X_train, y_train, X_val, y_val, X_test, y_test, "switch", joblib.load("models/switch/util/label_encoder.pkl"))
    save_data(X_train, y_train, X_val, y_val, X_test, y_test, "move_damage", joblib.load("models/move/util/label_encoder.pkl"))
    save_data(X_train, y_train, X_val, y_val, X_test, y_test, "move_utility", joblib.load("models/move/util/label_encoder.pkl"))