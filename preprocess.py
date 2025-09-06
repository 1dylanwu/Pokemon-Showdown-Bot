import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    MultiLabelBinarizer,
)


def load_and_clean(csv_path: Path) -> pd.DataFrame:

    df = pd.read_csv(csv_path, dtype=str)

    # drop the redudant turn column
    df.drop(columns=["turn"], errors="ignore", inplace=True) 

    # strip leading "state_" from nested-field columns
    df.rename(columns=lambda c: c[6:] if c.startswith("state_") else c,
              inplace=True)

    # restore team‐species lists (they were saved as strings)
    for col in ("p1_team_species", "p2_team_species"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: eval(x) if isinstance(x, str) else x
            )

    # coerce player‐HP% and fainted counts to numeric
    for num_col in (
        "turn",
        "p1a_hp_pct",
        "p2a_hp_pct",
        "p1a_fainted",
        "p2a_fainted",
    ):
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    return df


def flatten_sets(
    df: pd.DataFrame,
    col: str,
    prefix: str,
    mlb: MultiLabelBinarizer = None
) -> tuple[pd.DataFrame, MultiLabelBinarizer]:
    # one-hot encode a column of sets/lists
    # takes in mlb to use for transform (None to fit a new one)
    seqs = df[col].apply(lambda x: list(x) if isinstance(x, (set, list)) else [])
    
    if mlb is None:
        mlb = MultiLabelBinarizer(sparse_output=False)
        matrix = mlb.fit_transform(seqs)
    else:
        matrix = mlb.transform(seqs)
    
    cols = [f"{prefix}{s}" for s in mlb.classes_]
    onehot = pd.DataFrame(matrix, columns=cols, index=df.index)
    
    return onehot, mlb



def build_feature_matrix(
    df: pd.DataFrame,
    mlb1: MultiLabelBinarizer = None,
    mlb2: MultiLabelBinarizer = None
):
    # from the cleaned dataframe, get x(feature matrix) and y(targets)
    # multi-hot encode team species
    p1_ts, mlb1 = flatten_sets(df, "p1_team_species", "p1_team_", mlb1)
    p2_ts, mlb2 = flatten_sets(df, "p2_team_species", "p2_team_", mlb2)


    # for the known HP and status columns (already flattened)
    hp_cols = [c for c in df.columns if c.startswith("p1_known_hp_") 
                                   or c.startswith("p2_known_hp_")]
    if hp_cols:
        df[hp_cols] = df[hp_cols].apply(pd.to_numeric, errors="coerce")

    status_cols = [c for c in df.columns if c.startswith("p1_known_status_")
                                       or c.startswith("p2_known_status_")]

    # raw numeric and categorical columns
    raw_nums = ["turn", "p1a_hp_pct", "p2a_hp_pct", "p1a_fainted", "p2a_fainted"]
    num_cols = [c for c in raw_nums + hp_cols if c in df.columns]

    raw_cats = ["p1a_active", "p2a_active", "p1a_status", "p2a_status", "weather", "terrain"]
    cat_cols = [c for c in raw_cats + status_cols if c in df.columns]

    # assemble feature matrix!!!
    X = pd.concat(
        [
            df[num_cols].astype(float),
            df[cat_cols].astype(str),
            p1_ts,
            p2_ts,
        ],
        axis=1,
    )

    # target
    y = df["action"] if "action" in df.columns else None

    return X, y, mlb1, mlb2


def preprocess(
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    out_dir: Path,
    seed: int = 42,
):

    out_dir.mkdir(parents=True, exist_ok=True)

    # TRAIN
    df_train = load_and_clean(train_csv)
    X_train, y_train, mlb1, mlb2 = build_feature_matrix(df_train)

    # identify numeric vs categorical for ColumnTransformer
    num_cols = X_train.select_dtypes("number").columns.tolist()
    cat_cols = X_train.select_dtypes("object").columns.tolist()

    ct = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    pipeline = Pipeline([("trans", ct)])
    X_train_proc = pipeline.fit_transform(X_train)

    np.save(out_dir / "X_train.npy", X_train_proc)
    if y_train is not None:
        np.save(out_dir / "y_train.npy", y_train.to_numpy())
    joblib.dump(pipeline, out_dir / "pipeline.pkl")

    print(f"[train] {len(df_train)} rows → {X_train_proc.shape[1]} features saved")

    # VAL
    df_val = load_and_clean(val_csv)
    X_val, y_val, *_ = build_feature_matrix(df_val, mlb1=mlb1, mlb2=mlb2)
    X_val_proc = pipeline.transform(X_val)

    np.save(out_dir / "X_val.npy", X_val_proc)
    if y_val is not None:
        np.save(out_dir / "y_val.npy", y_val.to_numpy())

    print(f"[val]   {len(df_val)} rows → {X_val_proc.shape[1]} features saved")

    # TEST
    df_test = load_and_clean(test_csv)
    X_test, y_test, *_ = build_feature_matrix(df_val, mlb1=mlb1, mlb2=mlb2)
    X_test_proc = pipeline.transform(X_test)

    np.save(out_dir / "X_test.npy", X_test_proc)
    if y_test is not None:
        np.save(out_dir / "y_test.npy", y_test.to_numpy())

    # total number of features is 7210. wowzers!
    print(f"[test]  {len(df_test)} rows → {X_test_proc.shape[1]} features saved")


if __name__ == "__main__":
    TRAIN = Path("data/logs/parsed/train.csv")
    VAL   = Path("data/logs/parsed/val.csv")
    TEST  = Path("data/logs/parsed/test.csv")
    OUT   = Path("data/processed")

    preprocess(TRAIN, VAL, TEST, OUT, seed=42)