import json
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils.utils import normalize
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from src.utils.utils import canonicalize_player_df

def load_stats(p: Path) -> pd.DataFrame:
    out = {}
    with p.open("r", encoding="utf-8") as f:
        doc = json.load(f)
        for name, stats in doc.items():
            out[name] = {
                "hp": float(stats.get("hp", 0.0)),
                "atk": float(stats.get("atk", 0.0)),
                "def": float(stats.get("def", 0.0)),
                "spa": float(stats.get("spa", 0.0)),
                "spd": float(stats.get("spd", 0.0)),
                "spe": float(stats.get("spe", 0.0)),
            }
    return out

def boost_stage_to_multiplier(stage: float) -> float:
    if stage >= 0:
        return (2.0 + stage) / 2.0
    else:
        return 2.0 / (2.0 - stage)
    
def compute_effective_from_base(row, stats_table: Dict[str, Dict[str, float]]):
    """
    Given a pd.Series row and a stats_table (normalized_species -> stats),
    return a dict of boost-aware numeric features for p1 and p2.
    """

    def calc_side(side: str) -> Dict[str, float]:
        raw = row[f"{side}_active"]

        sp = normalize(raw)
        base = stats_table.get(sp)
        if base is None:
            raise KeyError(f"Species {sp} not found")

        # read boost stages
        atk_s = row.get(f"{side}_boost_atk", 0.0)
        def_s = row.get(f"{side}_boost_def", 0.0)
        spa_s = row.get(f"{side}_boost_spa", 0.0)
        spd_s = row.get(f"{side}_boost_spd", 0.0)
        spe_s = row.get(f"{side}_boost_spe", 0.0)

        # multipliers
        m_atk = boost_stage_to_multiplier(atk_s)
        m_def = boost_stage_to_multiplier(def_s)
        m_spa = boost_stage_to_multiplier(spa_s)
        m_spd = boost_stage_to_multiplier(spd_s)
        m_spe = boost_stage_to_multiplier(spe_s)

        # effective stats
        eff_atk = base["atk"] * m_atk
        eff_def = base["def"] * m_def
        eff_spa = base["spa"] * m_spa
        eff_spd = base["spd"] * m_spd
        eff_spe = base["spe"] * m_spe
        hp_abs = float(base["hp"])

        return {
            f"{side}_eff_atk": eff_atk,
            f"{side}_eff_def": eff_def,
            f"{side}_eff_spa": eff_spa,
            f"{side}_eff_spd": eff_spd,
            f"{side}_eff_spe": eff_spe,
            f"{side}_eff_hp": hp_abs,
            f"{side}_boost_total_pos": sum(max(0.0, s) for s in (atk_s, spa_s, spe_s)),
            f"{side}_boost_total_neg": sum(min(0.0, s) for s in (def_s, spd_s)),
        }

    left = calc_side("p1")
    right = calc_side("p2")
    out = {}
    out.update(left)
    out.update(right)

    out["p1_outspeed"] = int(out["p1_eff_spe"] > out["p2_eff_spe"])
    out["p2_outspeed"] = int(out["p2_eff_spe"] > out["p1_eff_spe"])
    return out


def load_and_clean(csv_path: Path) -> pd.DataFrame:

    df = pd.read_csv(csv_path, dtype=str)
    df["action_full"] = df["action_type"].str.lower() + "_" + df["action"].apply(normalize)
        
    # restore team‐species and typing lists from strings
    for col in ("p1_team_species", "p2_team_species", "p1_types", "p2_types"):
        df[col] = df[col].apply(
            lambda x: [normalize(s) for s in eval(x)] if isinstance(x, str) else x
        )

    for col in ("p1_active", "p2_active"):
        df[col] = df[col].apply(normalize)

    # terastallization status to 0/1
    for col in ("p1_is_terastallized", "p2_is_terastallized"):
        # map string → 0/1
        df[col] = df[col].map({"True": 1, "False": 0}).fillna(0).astype(int)

    # type matchup statistics
    for col in ("p1_type_matchup", "p2_type_matchup"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(1.5)

    # coerce player‐HP% and fainted counts to numeric
    for num_col in (
        "turn",
        "p1_hp_pct",
        "p2_hp_pct",
        "p1_fainted",
        "p2_fainted",
    ):
        df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    for col in ("hazards_p1", "hazards_p2"):
        df[col] = df[col].apply(
            lambda x: eval(x) if isinstance(x, str) else {}
        )

    # flags for low hp (<30%)
    hp_threshold = 0.3
    df["p1_low_hp"] = (pd.to_numeric(df["p1_hp_pct"], errors="coerce") < hp_threshold).fillna(False).astype(int)
    df["p2_low_hp"] = (pd.to_numeric(df["p2_hp_pct"], errors="coerce") < hp_threshold).fillna(False).astype(int)

    df.rename(columns=lambda c: (
        c[:14] + normalize(c[14:]) if c.startswith("p1_known_hp_") else
        c[:14] + normalize(c[14:]) if c.startswith("p2_known_hp_") else c
    ), inplace=True)

    boost_cols = [c for c in df.columns if c.startswith("p1_boost_") or c.startswith("p2_boost_")]
    known_hp_cols = [c for c in df.columns if c.startswith("p1_known_hp_") or c.startswith("p2_known_hp_")]
    # fill with 0s
    for c in boost_cols + known_hp_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[boost_cols] = df[boost_cols].fillna(0.0)
    df[known_hp_cols] = df[known_hp_cols].fillna(0.0)
    
    # Fill categoricals with explicit tokens so OHE can encode them
    cat_fill = {}
    cat_fill["p1_status"] = "none"
    cat_fill["p2_status"] = "none"
    cat_fill["weather"] = "clear"
    cat_fill["terrain"] = "none"
    cat_fill["p1_tera_type"] = "none"
    cat_fill["p2_tera_type"] = "none"
    df.fillna(cat_fill, inplace=True)
    df = df.copy()
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

# entry hazards
def collect_hazards(df, col):
    all_keys = set()
    for d in df[col].dropna():
        if isinstance(d, dict):
            all_keys |= set(d.keys())
        elif isinstance(d, list):
            # if it's a list of dicts, flatten it
            for item in d:
                if isinstance(item, dict):
                    all_keys |= set(item.keys())
        # else: skip anything that's not dict or list
    return sorted(all_keys)

def flatten_haz(df, col, prefix, keys):
    data = {}
    for k in keys:
        name = f"{prefix}{k}"
        data[name] = df[col].apply(lambda d: float(d.get(k, 0)) if isinstance(d, dict) else 0.0)
    return pd.DataFrame(data, index=df.index)


def build_feature_matrix(
    df: pd.DataFrame,
    mlb1: MultiLabelBinarizer = None,
    mlb2: MultiLabelBinarizer = None,
    mlb_types: MultiLabelBinarizer = None
) -> Tuple[pd.DataFrame, pd.Series, MultiLabelBinarizer, MultiLabelBinarizer, MultiLabelBinarizer, List[str], List[str]]:
    # from the cleaned dataframe, get x(feature matrix) and y(targets)
    # multi-hot encode team species
    p1_ts, mlb1 = flatten_sets(df, "p1_team_species", "p1_team_", mlb1)
    p2_ts, mlb2 = flatten_sets(df, "p2_team_species", "p2_team_", mlb2)
    
    #idk why p1 and p2 are at the end :(
    p1_haz_keys = collect_hazards(df, "hazards_p1")
    p2_haz_keys = collect_hazards(df, "hazards_p2")

    p1_haz = flatten_haz(df, "hazards_p1", "p1_haz_", p1_haz_keys)
    p2_haz = flatten_haz(df, "hazards_p2", "p2_haz_", p2_haz_keys)

    stats_table = load_stats(Path("data/raw/computed_stats.json"))
    eff_df = df.apply(lambda row: pd.Series(compute_effective_from_base(row, stats_table)), axis=1)
    eff_df = eff_df.fillna(0.0).astype(float)

    # one hot encoded categories for active pokemon types
    p1_types_ohe, mlb_types = flatten_sets(
        df, 'p1_types', 'p1_type_', mlb_types
    )
    p2_types_ohe, _ = flatten_sets(
        df, 'p2_types', 'p2_type_', mlb_types
    )

    hp_cols = [c for c in df.columns if c.startswith("p1_known_hp_") or c.startswith("p2_known_hp_")] 
    boost_cols = [c for c in df.columns if c.startswith("p1_boost_") or c.startswith("p2_boost_")]

    # raw numeric and categorical columns
    # i removed status for inactive pokemon due to too many features
    raw_nums = ["turn", "p1_hp_pct", "p2_hp_pct", "p1_fainted", "p2_fainted", "p1_is_terastallized", "p2_is_terastallized", "p1_type_matchup", "p2_type_matchup", "p1_low_hp", "p2_low_hp"]
    num_cols = [c for c in raw_nums + hp_cols + boost_cols if c in df.columns]

    cat_cols = ["side", "p1_active", "p2_active", "p1_status", "p2_status", "weather", "terrain", "p1_tera_type", "p2_tera_type"]

    # assemble feature matrix!!!
    X = pd.concat(
        [
            df[num_cols].astype(float),
            df[cat_cols].astype(str),
            p1_ts,
            p2_ts,
            p1_haz,
            p2_haz,
            p1_types_ohe,
            p2_types_ohe,
            eff_df
        ],
        axis=1,
    )

    # target
    y = df["action_full"]
    return X, y, mlb1, mlb2, mlb_types, p1_haz_keys, p2_haz_keys


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
    canonicalize_player_df(df_train, player_col = "side")
    X_train, y_train, mlb1, mlb2, mlb_types, hz1_keys, hz2_keys = build_feature_matrix(df_train)

    
    joblib.dump(mlb1, out_dir / "mlb_p1_team.pkl")
    joblib.dump(mlb2, out_dir / "mlb_p2_team.pkl")
    joblib.dump(mlb_types, out_dir / "mlb_types.pkl")
    joblib.dump(hz1_keys, out_dir / "hazard_keys_p1.pkl")
    joblib.dump(hz2_keys, out_dir / "hazard_keys_p2.pkl")
    
    # identify numeric vs categorical for ColumnTransformer
    num_cols = X_train.select_dtypes("number").columns.tolist()
    cat_cols = X_train.select_dtypes("object").columns.tolist()

    # removed scaling for hp columns
    hp_cols = [
        c for c in num_cols
        if c in ("p1_hp_pct", "p2_hp_pct")
        or c.startswith("p1_known_hp_")
        or c.startswith("p2_known_hp_")
    ]

    bin_prefixes = ("p1_team_", "p2_team_", "p1_type_", "p2_type_", "p1_haz_", "p2_haz_", "p1_is_terastallized", "p2_is_terastallized")
    bin_cols = [c for c in num_cols if c.startswith(bin_prefixes)]

    other_num_cols = [c for c in num_cols if c not in hp_cols + bin_cols]

    ct = ColumnTransformer(
        [
            ("num", StandardScaler(), other_num_cols),
            ("hp",  "passthrough",  hp_cols),
            ("bin", "passthrough", bin_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    pipeline = Pipeline([("trans", ct)])

    X_train_proc = pipeline.fit_transform(X_train)

    np.save(out_dir / "X_train.npy", X_train_proc)
    np.save(out_dir / "y_train.npy", y_train.to_numpy())
    joblib.dump(pipeline, out_dir / "pipeline.pkl")

    print(f"[train] {len(df_train)} rows → {X_train_proc.shape[1]} features saved")

    # VAL
    df_val = load_and_clean(val_csv)
    canonicalize_player_df(df_val, player_col = "side")
    X_val, y_val, *_ = build_feature_matrix(df_val, mlb1=mlb1, mlb2=mlb2)

    X_val_proc = pipeline.transform(X_val)

    np.save(out_dir / "X_val.npy", X_val_proc)
    np.save(out_dir / "y_val.npy", y_val.to_numpy())

    print(f"[val]   {len(df_val)} rows → {X_val_proc.shape[1]} features saved")

    # TEST
    df_test = load_and_clean(test_csv)
    canonicalize_player_df(df_test, player_col = "side")
    X_test, y_test, *_ = build_feature_matrix(df_test, mlb1=mlb1, mlb2=mlb2)

    X_test_proc = pipeline.transform(X_test)

    np.save(out_dir / "X_test.npy", X_test_proc)
    np.save(out_dir / "y_test.npy", y_test.to_numpy())

    # total number of features is 3228
    print(f"[test]  {len(df_test)} rows → {X_test_proc.shape[1]} features saved")


if __name__ == "__main__":
    TRAIN = Path("data/parsed/train.csv")
    VAL = Path("data/parsed/val.csv")
    TEST = Path("data/parsed/test.csv")
    OUT = Path("data/processed/general")

    preprocess(TRAIN, VAL, TEST, OUT, seed=42)