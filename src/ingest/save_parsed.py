from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.ingest.parse_replays import parse_battle_log

# processes raw logs, builds master CSV, and splits into train/val/test sets
def build_and_split(
    raw_log_dir: str | Path,
    out_dir: str | Path,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    random_seed: int = 42,
):
    raw_dir = Path(raw_log_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # parse all logs
    all_records = []
    for log_file in raw_dir.glob("*.log"):
        try:
            recs = parse_battle_log(log_file)
            rid = log_file.stem
            for r in recs:
                r["replay_id"] = rid
                # sets replay id field for each record
            all_records.extend(recs)
        except Exception as e:
            print(f"[WARN] failed to parse {log_file.name}: {e}")

    # builds master dataframe
    master_df = pd.json_normalize(all_records, sep="_")
    master_df = master_df[master_df["turn"] > 0]
    master_csv = out_dir / "master.csv"
    master_df.to_csv(master_csv, index=False)
    print(f"Saved master.csv ({len(master_df)} rows) → {master_csv}")

    # split replay_ids into train/val/test
    replays = master_df["replay_id"].unique()
    # keeps replay IDs unique
    train_reps, temp_reps = train_test_split(
        replays,
        train_size=train_frac,
        random_state=random_seed,
    )
    # allocate val vs test from the remainder
    val_reps, test_reps = train_test_split(
        temp_reps,
        test_size=test_frac / (val_frac + test_frac),
        random_state=random_seed,
    )

    # subset and save splits
    splits = {
        "train.csv": train_reps,
        "val.csv":   val_reps,
        "test.csv":  test_reps,
    }
    for fname, rep_ids in splits.items():
        df_split = master_df[master_df["replay_id"].isin(rep_ids)]
        path = out_dir / fname
        df_split.to_csv(path, index=False)
        print(f"Saved {fname} ({len(df_split)} rows, {len(rep_ids)} replays)")

if __name__ == "__main__":
    build_and_split(
        raw_log_dir = "data/raw/gen9randombattle_logs",
        out_dir = "data/parsed",
        train_frac = 0.8,
        val_frac = 0.1,
        test_frac = 0.1,
        random_seed = 42,
    )