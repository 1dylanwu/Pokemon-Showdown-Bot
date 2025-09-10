from collections import defaultdict
import re
import pandas as pd
from pathlib import Path
import traceback
# buils a data set for testing/inference of all available pokemon for switches (although only those that ever get revealed)
def extract_team_info(log_lines):
    slot_re = re.compile(r'^(p[12][ab]):\s*([^\|,]+)')

    revealed = {"p1": set(), "p2": set()}
    for line in log_lines:
        parts = line.strip().split("|")
        for part in parts[2:]:
            m = slot_re.match(part)
            if m:
                side, mon = m.groups()
                side = side[:2]
                revealed[side].add(mon)

    fainted = {"p1": set(), "p2": set()}
    records = []
    current_turn = None

    for line in log_lines:
        parts = line.strip().split("|")
        if len(parts) < 2:
            continue
        tag = parts[1]

        if tag == "turn":
            try:
                current_turn = int(parts[2])
            except ValueError:
                continue

            available = {
                "p1": sorted(revealed["p1"] - fainted["p1"]),
                "p2": sorted(revealed["p2"] - fainted["p2"]),
            }

            records.append({
                "turn": current_turn,
                "p1_available": available["p1"],
                "p2_available": available["p2"],
            })

        elif tag == "faint":
            for part in parts[2:]:
                m = slot_re.match(part)
                if m:
                    side, mon = m.groups()
                    side = side[:2]
                    fainted[side].add(mon)

    return records

def save_team_info_by_split(master_csv, raw_log_dir, out_dir):
    raw_dir = Path(raw_log_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load replay splits from master.csv
    train_ids = pd.read_csv("data/parsed/train.csv")["replay_id"].unique()
    val_ids   = pd.read_csv("data/parsed/val.csv")["replay_id"].unique()
    test_ids  = pd.read_csv("data/parsed/test.csv")["replay_id"].unique()

    split_map = {rid: "train" for rid in train_ids}
    split_map.update({rid: "val" for rid in val_ids})
    split_map.update({rid: "test" for rid in test_ids})


    # Prepare containers
    split_records = defaultdict(list)

    # Parse each log and assign to correct split
    for log_file in raw_dir.glob("*.log"):
        rid = log_file.stem
        split = split_map.get(rid)
        if not split:
            continue  # skip logs not in master.csv
        try:
            try:
                lines = log_file.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError as e:
                print(f"[ERROR] UnicodeDecodeError in {log_file.name}: {e}")
                traceback.print_exc()
                continue
            recs = extract_team_info(lines)
            for r in recs:
                r["replay_id"] = rid
            split_records[split].extend(recs)
        except Exception as e:
            print(f"[WARN] failed to parse {rid}: {e}")

    # Save each split
    for split_name, records in split_records.items():
        df = pd.json_normalize(records, sep="_")
        out_path = out_dir / f"{split_name}_team_info.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {split_name}_team_info.csv ({len(df)} rows) → {out_path}")

if __name__ == "__main__":
    save_team_info_by_split(
        master_csv="data/parsed/master.csv",
        raw_log_dir="data/raw/gen9randombattle_logs",
        out_dir="data/team_info",
    )