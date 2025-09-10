from pathlib import Path
import pandas as pd

log_dir = Path("data/raw/gen9randombattle_logs")
master_df = pd.read_csv("data/parsed/master.csv")

all_logs = set(f.stem for f in log_dir.glob("*.log"))
parsed_replays = set(master_df["replay_id"].unique())

missing = sorted(all_logs - parsed_replays)
print(f"Missing replays ({len(missing)}):", missing)
