import pandas as pd
from pathlib import Path

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Drop the 'turn' column if it exists
    if "turn" in df.columns:
        df.drop(columns=["turn"], inplace=True)

    # Rename columns: strip 'state_' prefix and replace 'p1a'/'p2a' with 'p1'/'p2'
    def rename_col(col: str) -> str:
        if col.startswith("state_"):
            col = col[len("state_"):]
        col = col.replace("p1a", "p1").replace("p2a", "p2")
        return col

    df.rename(columns={col: rename_col(col) for col in df.columns}, inplace=True)

    # Update 'side' column values if present
    if "side" in df.columns:
        df["side"] = df["side"].astype(str).replace({
            "p1a": "p1", "p2a": "p2",
            "P1a": "p1", "P2a": "p2"
        })

    return df

def process_csvs(paths: list[Path]) -> None:
    for path in paths:
        df = pd.read_csv(path, dtype=str)  # Read all as strings to avoid dtype issues
        clean_columns(df)
        out_path = path.with_name(path.stem + "_cleaned.csv")
        df.to_csv(out_path, index=False)
        print(f"✅ Saved cleaned file: {out_path}")

if __name__ == "__main__":
    files = [
        Path("data/parsed/train.csv"),
        Path("data/parsed/test.csv"),
        Path("data/parsed/val.csv"),
    ]
    process_csvs(files)