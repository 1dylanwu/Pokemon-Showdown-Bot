import os
import time
from tqdm import tqdm
from replay_scraper import get_replays, download_log

OUTPUT_DIR = "data/logs/gen9randombattle_logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
#create directory

def save_replays(pages: int = 50, delay: float = 0.5):
    #saves replay logs to text files into the output directory
    replays = get_replays(max_pages=pages)

    for replay in tqdm(replays, desc="Downloading logs"):
        #progress bar (tqdm)

        replay_id = replay["id"]
        filename = os.path.join(OUTPUT_DIR, f"{replay_id}.log")
        #file name based on replay ID

        if os.path.exists(filename):
            print(f"Skipping {replay_id}, already exists.")
            continue  #skip if already downloaded
        try:
            log_text = download_log(replay_id)
            with open(filename, "w", encoding="utf-8") as f:
                f.write(log_text)
                #downloads log and saves to file
        except Exception as e:
            print(f"Failed {replay_id}: {e}")
        
        #wait delay seconds between every replay download
        time.sleep(delay)

if __name__ == "__main__":
    #script runs directly
    save_replays(pages=50, delay=0.2)