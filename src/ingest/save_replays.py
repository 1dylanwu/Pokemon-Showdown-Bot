import os
import time
from tqdm import tqdm
import requests

OUTPUT_DIR = "data/raw/gen9randombattle_logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
#create directory
def get_replays(format_id="gen9randombattle", max_pages=1, delay:float = 1.0):
    base_url = "https://replay.pokemonshowdown.com/search.json"
    all_replays = []

    for page in range(1, max_pages + 1):
        #loops through each page of replay results
        params = {
            "format": format_id,
            "sort": "rating",
            #sorts by rating to get higher rated battles first
            "page": page
        }

        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()
        #raises error if request fails

        data = resp.json()
        if not data:
            break
        all_replays.extend(data)
        #parses and adds json data to the list

        time.sleep(delay)
        #time delay to avoid rate limiting(delay seconds for every page)
    return all_replays

def download_log(replay_id: str) -> str:
    #downloads the log text for a given replay ID
    log_url = f"https://replay.pokemonshowdown.com/{replay_id}.log"
    resp = requests.get(log_url, timeout=10)
    resp.raise_for_status()
    return resp.text
    #plain text of full battle log

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