import requests
import time

#fetches a list of replay from Pokemon Showdown replay search API
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