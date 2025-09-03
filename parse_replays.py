import re
import pandas as pd

def parse_showdown_log(log_path):
    with open(log_path, 'r', encoding='utf8') as f:
        lines = f.readlines()

    battle_state = init_battle_state()
    features_per_turn = []
    current_turn = 0

    for line in lines:
        line = line.strip()
        if line.startswith("|turn|"):
            if current_turn > 0:
                features_per_turn.append(flatten_state(battle_state, current_turn))
            current_turn = int(line.split('|')[2])
        else:
            update_battle_state(line, battle_state)
    # Capture final turn
    features_per_turn.append(flatten_state(battle_state, current_turn))
    return pd.DataFrame(features_per_turn)

def init_battle_state():
    # Initializes a dictionary with all features to track per turn
    return {
        'active_pokemon': {'p1': None, 'p2': None},
        'hp': {'p1': None, 'p2': None},
        'status': {'p1': None, 'p2': None},
        'last_move': {'p1': None, 'p2': None},
        'field': {'weather': None, 'terrain': None},
        'hazards': {'p1': set(), 'p2': set()},
        'turn_result': {'faint': False, 'gameover': False}
    }

def update_battle_state(line, state):
    # Example patterns:
    # |switch|p1a: Garchomp|Garchomp, L80, F|100/100
    if line.startswith('|switch|'):
        m = re.match(r'\|switch\|(p\d)a?: ([^|]+)\|([^,]+),.*\|([\d/]+)', line)
        if m:
            player = m.group(1)
            poke = m.group(3)
            hp = m.group(4)
            state['active_pokemon'][player] = poke
            state['hp'][player] = hp
    elif line.startswith('|move|'):
        m = re.match(r'\|move\|(p\d)a?: ([^|]+)\|([^\|]+)', line)
        if m:
            player = m.group(1)
            move = m.group(3)
            state['last_move'][player] = move
    elif line.startswith('|-status|'):
        m = re.match(r'\|-status\|(p\d)a?: ([^|]+)\|([A-Za-z]+)', line)
        if m:
            player = m.group(1)
            status = m.group(3)
            state['status'][player] = status
    elif line.startswith('|-weather|'):
        weather = line.split('|')[2]
        state['field']['weather'] = weather
    elif line.startswith('|win|'):
        state['turn_result']['gameover'] = True

def flatten_state(state, turn):
    # Builds a flat dict for the DataFrame row
    flat = {'turn': turn}
    for key, d in state.items():
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    for subk, subv in v.items():
                        flat[f'{key}_{k}_{subk}'] = subv
                else:
                    flat[f'{key}_{k}'] = v
        else:
            flat[key] = d
    return flat

df = parse_showdown_log("data/logs/gen9randombattle_logs/gen9randombattle-2099652333.log")
print(df[df["turn"] == 5])