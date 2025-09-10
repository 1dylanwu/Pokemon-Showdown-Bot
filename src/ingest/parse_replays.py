from pathlib import Path
from collections import defaultdict
import pandas as pd

# entry hazards to track
HAZARDS = {"Stealth Rock", "Spikes", "Toxic Spikes", "Sticky Web"}

# status conditions to track
STATUS_TOKENS = {"brn", "par", "psn", "tox", "slp", "frz", "fnt"}

def normalize_species(mon: str) -> str:
    # stops treating all the different vivillon forms (vivillon-jungle, vivillon-modern etc) as different pokemon
    key = mon.lower()
    if key.startswith("vivillon-"):
        return "Vivillon"
    return mon

def slot_from(parts2: str) -> str:
    # extracts the slot (p1a, p2b, etc) from a parts[2] string
    # handles both pokemon and player slots
    return parts2.split(":", 1)[0].strip()


def side_from(parts2: str) -> str:
    # truncate slot to just the side (p1 or p2)
    return slot_from(parts2)[:2]


def parse_hp_fraction(hp_field: str):
    # parse hp tokens into a float fraction (0.0 to 1.0)
    if not hp_field:
        return None
    token = hp_field.split()[0]  # '72/100' or '0'
    if "/" in token:
        cur, mx = token.split("/")
        try:
            cur_i, mx_i = int(cur), int(mx)
            if mx_i > 0:
                return max(0.0, min(1.0, cur_i / mx_i))
        except ValueError:
            return None
    elif token == "0":
        return 0.0
    return None


def parse_status_from_hp_field(hp_field: str):
    # extract status from hp field
    # for example, get the psn from "20/281 psn"
    if not hp_field:
        return None
    parts = hp_field.split()
    if len(parts) >= 2:
        st = parts[1].strip()
        if st in STATUS_TOKENS:
            return st
    return None


def freeze_state(state: dict) -> dict:
    # create snapshot of live state with only plain types
    # what model will actually see as pre-action state
    is_tera_1 = state["p1a_active"] == state["p1a_tera_species"]
    is_tera_2 = state["p2a_active"] == state["p2a_tera_species"]

    return {
        "turn": state["turn"],

        # active slot (singles => a-slot)
        "p1a_active": state["p1a_active"],
        "p2a_active": state["p2a_active"],

        # active hp percentages
        "p1a_hp_pct": state["p1a_hp_pct"],
        "p2a_hp_pct": state["p2a_hp_pct"],

        # active status conditions (sleep, burn etc)
        "p1a_status": state["p1a_status"],
        "p2a_status": state["p2a_status"],

        # active stat boosts (can be negative)
        "p1a_boosts": dict(state["p1a_boosts"]),
        "p2a_boosts": dict(state["p2a_boosts"]),

        # entry hazards on each side
        "hazards_p1": list(state["hazards_p1"]),
        "hazards_p2": list(state["hazards_p2"]),
        "weather": state["weather"],
        "terrain": state["terrain"],

        # current alive/known team species
        "p1_team_species": sorted(state["p1_team_species"]),
        "p2_team_species": sorted(state["p2_team_species"]),

        # number of fainted mons on each side
        "p1a_fainted": state["p1a_fainted"],
        "p2a_fainted": state["p2a_fainted"],

        # known hp% of seen mons on each side
        "p1_known_hp": dict(state["p1_known_hp"]),
        "p2_known_hp": dict(state["p2_known_hp"]),

        # terastallization state
        "p1a_tera_type": state["p1a_tera_type"] if is_tera_1 else None,
        "p2a_tera_type": state["p2a_tera_type"] if is_tera_2 else None,
        "p1a_is_terastallized": is_tera_1,
        "p2a_is_terastallized": is_tera_2,
    }


def parse_battle_log(path: str | Path):
    # extract structured decision records from battle log
    # turn, side, action_type, action, state (pre-turn snapshot)
    # decisions for each side in a turn use the same pre-turn snapshot because they happen at the same time
    # we use a decision buffer because the state only updates at the start of each turn so we need to delay changing data until the next turn
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    fainted_slots = set() # track forced switches after faint
    # Live state for snapshots (what the model will see)
    state = {
        "turn": 0,
        "p1a_active": None, "p2a_active": None,
        "p1a_hp_pct": None, "p2a_hp_pct": None,
        "p1a_status": None, "p2a_status": None,
        "p1a_boosts": defaultdict(int), "p2a_boosts": defaultdict(int),
        "hazards_p1": [], "hazards_p2": [],
        "weather": None, "terrain": None,
        "p1_team_species": set(), "p2_team_species": set(),
        "p1a_fainted": 0, "p2a_fainted": 0,
        "p1_known_hp": {}, "p2_known_hp": {},
        "p1a_tera_type": None, "p2a_tera_type": None,
        "p1a_tera_species": None, "p2a_tera_species": None,
    }

    records = []
    decision_buffer = []
    turn_start_state = freeze_state(state)
    # the state each player sees before they pick their move

    def save_slot(slot: str):
        #save current active mon to team tracker for switching
        mon = state.get(f"{slot}_active")
        if mon:
            side = slot[:2]
            state[f"{side}_known_hp"][mon] = state.get(f"{slot}_hp_pct")


    def apply_switch(slot: str, mon_name: str, hp_field: str | None, tera_type: str | None):
        # apply switch effects to live state
        side = slot[:2]
        save_slot(slot)  # persist outgoing mon
        state[f"{slot}_active"] = mon_name
        state[f"{slot}_boosts"].clear()
        state[f"{side}_team_species"].add(mon_name)

        hp_pct = parse_hp_fraction(hp_field) if hp_field else None
        st = parse_status_from_hp_field(hp_field) if hp_field else None
        # If this mon is terastallized, record its Tera Type and species
        if tera_type:
            state[f"{slot}_tera_type"] = tera_type
            state[f"{slot}_tera_species"] = mon_name
        else:
            state[f"{slot}_tera_type"] = None
            state[f"{slot}_tera_species"] = None

        # defaults if not provided or not previously seen for new mon
        if hp_pct is None:
            hp_pct = state[f"{side}_known_hp"].get(mon_name, 1.0)

        state[f"{slot}_hp_pct"] = hp_pct
        state[f"{slot}_status"] = st

        # persist after switch-in
        state[f"{side}_known_hp"][mon_name] = hp_pct

    for raw in lines:
        parts = raw.split("|")
        if len(parts) < 2:
            continue
        tag = parts[1]
        if tag == "-terastallize":
            # |-terastallize|p1a: Kyogre|Water
            slot = slot_from(parts[2])
            tera_type = parts[3].strip() if len(parts) > 3 else None
            state[f"{slot}_tera_type"] = tera_type
            state[f"{slot}_tera_species"] = state[f"{slot}_active"]
            continue
        
        # New turn
        if tag == "turn":
            # add decisions from the previous turn
            if decision_buffer:
                records.extend(decision_buffer)
                decision_buffer.clear()

            # snapshot state at start of this turn (both players decide from this)
            turn_start_state = freeze_state(state)
            fainted_slots.clear()
            # Update turn counter
            try:
                state["turn"] = int(parts[2])
            except ValueError:
                pass
        
        elif tag == "faint":
            slot = slot_from(parts[2])
            side = slot[:2]
            mon = state[f"{slot}_active"]
            # clear active mon
            state[f"{side}_team_species"].discard(mon)
            state[f"{side}_known_hp"].pop(mon, None)
            state[f"{slot}_hp_pct"] = 0.0
            state[f"{slot}_status"] = "fnt"
            state[f"{slot}_boosts"].clear()
            state[f"{side}a_fainted"] += 1
            state[f"{side}_known_hp"][mon] = 0.0

            # set last fainted so we can track forced switch next
            fainted_slots.add(slot)

        # decision lines (buffered with pre-turn state)
        elif tag == "switch":
            slot = slot_from(parts[2])
            mon_name= normalize_species(parts[3].split(",", 1)[0].strip())
            hp_field = parts[4] if len(parts) > 4 else parts[3] if len(parts) > 3 else None
            meta_fields = parts[3].split(",") 
            tera_type = None
            for field in meta_fields:
                if field.strip().startswith("tera:"):
                    tera_type = field.strip().split(":", 1)[1]
            if slot in fainted_slots:
                # forced switch: record post-faint state
                records.append({
                    "turn": state["turn"],
                    "side": slot,
                    "action_type":"forced_switch",
                    "action": mon_name,
                    "state": freeze_state(state),
                })
                # now apply the incoming mon
                apply_switch(slot, mon_name, hp_field, tera_type)
                fainted_slots.remove(slot)
                #remove from fainted slots

            else:
                # player choice switch: buffer with pre-turn snapshot
                decision_buffer.append({
                    "turn": state["turn"],
                    "side": slot,
                    "action_type":"switch",
                    "action": mon_name,
                    "state": turn_start_state,
                })
                apply_switch(slot, mon_name, hp_field, tera_type)

        elif tag == "drag":
            # only update state for drag, dont record as decision since not player choice
            slot = slot_from(parts[2])
            mon_name = normalize_species(parts[3].split(",", 1)[0].strip())
            hp_field = parts[4] if len(parts) > 4 else None
            apply_switch(slot, mon_name, hp_field, tera_type)
            continue 

        elif tag == "move":
            slot = slot_from(parts[2])
            move_name = parts[3]
            if "[from]ability: Magic Bounce" in raw:
                continue
            decision_buffer.append({
                "turn": state["turn"],
                "side": slot,
                "action_type": "move",
                "action": move_name,
                "state": turn_start_state
            })
            # effects (damage, status) arrive via later tags

        # HP changes (damage, healing, status from damage)
        elif tag in ("damage", "heal", "-damage", "-heal"):
            slot = slot_from(parts[2])
            side = slot[:2]
            mon = state.get(f"{slot}_active")
            if len(parts) > 3:
                hp_pct = parse_hp_fraction(parts[3])
                if hp_pct is not None:
                    state[f"{slot}_hp_pct"] = hp_pct
                    if mon:
                        state[f"{side}_known_hp"][mon] = hp_pct

                st = parse_status_from_hp_field(parts[3])
                if st in STATUS_TOKENS and st != "fnt":
                    state[f"{slot}_status"] = st

        # Status application and cure
        elif tag in ("status", "-status"):
            slot = slot_from(parts[2])
            side = slot[:2]
            mon = state.get(f"{slot}_active")
            st = parts[3] if len(parts) > 3 else None
            state[f"{slot}_status"] = st

        elif tag in ("curestatus", "-curestatus"):
            slot = slot_from(parts[2])
            side = slot[:2]
            mon = state.get(f"{slot}_active")
            state[f"{slot}_status"] = None

        # boost changes
        elif tag in ("-boost", "-unboost"):
            slot = slot_from(parts[2])
            stat = parts[3] if len(parts) > 3 else None
            try:
                change = int(parts[4]) if len(parts) > 4 else 0
            except ValueError:
                change = 0
            delta = change if tag == "-boost" else -change
            if stat:
                state[f"{slot}_boosts"][stat] += delta

        elif tag == "-clearallboost":
            state["p1a_boosts"].clear()
            state["p2a_boosts"].clear()

        elif tag == "-clearboost":
            slot = slot_from(parts[2])
            stat = parts[3] if len(parts) > 3 else None
            if stat and stat in state[f"{slot}_boosts"]:
                del state[f"{slot}_boosts"][stat]

        # Hazards
        elif tag == "-sidestart":
            side = side_from(parts[2])
            cond = parts[3] if len(parts) > 3 else ""
            if ": " in cond:  # e.g., 'move: Stealth Rock'
                cond = cond.split(": ", 1)[1]
            if cond in HAZARDS and cond not in state[f"hazards_{side}"]:
                state[f"hazards_{side}"].append(cond)

        elif tag == "-sideend":
            side = side_from(parts[2])
            cond = parts[3] if len(parts) > 3 else ""
            if ": " in cond:
                cond = cond.split(": ", 1)[1]
            if cond in HAZARDS and cond in state[f"hazards_{side}"]:
                state[f"hazards_{side}"].remove(cond)

        # Weather and terrain
        elif tag == "-weather":
            w = parts[2] if len(parts) > 2 else None
            state["weather"] = None if w in (None, "none") else w

        elif tag == "-fieldstart":
            cond = parts[2] if len(parts) > 2 else ""
            if ": " in cond:
                cond = cond.split(": ", 1)[1]
            state["terrain"] = cond or None

        elif tag == "-fieldend":
            state["terrain"] = None

        # Ignore other tags (including win/tie)

    # flush any decisions from the final turn
    if decision_buffer:
        records.extend(decision_buffer)
        decision_buffer.clear()

    return records
