import numpy as np

def split_action_type(y):
    types = np.array([
        "move" if act.startswith("move_") else
        "forced" if act.startswith("forced_switch_") else
        "switch"
        for act in y
    ])
    return types
