import numpy as np

def split_action_type(y):
    types = np.array(["move" if act.startswith("move_") else "switch"
                      for act in y])
    return types