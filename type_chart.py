
TYPE_EFFECTIVENESS = {
    "NORMAL": {
        "ROCK": 0.5, "STEEL": 0.5, "GHOST": 0.0
    },
    "FIRE": {
        "GRASS": 2.0, "ICE": 2.0, "BUG": 2.0, "STEEL": 2.0,
        "FIRE": 0.5, "WATER": 0.5, "ROCK": 0.5, "DRAGON": 0.5
    },
    "WATER": {
        "FIRE": 2.0, "GROUND": 2.0, "ROCK": 2.0,
        "WATER": 0.5, "GRASS": 0.5, "DRAGON": 0.5
    },
    "ELECTRIC": {
        "WATER": 2.0, "FLYING": 2.0,
        "ELECTRIC": 0.5, "GRASS": 0.5, "DRAGON": 0.5,
        "GROUND": 0.0
    },
    "GRASS": {
        "WATER": 2.0, "GROUND": 2.0, "ROCK": 2.0,
        "FIRE": 0.5, "GRASS": 0.5, "POISON": 0.5, "FLYING": 0.5,
        "BUG": 0.5, "DRAGON": 0.5, "STEEL": 0.5
    },
    "ICE": {
        "GRASS": 2.0, "GROUND": 2.0, "FLYING": 2.0, "DRAGON": 2.0,
        "FIRE": 0.5, "WATER": 0.5, "ICE": 0.5, "STEEL": 0.5
    },
    "FIGHTING": {
        "NORMAL": 2.0, "ICE": 2.0, "ROCK": 2.0, "DARK": 2.0, "STEEL": 2.0,
        "POISON": 0.5, "FLYING": 0.5, "PSYCHIC": 0.5, "BUG": 0.5, "FAIRY": 0.5,
        "GHOST": 0.0
    },
    "POISON": {
        "GRASS": 2.0, "FAIRY": 2.0,
        "POISON": 0.5, "GROUND": 0.5, "ROCK": 0.5, "GHOST": 0.5,
        "STEEL": 0.0
    },
    "GROUND": {
        "FIRE": 2.0, "ELECTRIC": 2.0, "POISON": 2.0, "ROCK": 2.0, "STEEL": 2.0,
        "GRASS": 0.5, "BUG": 0.5,
        "FLYING": 0.0
    },
    "FLYING": {
        "GRASS": 2.0, "FIGHTING": 2.0, "BUG": 2.0,
        "ELECTRIC": 0.5, "ROCK": 0.5, "STEEL": 0.5
    },
    "PSYCHIC": {
        "FIGHTING": 2.0, "POISON": 2.0,
        "PSYCHIC": 0.5, "STEEL": 0.5,
        "DARK": 0.0
    },
    "BUG": {
        "GRASS": 2.0, "PSYCHIC": 2.0, "DARK": 2.0,
        "FIRE": 0.5, "FIGHTING": 0.5, "POISON": 0.5, "FLYING": 0.5,
        "GHOST": 0.5, "STEEL": 0.5, "FAIRY": 0.5
    },
    "ROCK": {
        "FIRE": 2.0, "ICE": 2.0, "FLYING": 2.0, "BUG": 2.0,
        "FIGHTING": 0.5, "GROUND": 0.5, "STEEL": 0.5
    },
    "GHOST": {
        "PSYCHIC": 2.0, "GHOST": 2.0,
        "DARK": 0.5, "NORMAL": 0.0
    },
    "DRAGON": {
        "DRAGON": 2.0,
        "STEEL": 0.5, "FAIRY": 0.0
    },
    "DARK": {
        "PSYCHIC": 2.0, "GHOST": 2.0,
        "FIGHTING": 0.5, "DARK": 0.5, "FAIRY": 0.5
    },
    "STEEL": {
        "ICE": 2.0, "ROCK": 2.0, "FAIRY": 2.0,
        "FIRE": 0.5, "WATER": 0.5, "ELECTRIC": 0.5, "STEEL": 0.5
    },
    "FAIRY": {
        "FIGHTING": 2.0, "DRAGON": 2.0, "DARK": 2.0,
        "FIRE": 0.5, "POISON": 0.5, "STEEL": 0.5
    }
}
def get_effectiveness(attacking_type, defending_types):
    #will return the effectiveness of attacking type vs defending type(s)
    mult = 1.0
    for defending_type in defending_types:
        mult *= TYPE_EFFECTIVENESS.get(attacking_type, {}).get(defending_type, 1.0)
    return mult