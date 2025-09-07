import asyncio
from poke_env import RandomPlayer, SimpleHeuristicsPlayer
from poke_env import LocalhostServerConfiguration
from src.bot.rules_bot import RulesBot
# tests connecting to local showdown server and using poke-env/async
# by creating 2 bots that choose random moves and battling them against each other
# config
BATTLE_FORMAT = "gen9randombattle"
N_BATTLES = 1

# connect to server
server_config = LocalhostServerConfiguration

async def main():
    # create 2 random players
    player1 = SimpleHeuristicsPlayer(
        battle_format=BATTLE_FORMAT,
        server_configuration=server_config,
        max_concurrent_battles=1
    )
    player2 = RulesBot(
    )
    player1._username = "SimpleHeuristicsBot"
    player2._username = "RulesBot"
    await player1.battle_against(player2, n_battles=N_BATTLES)

    print(f"\nResults after {N_BATTLES} battles:")
    print(f"{player1._username} won {player1.n_won_battles}")
    print(f"{player2._username} won {player2.n_won_battles}")
    #print results

if __name__ == "__main__":
    asyncio.run(main())
