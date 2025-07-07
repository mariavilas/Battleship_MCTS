"""
self_play.py
==================
This script generates self-play data for the Battleship game using MCTS,
balances the dataset, and saves it in JSON Lines format.
"""

import json
import random
from tfg.game.board      import Board
from tfg.algorithms.mcts import MCTS

#  Configurable parameters 
NUM_GAMES  = 500 # total number of games to generate
ITERATIONS = 200 # MCTS simulations per move
OUT_FILE   = 'data_balanced.jsonl'

def play_one_game(mcts):
    """ Play a single game of Battleship using MCTS for self-play.

    Args:
        mcts (MCTS): An instance of the MCTS class to run simulations.

    Returns:
        list: A list of records, each containing the state, policy distribution (pi), and outcome (z).
    """
    # Initialize two boards and place fleets
    boardA = Board()
    boardB = Board()
    boardA.place_fleet()
    boardB.place_fleet()

    trajectory = []  # hold tuples - state, pi, current_player
    current = 'A'

    while True:
        target_board = boardB if current == 'A' else boardA

        # Run MCTS to get selected move and full policy distribution
        move, pi = mcts.run_with_policy(target_board)

        # Record the state and policy before executing the move
        trajectory.append((target_board.to_tensor().tolist(), pi, current))

        # Execute the move and check for a win
        target_board.shoot(*move)
        if target_board.has_won():
            winner = current
            break

        # change player
        current = 'B' if current == 'A' else 'A'

    # Convert trajectory into list of dicts with outcomes z
    records = []
    for state, pi, player in trajectory:
        z = 1 if player == winner else -1
        records.append({'state': state, 'pi': pi, 'z': z})

    return records

def balance_records(records):
    """ Balance the dataset by sampling an equal number of winning and losing records.

    Args:
        records (list): List of game records, each a dict with 'z' indicating win/loss.

    Returns:
        list: A balanced list of records with equal numbers of winners and losers.
    """
    winners = [r for r in records if r['z'] == 1]
    losers  = [r for r in records if r['z'] == -1]
    m = min(len(winners), len(losers))
    sampled = random.sample(winners, m) + random.sample(losers, m)
    random.shuffle(sampled)
    return sampled

def generate_and_balance(num_games=NUM_GAMES,
                         iters=ITERATIONS,
                         out_file=OUT_FILE):
    """ Generate self-play data for Battleship and balance the dataset.

    Args:
        num_games (int): Total number of games to generate.
        iters (int): Number of MCTS iterations per move.
        out_file (str): Output file path for the balanced dataset in JSON Lines format.
    """
    all_records = []

    for i in range(num_games):
        # Instantiate a fresh MCTS for each game
        mcts = MCTS(iterations=iters)
        game_recs = play_one_game(mcts)
        all_records.extend(game_recs)

        if (i + 1) % 100 == 0:
            print(f" Generated {i+1}/{num_games} games, "
                  f"{len(all_records)} total records so far")

    print("Balancing dataset")
    balanced = balance_records(all_records)
    print(f"Final balanced dataset size: {len(balanced)} records")

    print(f"Writing balanced records to {out_file}…")
    with open(out_file, 'w') as f:
        for rec in balanced:
            f.write(json.dumps(rec) + '\n')

    print("Self-play generation and balancing completed")

if __name__ == '__main__':
    generate_and_balance()
