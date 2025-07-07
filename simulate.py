"""
simulate.py
==================
This script simulates a series of games between MCTS and ML-MCTS bots,
records the results, and saves them to a database.
"""

import argparse
import time
from app import app, db, GameResult, vanilla_mcts, ml_mcts
from tfg.game.board import Board

def play_game(bot1, bot2, iters):
    """Play a game between two bots and return the winner and duration.

    Args:
        bot1 (_type_): The first bot to play.
        bot2 (_type_): The second bot to play.
        iters (int): Number of iterations for MCTS simulations.

    Returns:
        tuple: A tuple containing the winner's class name and the duration of the game in seconds.
    """
    b1, b2 = Board(), Board()
    b1.place_fleet(); b2.place_fleet()
    turn   = 1
    start  = time.time()

    while True:
        if turn == 1:
            x,y = bot1.run(b2)
            b2.shoot(x,y)
            if b2.has_won():
                winner = bot1.__class__.__name__
                break
            turn = 2
        else:
            x,y = bot2.run(b1)
            b1.shoot(x,y)
            if b1.has_won():
                winner = bot2.__class__.__name__
                break
            turn = 1

    return winner, time.time() - start

def main():
    """Main function to parse arguments and simulate games between MCTS and ML-MCTS bots.
    """
    parser = argparse.ArgumentParser(description="Simulate MCTS vs ML-MCTS and record to DB")
    parser.add_argument('--games', type=int, default=100, help="Number of games to simulate")
    parser.add_argument('--iters', type=int, default=100, help="Simulations per move")
    args = parser.parse_args()

    # Push a Flask app context so SQLAlchemy knows which db to use
    with app.app_context():
        for i in range(1, args.games + 1):
            winner, dur = play_game(vanilla_mcts, ml_mcts, args.iters)
            gr = GameResult(
                game_mode='mcts_vs_ml_mcts',
                winner=winner,
                duration=dur
            )
            db.session.add(gr)
            db.session.commit()
            print(f"[{i}/{args.games}] Winner: {winner}, duration {dur:.2f}s")

if __name__ == '__main__':
    main()
