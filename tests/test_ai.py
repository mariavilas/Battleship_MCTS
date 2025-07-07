# tests/test_ai.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tfg.game.board      import Board, BOARD_SIZE
from tfg.algorithms.mcts import Node, MCTS
from tfg.ai.mcts_ml      import NeuralMCTS


def test_best_child_selection():
    """Ensure best_child picks the node with the highest UCT score."""
    parent = Node(Board())
    parent.visits = 10

    # Child A: lower prior, more visits
    a = Node(Board(), parent=parent, action=(0, 0), prior=0.2)
    a.visits, a.wins = 5, 3

    # Child B: higher prior, fewer visits
    b = Node(Board(), parent=parent, action=(1, 1), prior=0.8)
    b.visits, b.wins = 3, 2

    parent.children = [a, b]
    best = parent.best_child(c=1.41, c_puct=0.5)
    assert best is b


def test_expand_single_move():
    """When only one legal move remains, expand() should create exactly that child."""
    board = Board()
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if (i, j) != (2, 3):
                board.shoot(i, j)

    node = Node(board)
    child = node.expand()
    assert child is not None
    assert child.action == (2, 3)
    assert len(node.children) == 1


def test_backpropagate_path():
    """Backpropagate a win from a depth-3 leaf: each node on the path gets +1 visit and +1 win."""
    root   = Node(Board())
    child1 = Node(Board(), parent=root)
    child2 = Node(Board(), parent=child1)
    leaf   = Node(Board(), parent=child2)

    root.children   = [child1]
    child1.children = [child2]
    child2.children = [leaf]

    leaf.backpropagate(result=1)

    for node in (leaf, child2, child1, root):
        assert node.visits == 1
        assert node.wins   == 1


def test_mcts_returns_legal_move():
    """MCTS.run should return a tuple within bounds and target an unshot cell."""
    board = Board()
    mcts  = MCTS(iterations=20)
    move  = mcts.run(board)

    assert isinstance(move, tuple) and len(move) == 2
    x, y = move
    assert 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE
    assert board.get_cell(x, y) not in ['X', 'O']


def test_mcts_with_policy_returns_valid_pi():
    """run_with_policy should produce a valid move and a normalized probability vector."""
    board = Board()
    mcts  = MCTS(iterations=20)
    move, pi = mcts.run_with_policy(board)

    # Move is within the board and not a repeat
    assert isinstance(move, tuple) and len(move) == 2
    x, y = move
    assert 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE
    assert board.get_cell(x, y) not in ['X', 'O']

    # π-vector sums to 1 and contains non-negative floats
    assert isinstance(pi, list)
    assert len(pi) == BOARD_SIZE * BOARD_SIZE
    assert abs(sum(pi) - 1.0) < 1e-6
    assert all(p >= 0.0 for p in pi)


def test_ml_mcts_uniform_network():
    """ML-MCTS with a uniform policy network should still return a legal move."""
    class DummyMLMCTS(NeuralMCTS):
        def __init__(self, iters=20, c_puct=1.0):
            # Bypass loading any actual model
            self.iters       = iters
            self.c_puct      = c_puct
            self.alpha_noise = 0.3
            self.eps_noise   = 0.25  # matches the attribute used in run()

        def _evaluate(self, board):
            total = BOARD_SIZE * BOARD_SIZE
            priors = [1/total] * total
            return priors, 0.0

    board = Board()
    mcts  = DummyMLMCTS(iters=20)
    move  = mcts.run(board)

    assert isinstance(move, tuple) and len(move) == 2
    x, y = move
    assert 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE
