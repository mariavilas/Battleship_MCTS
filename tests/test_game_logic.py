# tests/test_game_logic.py

import sys
import os
import pytest

# Ensure that the project root (where tfg/) is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tfg.game.board import Board, SHIPS, BOARD_SIZE, HIT, MISS

def test_place_full_fleet():
    """Exactly five ships are placed, none overlap, and all remain inside the grid."""
    board = Board()
    board.place_fleet()
    assert len(board.boats) == len(SHIPS)

    all_positions = [pos for boat in board.boats for pos in boat["positions"]]
    # No overlapping coordinates
    assert len(all_positions) == len(set(all_positions))
    # All positions lie within valid range
    for x, y in all_positions:
        assert 0 <= x < BOARD_SIZE
        assert 0 <= y < BOARD_SIZE

def test_place_3_cell_ship_horizontal_top_left():
    """A length-3 ship can be successfully placed horizontally at (0,0)."""
    board = Board()
    assert board.can_place_ship(0, 0, 'H', 3) is True

    positions = board.place_ship(0, 0, 'H', 3)
    assert positions == [(0, 0), (0, 1), (0, 2)]

    for x, y in positions:
        # The board should store the ship’s size as its marker
        assert board.board[x][y] == '3'

def test_place_3_cell_ship_vertical_out_of_bounds():
    """Attempting to place a length-3 ship vertically at (4,4) should be rejected."""
    board = Board()
    assert board.can_place_ship(4, 4, 'V', 3) is False

def test_overlap_rejected_after_initial_placement():
    """Placing a second ship directly on top of an existing one must be rejected."""
    board = Board()
    # First placement succeeds
    assert board.can_place_ship(0, 0, 'H', 3) is True
    board.place_ship(0, 0, 'H', 3)
    # Same placement now overlaps
    assert board.can_place_ship(0, 0, 'H', 3) is False

def test_shoot_hit_and_record():
    """Shooting a cell containing a ship part returns True and marks a hit."""
    board = Board()
    # Manually plant a ship fragment at (2,3)
    board.board[2][3] = '2'

    result = board.shoot(2, 3)
    assert result is True
    assert board.board[2][3] == HIT

def test_shoot_same_cell_twice():
    """Shooting an already-hit cell returns False and does not change the marker."""
    board = Board()
    board.board[1][1] = '1'

    assert board.shoot(1, 1) is True
    # second shot on the same square should do nothing
    assert board.shoot(1, 1) is False
    assert board.board[1][1] == HIT


def test_has_won_after_all_ship_cells_hit():
    """has_won() must return True only after every ship cell has been hit."""
    board = Board()
    # Place a 1 cell ship at (0,0) and register it in the boats list
    positions = board.place_ship(0, 0, 'H', 1)
    board.boats = [{"value": "1", "positions": positions}]
    assert board.has_won() is False

    # Hit that only cell
    board.shoot(0, 0)
    assert board.has_won() is True
