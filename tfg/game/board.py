"""
tfg.game.board
===================
This module defines the game board for the Battleship game, including methods for placing ships, shooting
ships, and checking game status. It also provides a method to convert the board state into a tensor representation suitable for neural networks.
"""

import random
import numpy as np

SEA = " "    
SHIPS = [3, 2, 2, 1, 1]
HIT = 'X'
MISS = 'O'
BOARD_SIZE = 6

class Board:
    """ Board represents the game board for Battleship.
    """
    def __init__(self):
        """ Initializes an empty board and an empty fleet of boats.
        """
        self.board = self.empty_board()
        self.boats = []

    def empty_board(self):
        """ Creates an empty board filled with SEA.

        Returns:
            list: A 2D list representing the empty board, where each cell is initialized to SEA.
        """
        return [[SEA for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

    def get_cell(self, x, y):
        """ Returns the value of the cell at position (x, y) on the board.

        Args:
            x (int): The row index of the cell.
            y (int): The column index of the cell.

        Returns:
            str: The value of the cell at (x, y), which can be SEA, HIT, or MISS.
        """
        return self.board[x][y]

    def place_fleet(self):
        """ Randomly places the fleet of ships on the board.
        """
        self.boats = []
        for ship in SHIPS:
            while True:
                x = random.randint(0, BOARD_SIZE - 1)
                y = random.randint(0, BOARD_SIZE - 1)
                direction = random.choice(['H', 'V'])
                if self.can_place_ship(x, y, direction, ship):
                    positions = self.place_ship(x, y, direction, ship)
                    self.boats.append({"value": str(ship), "positions": positions})
                    break

    def can_place_ship(self, x, y, direction, ship):
        """ Checks if a ship can be placed on the board at the specified position and direction.

        Args:
            x (int): The row index where the ship is to be placed.
            y (int): The column index where the ship is to be placed.
            direction (str): The direction of the ship ('H' for horizontal, 'V' for vertical).
            ship (int): The size of the ship to be placed.
        """
        def is_valid(i, j):
            """ Checks if the given indices (i, j) are within the bounds of the board.

            Args:
                i (int): The row index.
                j (int): The column index.

            Returns:
                bool: True if (i, j) is within the bounds of the board, False otherwise.
            """
            return 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE
        def is_empty_or_sea(i, j):
            """ Checks if the cell at (i, j) is either empty or SEA.

            Args:
                i (int): The row index.
                j (int): The column index.

            Returns:
                bool: True if the cell is SEA or empty, False otherwise.
            """
            if not is_valid(i, j):
                return True
            return self.board[i][j] == SEA

        if (direction == 'H' and y + ship > BOARD_SIZE) or (direction == 'V' and x + ship > BOARD_SIZE):
            return False

        for i in range(ship):
            nx, ny = (x, y + i) if direction == 'H' else (x + i, y)
            if not is_empty_or_sea(nx, ny):
                return False
            for adj_x, adj_y in [
                (nx-1,ny),(nx+1,ny),(nx,ny-1),(nx,ny+1),
                (nx-1,ny-1),(nx-1,ny+1),(nx+1,ny-1),(nx+1,ny+1)
            ]:
                if not is_empty_or_sea(adj_x, adj_y):
                    return False
        return True

    def place_ship(self, x, y, direction, ship):
        """ Places a ship on the board at the specified position and direction.

        Args:
            x (int): The row index where the ship is to be placed.
            y (int): The column index where the ship is to be placed.
            direction (str): The direction of the ship ('H' for horizontal, 'V' for vertical).
            ship (int): The size of the ship to be placed.

        Returns:
            list: A list of tuples representing the positions occupied by the ship on the board.
        """
        positions = []
        if direction == 'H':
            for i in range(ship):
                self.board[x][y+i] = str(ship)
                positions.append((x, y+i))
        else:
            for i in range(ship):
                self.board[x+i][y] = str(ship)
                positions.append((x+i, y))
        return positions

    def shoot(self, x, y):
        """ Shoots at the specified position on the board.

        Args:
            x (int): The row index of the cell to shoot at.
            y (int): The column index of the cell to shoot at.

        Returns:
            bool: True if the shot hit a ship, False if it was a miss or already shot.
        """
        if self.board[x][y] in [HIT, MISS]:
            return False
        self.board[x][y] = HIT if self.board[x][y] != SEA else MISS
        return self.board[x][y] == HIT

    def has_won(self):
        """ Checks if all boats have been sunk.

        Returns:
            bool: True if all boats have been sunk, False otherwise.
        """
        for boat in self.boats:
            for (x, y) in boat["positions"]:
                if self.board[x][y] != HIT:
                    return False
        return True

    def get_boats_status(self):
        """ Checks the status of each boat on the board.

        Returns:
            list: A list of dictionaries, each containing the ship value and whether it is sunk.
        """
        statuses = []
        for boat in self.boats:
            sunk = all(self.board[x][y] == HIT for (x, y) in boat["positions"])
            statuses.append({"ship": boat["value"], "sunk": sunk})
        return statuses

    def to_tensor(self):
        """ Converts the board state to a tensor representation.

        Returns:
            np.ndarray: A 3D numpy array of shape (3, BOARD_SIZE, BOARD_SIZE) representing the board state.
            The first layer represents hits, the second layer represents misses, and the third layer represents empty cells.
        """
        tensor = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                c = self.board[i][j]
                if c == HIT:
                    tensor[0, i, j] = 1.0
                elif c == MISS:
                    tensor[1, i, j] = 1.0
                else:
                    tensor[2, i, j] = 1.0
        return tensor
