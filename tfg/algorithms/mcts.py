"""
tfg.algorithms.mcts
======================
This module implements a Monte Carlo Tree Search (MCTS) algorithm for the Battleship game.  
"""

import random
import math
import copy
from tfg.game.board import Board, SHIPS, BOARD_SIZE

class Node:
    """Node in the Monte Carlo Tree Search (MCTS) tree.
    """
    def __init__(self, state: Board, parent=None, action=None, prior: float = 0.0):
        """ Initializes a new MCTS node.

        Args:
            state (Board): The game state represented by this node.
            parent (Node, optional): The parent node in the MCTS tree. Defaults to None.
            action (tuple, optional): The action taken to reach this node (x, y). Defaults to None.
            prior (float, optional): Prior probability of this node's action. Defaults to 0.0.
        """
        self.state = state        
        self.parent = parent      
        self.action = action      
        self.children = []           
        self.visits = 0        
        self.wins = 0            
        self.prior = prior    

    def uct_value(self, c: float = 1.41, c_puct: float = 0.5):
        """ Calculate the UCT value for this node.

        Args:
            c (float, optional): Exploration constant for balancing exploration and exploitation. Defaults to 1.41.
            c_puct (float, optional): Exploration constant for balancing prior probability. Defaults to 0.5.

        Returns:
            float: The UCT value of the node, combining exploitation and exploration.
        """
        if self.visits == 0:
            return float('inf')
        q = self.wins / self.visits
        u = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        p = c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return q + u + p

    def best_child(self, c: float = 1.41, c_puct: float = 0.5):
        """ Find the best child node based on UCT value.

        Args:
            c (float, optional): Exploration constant for balancing exploration and exploitation. Defaults to 1.41.
            c_puct (float, optional): Exploration constant for balancing prior probability. Defaults to 0.5.
        Returns:
            Node: The child node with the highest UCT value.
        """
        return max(self.children, key=lambda ch: ch.uct_value(c, c_puct))

    def expand(self):
        """ Expand the node by adding a new child node for an untried move.

        Returns:
            Node: A new child node representing the state after making an untried move, or None if no untried moves are available.
        """
        # Find all legal moves from this state, then pick one untried
        available = [
            (i, j)
            for i in range(BOARD_SIZE)
            for j in range(BOARD_SIZE)
            if self.state.get_cell(i, j) not in ['X', 'O']
        ]
        tried = {ch.action for ch in self.children}
        untried = [m for m in available if m not in tried]
        if not untried:
            return None

        move = random.choice(untried)
        new_state = copy.deepcopy(self.state)
        new_state.shoot(*move)

        # lookup prior from the combined heatmap (attached to this node)
        prior = 0.0
        if hasattr(self, 'heatmap') and self.heatmap:
            idx = move[0] * BOARD_SIZE + move[1]
            prior = self.heatmap[idx]

        child = Node(new_state, parent=self, action=move, prior=prior)
        # pass the same heatmap down to children
        child.heatmap = getattr(self, 'heatmap', None)
        self.children.append(child)
        return child

    def backpropagate(self, result: int):
        """ Backpropagate the result of a simulation up the tree.

        Args:
            result (int): The result of the simulation (1 for win, 0 for loss).
        """
        node = self
        while node:
            node.visits += 1
            node.wins   += result
            node = node.parent
    
    def to_dict(self):
        """ Convert the node to a dictionary representation.

        Returns:
            dict: A dictionary containing the node's action, visit count, win count, win rate, and children.
        """
        return {
            "action": self.action,
            "visits": self.visits,
            "wins": self.wins,
            "win_rate": (self.wins / self.visits) if self.visits else 0.0,
            "children": [c.to_dict() for c in self.children]
    }


class MCTS:
    """Monte Carlo Tree Search (MCTS) algorithm for the Battleship game.
    """
    def __init__(self, iterations: int = 200):
        """ Initializes the MCTS instance.

        Args:
            iterations (int, optional): Number of MCTS iterations to run. Defaults to 200.
        """
        self.iterations = iterations
        self.heatmap = []
        self.root = None
        
    def update_with_move(self, move):
        """ Updates the MCTS tree to reflect a move made by the opponent.

        Args:
            move (tuple): The move made by the opponent, represented as (x, y).
        """
        if not self.root:
            return
        for c in self.root.children:
            if c.action == move:
                c.parent = None
                self.root = c
                return
        self.root = None

    def compute_heatmap(self, board: Board):
        """ Computes a static heatmap for the board based on valid ship placements.

        Args:
            board (Board): The current game board.

        Returns:
            list: A flat list representing the heatmap, where each cell's value indicates the number of valid ship placements covering that cell.
        """
        # Static heatmap: count every valid ship placement covering each cell
        counts = [[0]*BOARD_SIZE for _ in range(BOARD_SIZE)]
        for ship in SHIPS:
            for x in range(BOARD_SIZE):
                for y in range(BOARD_SIZE):
                    if board.can_place_ship(x, y, 'H', ship):
                        for k in range(ship):
                            counts[x][y+k] += 1
                    if board.can_place_ship(x, y, 'V', ship):
                        for k in range(ship):
                            counts[x+k][y] += 1
        flat = [counts[i][j] for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)]
        total = sum(flat) or 1
        return [c/total for c in flat]

    def run(self, board: Board, iterations: int = None):
        """ Runs the MCTS algorithm for a given number of iterations.

        Args:
            board (Board): The current game board.
            iterations (int, optional): Number of MCTS iterations to run. Defaults to self.iterations.

        Returns:
            tuple: The best move (x, y) determined by MCTS.
        """
        iters = iterations or self.iterations

        # Recompute heatmap (static + dynamic) 
        static_map = self.compute_heatmap(board)
        target_map = [0.0] * (BOARD_SIZE * BOARD_SIZE)
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if board.get_cell(x, y) == 'X':
                    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        nx, ny = x+dx, y+dy
                        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                            if board.get_cell(nx, ny) == '?':
                                target_map[nx*BOARD_SIZE + ny] = 1.0
        combo = [
            0.3 * static_map[i] + 0.7 * target_map[i]
            for i in range(BOARD_SIZE * BOARD_SIZE)
        ]
        s = sum(combo) or 1.0
        self.heatmap = [c/s for c in combo]

        # Initialize or reuse root
        if self.root is None:
            self.root        = Node(copy.deepcopy(board))
            self.root.visits = 1
        self.root.heatmap = self.heatmap

        # MCTS main loop
        for _ in range(iters):
            node = self.root

            # selection
            while True:
                # all legal moves from this node
                moves = [
                    (i, j)
                    for i in range(BOARD_SIZE)
                    for j in range(BOARD_SIZE)
                    if node.state.get_cell(i, j) not in ['X', 'O']
                ]
                # stop if leaf or there are untried moves here
                if not moves or len(node.children) < len(moves):
                    break
                # otherwise descend along best UCT child
                node = node.best_child()

            # expansion
            moves = [
                (i, j)
                for i in range(BOARD_SIZE)
                for j in range(BOARD_SIZE)
                if node.state.get_cell(i, j) not in ['X', 'O']
            ]
            tried   = {c.action for c in node.children}
            untried = [m for m in moves if m not in tried]
            if untried:
                move = random.choice(untried)
                new_state = copy.deepcopy(node.state)
                new_state.shoot(*move)
                child = Node(new_state, parent=node, action=move, prior=0.0)
                child.heatmap = self.heatmap
                node.children.append(child)
                node = child

            # simulation
            result = self.simulate(copy.deepcopy(node.state))

            # backpropagation
            node.backpropagate(result)

        # Select best move (most visits) from root
        best = max(self.root.children, key=lambda c: c.visits)
        return best.action



    def simulate(self, sim: Board):
        """ Simulates a game from the given board state until a win or draw is reached.

        Args:
            sim (Board): The current game board to simulate from.

        Returns:
            int: 1 if the simulation results in a win for the player, 0 otherwise.
        """
        hunt = []          
        flat = self.heatmap

        while not sim.has_won():
            if hunt:
                x, y = hunt.pop(0)
                if sim.get_cell(x, y) in ['X', 'O']:
                    continue
            else:
                avail = [
                    (i, j) for i in range(BOARD_SIZE)
                           for j in range(BOARD_SIZE)
                           if sim.get_cell(i, j) not in ['X','O']
                ]
                if not avail:
                    break
                weights = [flat[i*BOARD_SIZE + j] for (i, j) in avail]
                if sum(weights) > 0:
                    x, y = random.choices(avail, weights)[0]
                else:
                    x, y = random.choice(avail)

            hit = sim.shoot(x, y)
            if hit:
                for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                        if sim.get_cell(nx, ny) not in ['X','O']:
                            hunt.append((nx, ny))

        return 1 if sim.has_won() else 0

    def run_with_policy(self, board: Board, iterations: int = None):
        """ Runs MCTS with a policy that combines static heatmap and target map based on the current board state.

        Args:
            board (Board): The current game board.
            iterations (int, optional): Number of MCTS iterations to run. Defaults to self.iterations.

        Returns:
            tuple: The best move (x, y) determined by MCTS and a flat policy vector π.
        """
        iters = iterations or self.iterations

        # reuse heatmap logic from run()
        static_map = self.compute_heatmap(board)
        target_map = [0.0] * (BOARD_SIZE * BOARD_SIZE)
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if board.get_cell(x, y) == 'X':
                    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        nx, ny = x+dx, y+dy
                        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                            if board.get_cell(nx, ny) == '?':
                                target_map[nx*BOARD_SIZE + ny] = 1.0

        combo = [
            0.3 * static_map[i] + 0.7 * target_map[i]
            for i in range(BOARD_SIZE * BOARD_SIZE)
        ]
        s = sum(combo) or 1.0
        self.heatmap = [c/s for c in combo]

        # Build root
        root = Node(copy.deepcopy(board))
        root.visits  = 1
        root.heatmap = self.heatmap

        # MCTS loop
        for _ in range(iters):
            node = root
            while node.children:
                node = node.best_child()
            new_node = node.expand()
            node = new_node or node
            result = self.simulate(copy.deepcopy(node.state))
            node.backpropagate(result)

        # Collect visit counts for policy
        visits = { ch.action: ch.visits for ch in root.children }
        total  = sum(visits.values()) or 1

        # Build flat pi vector
        pi = [
            visits.get((i, j), 0) / total
            for i in range(BOARD_SIZE)
            for j in range(BOARD_SIZE)
        ]

        # Best move
        best_move = max(root.children, key=lambda ch: ch.visits).action
        return best_move, pi
