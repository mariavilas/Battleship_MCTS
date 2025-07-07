"""
tfg.ai.mcts_ml
========================
This module implements a Neural-guided Monte Carlo Tree Search (MCTS) for Battleship using a PyTorch policy/value network.
"""
import copy
import math
import numpy as np
import torch
from tfg.game.board import Board, BOARD_SIZE
from tfg.ai.network import GameNet

class NNode:
    """Node in the Monte Carlo Tree Search (MCTS) tree.
    
        Attributes:
            state (Board): The game state represented by this node.
            parent (NNode): The parent node in the MCTS tree.   
            action (tuple): The action taken to reach this node (i, j).
            children (list): List of child nodes.
            visit_count (int): Number of times this node has been visited.
            value_sum (float): Sum of values from backpropagation.
            prior (float): Prior probability of this node's action.
    """
    def __init__(self, state: Board, parent=None, action=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior  # policy prior pi(s,a)

    def q_value(self):
        """ Calculate the average value of this node based on visit counts.

        Returns:
            float: The average value of the node, or 0.0 if not visited.
        """
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def u_value(self, c_puct):
        """ Calculate the exploration value of this node.

        Args:
            c_puct (float): Exploration constant for balancing exploration and exploitation.

        Returns:
            float: The exploration value of the node.
        """
        return c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)

    def score(self, c_puct):
        """ Calculate the score of this node, combining exploitation and exploration.

        Args:
            c_puct (float): Exploration constant for balancing exploration and exploitation.

        Returns:
            float: The score of the node, combining Q-value and U-value.
        """
        return self.q_value() + self.u_value(c_puct)

    def expand(self, priors):
        """ Expand the node by generating child nodes for all possible moves.

        Args:
            priors (list): List of prior probabilities for each possible move.
        """
        moves = [
            (i, j)
            for i in range(BOARD_SIZE)
            for j in range(BOARD_SIZE)
            if self.state.get_cell(i,j) not in ['X','O']
        ]
        for (i, j) in moves:
            idx   = i * BOARD_SIZE + j
            new_st = copy.deepcopy(self.state)
            new_st.shoot(i, j)
            child = NNode(new_st, parent=self, action=(i, j), prior=priors[idx])
            self.children.append(child)

    def backpropagate(self, value):
        """ Backpropagate the value from this node up to the root.

        Args:
            value (float): The value to backpropagate, typically the outcome of the game.
        """
        node = self
        while node:
            node.visit_count += 1
            node.value_sum   += value
            node = node.parent
            
    def to_dict(self):
        """ Convert the node to a dictionary representation.

        Returns:
            dict: A dictionary containing the node's action, visit count, value sum, and children.
        """
        return {
            'action'  : self.action,
            'visits'  : self.visit_count,
            'wins'    : self.value_sum,
            'children': [c.to_dict() for c in self.children]
        }

class NeuralMCTS:
    """Neural-guided Monte Carlo Tree Search (MCTS) for Battleship.
    """
    def __init__(self, model_path, iters=200, c_puct=1.0,
                 alpha_noise=0.3, eps_noise=0.25, device='cpu'):
        """ Initializes the NeuralMCTS with a pre-trained model.

        Args:
            model_path (str): Path to the pre-trained PyTorch model.
            iters (int): Number of MCTS iterations to perform.
            c_puct (float): Exploration constant for balancing exploration and exploitation.
            alpha_noise (float): Dirichlet noise parameter for root node exploration.
            eps_noise (float): Epsilon for noise injection in root node children.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        self.device = device
        self.model = GameNet().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.iters = iters
        self.c_puct  = c_puct
        self.alpha_noise = alpha_noise
        self.eps_noise = eps_noise
        self.root = None 

    def _evaluate(self, board: Board):
        """ Evaluate the board state using the neural network.

        Args:
            board (Board): The current game state represented as a Board object.

        Returns:
            tuple: A tuple containing:
                - priors (list): List of prior probabilities for each possible move.
                - value (float): Value estimate of the board state.
        """
        tensor = torch.tensor(board.to_tensor(), dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model(tensor)
            priors = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
            return priors, value.item()

    def run(self, root_board: Board):
        """ Run the MCTS algorithm on the given root board.

        Args:
            root_board (Board): The initial game state to start the MCTS from.

        Returns:
            tuple: The action (i, j) to take based on the MCTS results.
        """
        # Create root node and evaluate
        root = NNode(copy.deepcopy(root_board), parent=None, action=None)
        priors, value = self._evaluate(root.state)
        root.prior = 0.0
        root.visit_count = 1
        root.value_sum = value

        # Store full tree for external serialization
        self.root = root

        # Inject noise into root children
        root.expand(priors)
        noise = np.random.dirichlet([self.alpha_noise] * len(root.children))
        for child, n in zip(root.children, noise):
            child.prior = child.prior * (1 - self.eps_noise) + n * self.eps_noise

        # Decide dynamic iteration count
        remaining_ship_cells = sum(
            1 for row in root_board.board for c in row if c.isdigit()
        )
        sims = self.iters * (2 if remaining_ship_cells <= 4 else 1)

        # MCTS main loop
        for _ in range(sims):
            node = root
            # selection
            while node.children:
                node = max(node.children, key=lambda n: n.score(self.c_puct))
            # expansion and evaluation
            if not node.state.has_won():
                priors_leaf, value_leaf = self._evaluate(node.state)
                node.expand(priors_leaf)
                node.backpropagate(value_leaf)
            else:
                # terminal state
                terminal_value = 1.0 if node.state.has_won() else -1.0
                node.backpropagate(terminal_value)

        # Choose the action with highest visit count
        best_child = max(root.children, key=lambda n: n.visit_count)
        return best_child.action
