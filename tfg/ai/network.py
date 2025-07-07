"""
tgf.ai.network
===================
This module defines the neural network architecture used for the game AI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GameNet(nn.Module):
    """GameNet is a neural network for the Battleship game.

    Args:
        nn.Module: Inherits from PyTorch's nn.Module to define a neural network.
    """
    def __init__(self):
        """ Initializes the GameNet architecture.
        """
        super().__init__()
        # conv trunk
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # policy head
        self.p_conv = nn.Conv2d(64, 4, kernel_size=1)
        self.p_fc = nn.Linear(4 * 6 * 6, 36)
        # value head
        self.v_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.v_fc1  = nn.Linear(2 * 6 * 6, 64)
        self.v_fc2  = nn.Linear(64, 1)

    def forward(self, x):
        """Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 6, 6).

        Returns:
            tuple: A tuple containing:
                - logits (torch.Tensor): Output tensor for policy head of shape (batch_size, 36).
                - value (torch.Tensor): Output tensor for value head of shape (batch_size, 1).
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # policy
        p = F.relu(self.p_conv(x))
        p = p.view(-1, 4 * 6 * 6)
        logits = self.p_fc(p)
        # value
        v = F.relu(self.v_conv(x))
        v = v.view(-1, 2 * 6 * 6)
        v = F.relu(self.v_fc1(v))
        value = torch.tanh(self.v_fc2(v)).squeeze(-1)
        
        return logits, value
