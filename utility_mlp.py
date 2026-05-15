"""
UtilityMLP for Scoring-Based KV Cache Pruning
From the paper: "Scoring-Based KV Cache Pruning with Learned Utility and Temporal Decay"
Author: Stanislav Usychenko, May 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UtilityMLP(nn.Module):
    """
    Small MLP that predicts token utility score w_j ∈ [0,1]
    Input: key embedding + position + local neighborhood mean
    """
    def __init__(self, d_key: int = 4096, hidden_dim: int = 128, local_window: int = 5):
        super().__init__()
        self.local_window = local_window
        input_dim = d_key + 1 + d_key  # key + pos + neighbor_mean
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, key: torch.Tensor, pos: torch.Tensor, neighbor_mean: torch.Tensor) -> torch.Tensor:
        """
        key: [batch, d_key] or [d_key]
        pos: [batch] or scalar
        neighbor_mean: [batch, d_key]
        """
        if key.dim() == 1:
            key = key.unsqueeze(0)
            pos = pos.unsqueeze(0)
            neighbor_mean = neighbor_mean.unsqueeze(0)
        
        pos = pos.unsqueeze(-1)  # [batch, 1]
        x = torch.cat([key, pos, neighbor_mean], dim=-1)  # [batch, 2*d_key + 1]
        
        hidden = F.relu(self.fc1(x))
        w = torch.sigmoid(self.fc2(hidden))  # utility score [0,1]
        return w.squeeze(-1)