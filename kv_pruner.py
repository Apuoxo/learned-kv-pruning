"""
KV Cache Pruner with Learned Utility + Temporal Decay
Paper: Scoring-Based KV Cache Pruning with Learned Utility and Temporal Decay
"""

import torch
import heapq
from typing import List, Tuple, Optional
from utility_mlp import UtilityMLP


class KVPruner:
    def __init__(self, 
                 budget: int = 1024,
                 epsilon: float = 1e-8,
                 device: str = "cuda"):
        
        self.budget = budget
        self.epsilon = epsilon
        self.device = device
        
        # Min-heap: (viability_score, counter, key, value, utility, age)
        self.heap = []
        self.counter = 0
        
    def compute_viability(self, utility: float, age: int) -> float:
        """v_j = w_j² / (a_j + ε)"""
        return (utility ** 2) / (age + self.epsilon)
    
    def add_token(self, 
                  key: torch.Tensor, 
                  value: torch.Tensor, 
                  utility: float, 
                  age: int = 0):
        """Add new token to cache with eviction if needed"""
        v = self.compute_viability(utility, age)
        
        item = (v, self.counter, key.clone(), value.clone(), utility, age)
        self.counter += 1
        
        if len(self.heap) < self.budget:
            heapq.heappush(self.heap, item)
        else:
            # Replace worst token if new one is better
            if v > self.heap[0][0]:
                heapq.heapreplace(self.heap, item)
    
    def get_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return current KV cache sorted by insertion order"""
        # Sort by counter to preserve relative order
        sorted_items = sorted(self.heap, key=lambda x: x[1])
        keys = torch.stack([item[2] for item in sorted_items])
        values = torch.stack([item[3] for item in sorted_items])
        return keys, values
    
    def update_ages(self):
        """Increase age of all tokens by 1 (call every step)"""
        new_heap = []
        for v, counter, key, value, util, age in self.heap:
            new_heap.append((v, counter, key, value, util, age + 1))
        self.heap = new_heap  # heapq doesn't support easy in-place update, so we rebuild


# ====================== EXAMPLE USAGE ======================
if __name__ == "__main__":
    print("KV Cache Pruner initialized")
    pruner = KVPruner(budget=1024, device="cpu")
    
    # Example: dummy key/value
    dummy_key = torch.randn(4096)
    dummy_value = torch.randn(4096)
    dummy_utility = 0.85
    
    pruner.add_token(dummy_key, dummy_value, dummy_utility)
    print(f"Cache size: {len(pruner.heap)} / {pruner.budget}")