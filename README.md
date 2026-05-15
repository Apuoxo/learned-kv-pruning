# KV Cache Pruner

**Scoring-Based KV Cache Pruning with Learned Utility and Temporal Decay**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

> Implementation of the method proposed in:  
> *"Scoring-Based KV Cache Pruning with Learned Utility and Temporal Decay"*  
> Stanislav Usychenko, May 2026

---

## TL;DR

Lightweight KV cache eviction policy for long-context Transformer inference.

**Instead of** expensive attention-based methods (like H2O), we train a tiny MLP (~5k params) to predict static utility score `wⱼ` for each token based on its key embedding, position, and local context.

**Viability score formula:**  
`vⱼ = wⱼ² / (aⱼ + ε)`

Tokens with lowest viability are evicted to keep cache under budget.

### Key Advantages
- ✅ **No attention weights needed** during inference (unlike H2O)
- ✅ Very low overhead (~1.5ms per step on A100)
- ✅ Simple integration into any Transformer decoder
- ✅ Up to **8× memory reduction** with minimal quality degradation

---

## Installation

```bash
git clone https://github.com/Apuoxo/learned-kv-pruning.git
cd learned-kv-pruning
pip install torch>=2.0.0
pip install -r requirements.txt
```

---

## Quick Start

```python
from kv_pruner import KVCachePruner

# Create pruner with budget of 2048 tokens
pruner = KVCachePruner(
    budget=2048,
    epsilon=1e-6,
    device="cuda"  # or "cpu"
)

# During generation loop after each step:
viable_scores = pruner.compute_viability(key_states, age)
pruned_kv = pruner.prune(kv_cache, viable_scores, budget)
```

### Integration Example

```python
# Example hook in your Transformer block's forward pass
class AttentionWithPruning(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.pruner = KVCachePruner(budget=2048)
        self.kv_cache = None
    
    def forward(self, query, key, value, ...):
        # Standard attention computation
        attn_output = F.scaled_dot_product_attention(query, key, value)
        
        # Update cache with pruning
        self.kv_cache = self.pruner.update_and_prune(
            key, value, self.kv_cache
        )
        return attn_output
```

---

## Results (Preliminary)

| Method | Compression | Accuracy Drop (LongBench) | Overhead |
|--------|-------------|---------------------------|----------|
| H2O | 8× | -2.1% | ~15 ms/step |
| **Ours** | 8× | **-1.8%** | **~1.5 ms/step** |
| StreamingLLM | 8× | -4.3% | ~0.5 ms/step |

---

## Repository Structure

```
learned-kv-pruning/
├── kv_pruner.py        # Main pruning logic
├── utility_mlp.py      # MLP model for utility prediction
├── requirements.txt    # Dependencies
└── README.md           # This file
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{usychenko2026scoring,
  title={Scoring-Based KV Cache Pruning with Learned Utility and Temporal Decay},
  author={Usychenko, Stanislav},
  year={2026},
  month={may}
}
```

---

## License

MIT © 2026 Apuoxo