# KV Cache Pruner

**Scoring-Based KV Cache Pruning with Learned Utility and Temporal Decay**

Implementation of the method proposed in the paper:

> **Scoring-Based KV Cache Pruning with Learned Utility and Temporal Decay**  
> Stanislav Usychenko, May 2026

---

## Overview

Lightweight and efficient KV cache eviction policy for long-context Transformer inference.

Instead of using expensive attention-based methods (like H2O), we train a tiny MLP (\~5k parameters) to predict **static utility score** `wⱼ` for each token based on its key embedding, position, and local context. This score is combined with temporal decay to compute **viability score**:

> **vⱼ = wⱼ² / (aⱼ + ε)**

Tokens with the lowest viability are evicted to keep cache size under a fixed budget.

## Features

- **No attention weights** required during inference (unlike H2O)
- Very low overhead (\~1.5ms per step on A100)
- Simple and easy to integrate into any Transformer decoder
- Competitive performance vs H2O, StreamingLLM and Sliding Window
- Up to **8× memory reduction** with minimal quality degradation

## Installation

```bash
git clone https://github.com/Apuoxo/kv-cache-pruner.git
cd kv-cache-pruner
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt