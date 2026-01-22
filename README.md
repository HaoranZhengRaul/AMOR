# When to Think Fast and Slow? AMOR: Entropy-Based Metacognitive Gate for Dynamic SSM-Attention Switching

Replication package for the preprint:

> **When to Think Fast and Slow? AMOR: Entropy-Based Metacognitive Gate for Dynamic SSM-Attention Switching**

## Overview

AMOR is a hybrid architecture combining State Space Models (SSMs) with selective attention, inspired by dual-process theories of cognition (Kahneman, 2011). The SSM backbone processes all positions, while sparse attention is dynamically engaged only when the model is "uncertain"—as measured by prediction entropy. When entropy is low, local processing suffices; when high, attention retrieves from the Ghost KV cache.

**Key Results:**
- **Entropy gap of 1.09 nats** (nearly half the entropy range) between retrieval and local positions validates entropy as a routing signal
- **100% retrieval accuracy** on Simple Task while engaging attention on only 22% of positions (78% FLOPs saved)
- **Ghost KV** (projecting K,V from SSM hidden states) achieves 6× improvement over raw embeddings (36% vs 6%)
- **Outperforms both SSM-only and transformer-only baselines**
- AMOR Oracle achieves **8.4× improvement** over Full Attention on NeedleHaystack (37% vs 4%)

## Repository Structure

```
├── src/
│   ├── models/
│   │   ├── amor_v2.py          # Main AMOR architecture
│   │   ├── ssm.py              # SSM backbone (GRU-based)
│   │   ├── gate.py             # Entropy-based metacognitive gate
│   │   ├── ghost_kv.py         # Ghost KV cache implementation
│   │   └── attention.py        # Sparse top-k attention
│   ├── data/
│   │   ├── retrieval_task.py   # Simple Retrieval Task
│   │   └── needle_haystack.py  # NeedleHaystack Task
│   ├── training/
│   │   └── supervised.py       # Training utilities
│   └── analysis/
│       └── entropy_verification.py
├── scripts/                    # Experiment scripts
├── results/                    # Pre-computed results (seed=42)
└── README.md
```

## Requirements

```
Python 3.11
PyTorch 2.1+
numpy
```

Install:
```bash
pip install torch numpy
```

**Note:** Training uses CPU by default. MPS (Apple Silicon) has numerical issues with GRU operations.

## Reproducing Results

All experiments use seed=42 for reproducibility.

### 1. Entropy Gap Validation (Table 1)
```bash
python scripts/verify_entropy.py --seed 42
```
Expected: Entropy gap ~1.09 nats

### 2. Simple Retrieval Task (Table 2)
```bash
python scripts/run_baselines.py --seed 42
```
Expected results:
| Model | Retrieval Acc | Gate Fires | FLOPs Saved |
|-------|---------------|------------|-------------|
| SSM Only | 68.35% | — | 100% |
| Full Attention | 87.30% | — | 0% |
| AMOR Oracle | 99.63% | 2.90% | 97.10% |
| AMOR Entropy | **100%** | 22.32% | **77.68%** |

### 3. NeedleHaystack Task (Table 3)
```bash
python scripts/run_needlehaystack_seeded.py
```
Expected results:
| Model | Retrieval Acc |
|-------|---------------|
| SSM Only | 8.02% |
| Full Attention | 4.40% |
| AMOR Entropy | 9.93% |
| AMOR Oracle | **37.08%** |

### 4. Ghost KV vs Raw Embedding (Table 4)
```bash
python scripts/compare_kv_methods_seeded.py
```
Expected: Ghost KV 36.28% vs Raw Embedding 6.08%

### 5. Ablations & Sweeps
```bash
python scripts/run_ablations.py --seed 42
python scripts/sweep_noise_length.py
```

### 6. Generate Figures
```bash
python scripts/generate_paper_figures.py
```

## Key Implementation Details

### Entropy-Based Gate
```python
# Normalized entropy for scale-invariance
entropy = raw_entropy / log(vocab_size)  # [0, 1]

# Learnable threshold with STE for hard decisions
gate_prob = sigmoid(scale * (entropy - threshold))
gate = (gate_prob > 0.5).float()  # hard decision, soft gradients
```

### Ghost KV Cache
Projects K,V from SSM hidden states (not raw embeddings):
```python
K = W_k @ H  # H = SSM hidden states, already O(n) computed
V = W_v @ H  # Reuses contextualized representations
```

### Balance Loss
Prevents gate collapse to always-on or always-off:
```python
L = L_CE + lambda * (mean_gate_rate - target_rate)^2
```

## Results Summary

| Experiment | Key Finding |
|------------|-------------|
| Entropy Gap | 1.09 nats (nearly half the entropy range) |
| Simple Task | 100% retrieval acc, 78% FLOPs saved |
| NeedleHaystack | Oracle 37% vs Full Attention 4% (8.4× improvement) |
| Ghost KV | 6× improvement over raw embeddings |
| Sparse Attention | k=3 >> k=16 (concentration beats dilution) |
| SSM Horizon | ~50 tokens before significant state decay |

## Citation

```bibtex
@misc{amor2026,
  title={When to Think Fast and Slow? AMOR: Entropy-Based Metacognitive Gate for Dynamic SSM-Attention Switching},
  author={[Authors]},
  year={2026},
  howpublished={Preprint}
}
```

## License

MIT License
