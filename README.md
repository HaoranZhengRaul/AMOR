# AMOR

**Adaptive Metacognitive Output Router** — a post-hoc hybrid architecture that
gates causal self-attention by output prediction entropy.

A recurrent backbone (Mamba-2 or Gated DeltaNet) is followed by `K` stacked
AMOR blocks. Each block measures the entropy of `softmax(lm_head(h))` and, where
the backbone is uncertain, fires self-attention to refine the residual stream.

```
Input -> [Backbone Mixer + SwiGLU MLP] x N -> [AMOR Block] x K -> Norm -> LM Head
                                                    |
                                            Entropy Gate  (fire if uncertain)
                                                    |
                                                Attention  (Q, K, V from residual)
```

K, V are standard linear projections of the residual stream — not SSM internal
state. The entropy gate concentrates attention compute on positions where the
backbone is least confident.

This repository is a single-file release that subsumes the six published
architecture variants behind three flags.

---

## Install

```bash
pip install -r requirements.txt
```

`mamba-ssm` and `causal-conv1d` are needed for the Mamba-2 backbone;
`flash-linear-attention` is needed for the Gated DeltaNet backbone. Both
backbones require a CUDA-enabled GPU for the fast Triton kernels; CPU-only
fallbacks exist but are slow.

---

## Quickstart

```python
import torch
from amor import AMOR
from amor_decode import AMORDecodeWrapper

model = AMOR(
    backbone='mamba2',          # 'mamba2' or 'gdn'
    n_amor_blocks=3,            # 1 or 3
    residual_mode='classic',    # 'classic' or 'h'
    vocab_size=128256,
    d_model=768, n_layer=12, d_ff=1216,
    n_heads=12, head_dim=64,
).cuda().eval()

# Standard forward
input_ids = torch.randint(0, 128256, (1, 1024), device='cuda')
logits = model(input_ids)                              # [1, 1024, 128256]
logits, details = model(input_ids, return_details=True)
print(details['fire_rate'], details['alpha'])

# Autoregressive generation with caches
wrapper = AMORDecodeWrapper(model)
logits, cache = wrapper.prefill(input_ids)
next_token = logits[:, -1].argmax(-1, keepdim=True)
for step in range(64):
    step_idx = input_ids.shape[1] + step
    logits, cache, fire_info = wrapper.decode_step(next_token, cache, step_idx)
    next_token = logits[:, -1].argmax(-1, keepdim=True)
```

A runnable end-to-end demo (tiny model, CPU-friendly) is in `example.py`:

```bash
python example.py
```

---

## Variants

The six variants are flag combinations of a single class:

| Configuration | `backbone` | `n_amor_blocks` | `residual_mode` |
|---|---|---|---|
| AMOR              | `mamba2` | `3` | `classic` |
| AMOR-classic      | `mamba2` | `1` | `classic` |
| AMOR-h            | `mamba2` | `3` | `h`       |
| AMOR-GDN          | `gdn`    | `3` | `classic` |
| AMOR-classic-GDN  | `gdn`    | `1` | `classic` |
| AMOR-GDN-h        | `gdn`    | `3` | `h`       |

### Flag reference

- **`backbone`** — recurrent sequence mixer.
  - `'mamba2'`: Mamba-2 SSD via `transformers.Mamba2ForCausalLM`.
  - `'gdn'`: Gated DeltaNet via `fla.layers.GatedDeltaNet`.

- **`n_amor_blocks`** — depth of the AMOR stack on top of the backbone (1 or 3 in
  the published configurations; any positive integer is accepted).

- **`residual_mode`** — placement of the post-backbone norm:
  - `'classic'`: `norm_f` is applied to the residual stream once, before the AMOR
    stack. Block 0 has no extra pre-norm; blocks 1+ each carry their own.
  - `'h'`: `norm_f` is reused inline as block 0's pre-norm. The residual stream
    stays un-normed through every AMOR block; only `final_norm` touches it
    before `lm_head`. All blocks become symmetric.

  Both modes have identical state_dict structure for a given `(backbone,
  n_amor_blocks)` pair, so checkpoints can be reloaded under either mode if
  desired.

---

## Reference dimensions

The published variants use these dimensions:

| Scale | `d_model` | `n_layer` | `d_ff` | `n_heads` | `head_dim` |
|-------|-----------|-----------|--------|-----------|------------|
| 180M  | 768       | 12        | 1216   | 12        | 64 |
| 440M  | 1024      | 24        | 1984   | 16        | 64 |
| 1.5B  | 2048      | 24        | 4096   | 32        | 64 |

Mamba-2 uses `d_state=128`, `expand=2`, `n_groups=1` and computes its own SSM
head count from `(d_model * expand) // head_dim`. GDN uses `expand_v=2.0` and
the `fla` convention `num_heads = int(0.75 * d_model / head_dim)`.

The default tokenizer assumed in the configurations above is a
Llama-3.1-compatible BPE with vocabulary 128256.

---

## Architecture details

### Entropy gate

```
entropy   = -sum(p * log(p)) / log(vocab_size)         # in [0, 1]
threshold = EMA(median) + offset_k * EMA(std)          # adaptive (default)
gate      = (entropy > threshold)                      # hard binary
```

The threshold tracks an EMA of the batch entropy median (and standard
deviation, in adaptive mode), so the gate has no learnable parameters and the
entropy signal does not flow gradients into the backbone — gating decisions are
detached. The CE loss trains the backbone exclusively through the final logits
path, never through the gate.

### AMOR block

```
gate     = EntropyGate(lm_head(h_normed))
attn_out = Attention(h_normed) * gate
output   = h_normed + alpha * attn_out
```

- Q/K/V are linear projections of the (pre-normed) residual stream. RoPE is
  applied to Q and K.
- `out_proj` is zero-initialized; the per-channel `alpha = sigmoid(raw_alpha)`
  (clamped from below at `raw_alpha = -1`, giving a floor of ~0.27) handles
  output magnitude scaling. At step 0 the AMOR stack is a no-op.
- Two attention modes:
  - `'full_with_mask'`: dense attention, output masked by the gate. Used for
    training; GPU-friendly.
  - `'true_sparse'`: Q and the attention computation only run for firing
    positions. Bit-identical output for the same weights/inputs. Used for
    inference.

### Stacking

When `n_amor_blocks > 1`, each block recomputes its own entropy from
`lm_head(pre_norm(h))`. After the first block fires attention and modifies the
residual stream, the second block sees a different entropy landscape: positions
that were resolved drop in entropy, while positions that remain hard stay high.
Each block has its own EntropyGate (independent EMA), Q/K/V/O, RoPE, and alpha.

### Initialization

All weights are initialized to `N(0, 0.02)` for apple-to-apple comparison
between backbones. AMOR `out_proj` stays at zero. The LM head is weight-tied to
the embedding.

---

## Files

| File | Purpose |
|------|---------|
| `amor.py`         | Unified model (building blocks + GDN block + `AMOR` class) |
| `amor_decode.py`  | Autoregressive decode wrapper (prefill + decode_step, with caches) |
| `example.py`      | Tiny end-to-end demo |
| `requirements.txt`| Dependencies |
| `LICENSE`         | MIT |

The model and decode wrapper are self-contained: no project-internal imports.

---

## License

MIT — see [LICENSE](LICENSE).
