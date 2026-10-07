# PyTorch baseline results — 2026-09-18

The Python training path is runnable alongside Julia. The accepted local run is
`runs/python-baseline-20260918/`; see `python/README.md` for setup and commands.

## Execution

- Apple M4 CPU, Python 3.12, PyTorch 2.14.0, NumPy 2.5.3, two PyTorch threads.
- Six real PDB structures exported through the Julia tokenizer, with four
  training sources and two held-out sources. Source-disjoint, not clustered.
- One Pairformer/one DiT block, single width 16, pair width 8, four heads.
- Residue-complete sequence crops capped at 128 tokens, batch size two.
- Three epochs followed by CLI resume to epoch five: 20 training presentations,
  10 optimizer updates, finite validation loss and finite sampled coordinates.
- Final fixed-validation CFM loss: 174.09755. This tiny run does not establish
  useful held-out improvement or protein quality.

## Checks

Four Python test cases passed against the real corpus, covering fixed-example
learning, masked loss normalization, padding/output/gradient invariance,
residue-complete cropping, checksum rejection, resume-config rejection and exact
CPU resume equality for model, optimizer, RNG and counters. The same test suite
also passed using generated offline fixtures during development.

The controlled 150-update real-crop overfit reduced loss from 100.075775 to
0.057668. This demonstrates learnability, not generalization.

The full-network numerical fixture uses identical Julia/Python weights and
inputs, including nonzero conditioning and a nonzero output head. Maximum
absolute errors after the normalization fix:

| Quantity | Maximum absolute error |
|---|---:|
| Forward output | 3.58e-7 |
| Squared-output loss | 1.19e-7 |
| Coordinate projection weight gradient | 2.24e-8 |
| Output head weight gradient | 3.58e-7 |

The parity check is a Float32 network/selected-gradient check, not a claim that
independently seeded training trajectories or every optimizer update match
between frameworks. CFM normalization/masking is tested independently.

## Julia bug found and corrected

Lux's default `LayerNorm(...; dims=:)` normalized the flattened feature/token/batch
array globally. The intended behavior is per-token feature normalization. All
five LayerNorm constructions in `src/Model/Network.jl` now specify `dims=1`.

The original padding test used the zero-initialized output head, so its outputs
were identically zero regardless of hidden-state leakage. It now randomizes that
head. All 18 Julia batching checks and all five Julia model-learning smoke checks
passed after the correction.

This changes model behavior for existing Julia checkpoints despite unchanged
weight shapes. Re-establish the baseline rather than treating old checkpoints as
numerically interchangeable with the corrected implementation.

## Performance limits and next gate

Warmed CPU forward/backward/optimizer stages for these tiny batches total roughly
9–10 ms; loading/preparation adds roughly 3 ms. Epoch training timings exclude
imports, initialization, checkpoint serialization and evaluation. They are not a
like-for-like speed comparison against the earlier Julia run, whose normalization,
random draws and evaluation workload differ.

The Python path is FP32, single-device, unconditional flow matching with Euler
sampling. CUDA/MPS selection exists but was not exercised. Geometry guidance,
motif clamping, verifier training, diffusion, mixed precision, distributed training,
full-structure generation and production sharded data loading remain unimplemented
in Python. No existing Julia capabilities were removed.

Next: reproduce this baseline on the target GPU, then increase crop lengths and
corpus size while tracking memory, padding, data wait, quality and cumulative
presentations. Diffusion requires its own objective/noise schedule/sampler and tests.
