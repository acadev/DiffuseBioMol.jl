# PyTorch training baseline

This is a runnable port of the Julia Pairformer-lite/DiT flow-matching backbone,
with CPU/CUDA/MPS device selection, residue-complete sequence crops, polymer
prior, centering/rotation augmentation, source-disjoint validation, checkpoint
resume and per-batch timings. CPU has been exercised on real structures.
See the [measured results](../docs/PYTHON_BASELINE_20260918.md), including the
Julia/PyTorch forward and gradient comparison.

Run commands from the repository root. Python 3.12 was used for validation.

```sh
python3.12 -m venv python/.venv
python/.venv/bin/python -m pip install -r python/requirements.txt

# One-time export using the existing Julia parser/tokenizer; choose a NEW directory.
julia --project=. scripts/export_python_corpus.jl runs/baseline-smoke-data runs/python-corpus

# The trainer itself runs entirely in Python. Choose a NEW run directory.
PYTHONPATH=python python/.venv/bin/python -m diffusebiomol.train \
  runs/python-corpus runs/python-experiment --epochs 10 --max-atoms 128 --batch-size 2

# Resume with the same data, device, seed and hyperparameters; increase target epochs.
PYTHONPATH=python python/.venv/bin/python -m diffusebiomol.train \
  runs/python-corpus runs/python-experiment --epochs 20 --max-atoms 128 --batch-size 2 --resume
```

`runs/baseline-smoke-data` contains the six structures staged for the Julia
baseline. For another dataset, replace this path with a local PDB/mmCIF directory.
The exporter selects the largest chain, as the Julia baseline does. It exports
linear-size token records, explicit vocabulary metadata and checksums. It does
not persist quadratic pair tensors. Julia is needed only for this initial export.
JSON-per-source is a baseline interchange format, not the final billion-sample
sharded storage design. Export failure details appear in `manifest.json`.

For larger experiments, pass `--model-config python/configs/small.json`, adjust
`--max-atoms`/`--batch-size`, and select `--device cuda` (or `cuda:1`). On Apple
hardware, `--device mps` selects Metal. Requested unavailable devices fail rather
than silently falling back. CUDA and MPS training have not yet been validated.
Resume currently requires the same device and thread settings as the checkpoint.

## Outputs and interpretation

- `manifest.json`: exact source split, corpus identity, model/configuration and
  software versions. The split is source-disjoint, not sequence-clustered.
- `steps.csv`: cumulative crop presentations, real/padded atoms, loss, host
  loading/preparation, transfer, forward, backward/optimizer timings.
- `checkpoint.pt`: model, optimizer, objective/configuration, RNG states, epoch,
  presentation and update counts. Saved atomically before evaluation each epoch.
- `epoch_*.json` and `summary.json`: fixed-validation CFM loss and a finite
  three-step sample check. There are no geometry/designability pass claims.

Resume is exact on tested CPU runs at epoch boundaries. An interrupted epoch is
replayed; its uncommitted timing rows are discarded. Checkpoint/corpus mismatch is
rejected. Training examples use fresh crop/prior/rotation/time draws; validation
uses a separate fixed RNG and does not advance training randomness.

Timing boundaries explicitly synchronize accelerators and gradient-finiteness
checks may synchronize too. These are diagnostic baseline measurements, not a
fully overlapped high-throughput trainer. First CUDA initialization, imports and
environment startup are outside the per-epoch training timings.

## Tests and Julia agreement

```sh
# Offline invariant and exact-resume tests using synthetic records:
PYTHONPATH=python python/.venv/bin/python -m unittest discover -s python/tests -v

# Repeat learning/resume/crop tests using the real exported sources:
DBM_CORPUS=runs/python-corpus PYTHONPATH=python python/.venv/bin/python \
  -m unittest discover -s python/tests -v

# Identical-weight full-network forward/loss and selected-gradient comparison:
julia --project=. scripts/export_python_parity.jl runs/python-parity.json
PYTHONPATH=python python/.venv/bin/python -m diffusebiomol.parity runs/python-parity.json
```

The parity fixture uses a nonzero output head and nonzero conditioning so it
exercises the encoder, decoder and backward path. It compares Float32 outputs,
squared-output loss and coordinate/head weight gradients. This does not establish
identical optimizer trajectories across frameworks: initialization, RNG sequences
and low-level implementations differ. Padding/gradient invariance and masked CFM
loss normalization are tested separately.

The parity exercise exposed a Julia LayerNorm axis bug: Lux's default normalized
across tokens and batch members. `src/Model/Network.jl` now explicitly uses
`dims=1`, and its padding regression test uses nonzero output weights. Earlier
Julia checkpoints retain compatible tensor shapes but changed normalization
semantics; restart the training baseline when comparing with this implementation.

## Scope

Implemented: the existing linear-path **flow matching**, Euler sampler, dense
pair-biased backbone and unconditional crop-training path. Categorical embeddings
retain the Julia export's vocabulary. The backbone accepts four conditioning
features, but motif clamping, CFG, geometry guidance, verifier training and
external verification have not been ported to this runner.

Diffusion remains an explicit future objective/schedule/sampler, not an alias for
flow matching. No Julia code has been replaced. There is no full-protein quality
claim, distributed trainer, mixed precision or asynchronous loader yet. The dense
pair representation remains quadratic even when PyTorch uses fused attention.
Neither increasing crop presentations nor stitching local outputs establishes
global protein topology; full-structure generation needs a separate design/gate.

The implementation uses [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
and tensor/state-dict [serialization](https://docs.pytorch.org/docs/2.14/notes/serialization.html).
