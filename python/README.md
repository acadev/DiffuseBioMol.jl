# PyTorch training baseline

This is the primary development implementation: a Pairformer-lite/DiT flow-matching backbone,
with CPU/CUDA/MPS device selection, residue-complete sequence crops, polymer
prior, centering/rotation augmentation, source-disjoint validation, checkpoint
resume and per-batch timings. CPU has been exercised on real structures.
See the [native baseline results](../docs/BASELINE.md).

Run commands from the repository root. Python 3.12 was used for validation.

```sh
python3.12 -m venv python/.venv
python/.venv/bin/python -m pip install -e .

# Native AtomWorks parsing/tokenization; choose a NEW corpus directory.
python/.venv/bin/python -m diffusebiomol.prepare_corpus \
  /path/to/structures runs/python-corpus

# The trainer itself runs entirely in Python. Choose a NEW run directory.
python/.venv/bin/python -m diffusebiomol.train \
  runs/python-corpus runs/python-experiment --epochs 10 --max-atoms 128 --batch-size 2

# Resume with the same data, device, seed and hyperparameters; increase target epochs.
python/.venv/bin/python -m diffusebiomol.train \
  runs/python-corpus runs/python-experiment --epochs 20 --max-atoms 128 --batch-size 2 --resume
```

The input may be a local file or a recursively scanned directory of `.pdb`, `.ent`,
`.cif`, `.mmcif`, or their `.gz` variants. Parsing, tokenization, and training are
entirely Python. Preparation uses
[AtomWorks 3.0](https://rosettacommons.github.io/atomworks/latest/) with its minimal
parser preset and Biotite's bundled CCD. No external CCD/PDB mirror is required.

The default selects the chain with the most canonical polymer residues (ties
break by chain ID). Use `--chain all` for the asymmetric unit, or `--chain A` for
one exact parser chain ID. Only the first model and first alternate conformer
are used; biological assemblies are not expanded. Waters and hydrogens are
removed. Canonical protein/RNA/DNA residues have fixed atom slots; missing,
nonfinite, unresolved, or zero-occupancy coordinates are masked and stored as
zero placeholders, never synthesized. Noncanonical polymer residues/PTMs and
ligands retain their heavy atoms with the catch-all polymer embedding. Common
single-atom metals/halides are classified as ions. This is a baseline component
classification policy, not a comprehensive chemistry ontology.

`diffusebiomol/vocabulary.json` freezes all 334 canonical atom IDs from the
baseline vocabulary plus an unknown bucket. It is a checked-in data asset,
not generated at runtime. Parser chain IDs are recoded in encounter order;
residue indices follow parser numbering, preserving gaps and separating insertion
codes. AtomWorks may use mmCIF label numbering/chain IDs rather than author IDs.
The tensor schema and vocabulary remain compatible with old exported corpora,
but selection/cleanup can change their contents: for example, the native 5PTI
largest-chain crop excludes the hetero components previously attached to the
author chain. Prepare a new corpus and start a new training run when switching;
checkpoint resume correctly rejects a changed corpus manifest.

Preparation writes checksummed, linear-size source records and parser/vocabulary
metadata to `manifest.json`. Failed files are listed with reasons; if every file
fails, diagnostics are still written and the command exits with an error. JSON
per source is a baseline interchange format, not the final sharded storage design.

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

## Tests

```sh
# Parser/tokenizer, invariant and exact-resume tests using offline fixtures:
python/.venv/bin/python -m unittest discover -s python/tests -v

# Repeat learning/resume/crop tests using a prepared real corpus:
DBM_CORPUS=runs/python-corpus python/.venv/bin/python \
  -m unittest discover -s python/tests -v

```

The native AtomWorks path was validated on six local baseline structures on
2026-10-07: all parsed successfully. Ten Python tests passed, including format
and compressed-file parsing, first-model/alternate-conformer selection, missing
atom masks, mixed modalities, insertion codes, crop integrity, and exact CPU
checkpoint resume on the newly prepared corpus. A fixed real crop's loss fell
from 100.075775 to 0.053838 over 150 updates; this checks learning mechanics,
not protein quality or held-out generalization.

## Scope

Implemented: the existing linear-path **flow matching**, Euler sampler, dense
pair-biased backbone and unconditional crop-training path. Categorical embeddings
retain the frozen baseline vocabulary. The backbone accepts four conditioning
features, but motif clamping, CFG, geometry guidance, verifier training and
external verification are not implemented.

Diffusion remains an explicit future objective/schedule/sampler, not an alias for
flow matching. There is no full-protein quality
claim, distributed trainer, mixed precision or asynchronous loader yet. The dense
pair representation remains quadratic even when PyTorch uses fused attention.
Neither increasing crop presentations nor stitching local outputs establishes
global protein topology; full-structure generation needs a separate design/gate.

The implementation uses [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
and tensor/state-dict [serialization](https://docs.pytorch.org/docs/2.14/notes/serialization.html).
