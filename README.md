# DiffuseBioMol

A native Python project for all-atom biomolecular flow matching, using
AtomWorks for structure parsing and PyTorch for modeling and training.

## Install

Run from the repository root with Python 3.12:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Dependency versions and command-line entry points are defined in `pyproject.toml`.
An existing Python environment can be reused with `python -m pip install -e .`.

## Prepare and train

```sh
# Parse local PDB/mmCIF files into a new corpus directory.
diffusebiomol-prepare /path/to/structures runs/corpus

# Train on residue-complete crops; use a new run directory.
diffusebiomol-train runs/corpus runs/experiment --epochs 10 --max-atoms 128 --batch-size 2

# Continue the same experiment.
diffusebiomol-train runs/corpus runs/experiment --epochs 20 --max-atoms 128 --batch-size 2 --resume
```

The equivalent module commands are `python -m diffusebiomol.prepare_corpus` and
`python -m diffusebiomol.train`. No `PYTHONPATH` setup is needed after installation.
Use `--help` for options and [the workflow guide](python/README.md) for parsing
policies, configuration, outputs, and checkpoint compatibility.

## Current capabilities

- PDB/mmCIF and gzip parsing with AtomWorks; versioned atom tokenization for
  proteins, RNA, DNA, ligands, ions, and modified residues.
- Pairformer-lite/DiT backbone, linear-path flow-matching objective, polymer
  prior, rotation/centering augmentation, and Euler sampling.
- Single-device FP32 training, source-disjoint validation, checksummed corpora,
  exact tested CPU checkpoint resume, and diagnostic stage timings.
- CPU, CUDA, and MPS device selection. CPU is validated; accelerator training
  and throughput still require validation on target hardware.

The current runner trains unconditional crops. Diffusion, motif clamping,
classifier-free guidance, geometry guidance, verifier training, distributed
training, and full-structure generation are future work. Tokenizer support for
multiple modalities does not establish generation quality for those modalities.
Dense pair features still require quadratic memory.

## Test

```sh
python -m unittest discover -s python/tests -v
# Optional real-data learning and resume checks:
DBM_CORPUS=runs/corpus python -m unittest discover -s python/tests -v
```

[Baseline evidence](docs/BASELINE.md) describes the six-source CPU validation.
The [scaling plan](docs/SCALING_CLEANUP_PLAN.md) separates current functionality
from the remaining scientific and engineering work. GitHub Actions runs the
Python tests and command-line smoke checks.

## Layout

```text
pyproject.toml              Package metadata, dependencies, and CLI commands
python/diffusebiomol/        Parser, tokenizer, corpus, model, objective, trainer
python/tests/               Offline parser and training regression tests
python/configs/small.json    Example larger backbone configuration
docs/                       Baseline evidence and roadmap
```

The versioned vocabulary is bundled in the installable package. Datasets,
checkpoints, and run outputs belong under ignored `runs/` directories.
