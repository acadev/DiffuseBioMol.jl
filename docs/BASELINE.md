# Native Python baseline — 2026-10-07

Environment: Apple M4 CPU, Python 3.12, PyTorch 2.14.0, NumPy 2.5.3,
AtomWorks 3.0.0, two PyTorch threads. No accelerator results are claimed.

## Corpus and validation

Six local structures (1CRN, 1L2Y, 1UBQ, 1VII, 2GB1, 5PTI) were prepared
through `diffusebiomol.prepare_corpus`; all succeeded. The default selects the
largest canonical polymer chain from the first model's asymmetric unit.

Ten tests passed, covering PDB/mmCIF and compressed parsing, alternate conformers,
missing atom masks, insertion codes, mixed modalities, residue-complete cropping,
masked flow loss, padding/output/gradient invariance, corpus integrity,
configuration mismatch rejection, and exact CPU checkpoint resume.

Training checks used 128-token crops, batch size two, four training sources and
two held-out sources, one Pairformer and one DiT block, single width 16,
pair width 8, and four heads. Three uninterrupted epochs matched two epochs
followed by resume to epoch three for model, optimizer, RNG state, and counters.
That trajectory contains 12 crop presentations and six optimizer updates.
Validation loss and sampled coordinates were finite.

A separate fixed real-crop experiment reduced loss from 100.075775 to 0.053838
in 150 updates. It demonstrates optimization can learn one example, not
held-out generalization or useful protein generation. Source-disjoint splitting
is not sequence-clustered splitting.

## Reproduce

Install the project from the repository root, then run:

```sh
diffusebiomol-prepare /path/to/six-local-structures runs/native-corpus
DBM_CORPUS=runs/native-corpus python -m unittest discover -s python/tests -v
```

Use a new corpus directory. The tests write temporary training outputs and
compare complete checkpoint state. Local preparation artifacts from the measured
run are in ignored `runs/atomworks-corpus-20261007/`; they are not shipped with
the repository. Synthetic offline fixtures run without setting `DBM_CORPUS`.

## Limits

The baseline is FP32, single-device, unconditional flow matching with Euler
sampling. It establishes parsing, learning mechanics, and restart correctness.
It does not establish GPU throughput, protein quality, diffusion training,
full-structure generation, or billion-presentation capacity.
