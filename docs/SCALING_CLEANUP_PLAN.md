# Python scaling plan

The implementation is Python, AtomWorks, and PyTorch. Current evidence and
reproduction commands are in [BASELINE.md](BASELINE.md).

## Implemented foundation

Native PDB/mmCIF parsing, versioned tokenization, checksummed source records,
residue-complete sequence cropping, a dense Pairformer-lite/DiT backbone,
linear-path flow matching, Euler sampling, diagnostic timings, and exact tested
CPU resume at epoch boundaries. Checkpoints are saved before validation.

## Next engineering gates

1. Validate training and resume on the target GPU. Measure cold startup,
   warm compute, loading, transfer, evaluation, checkpointing, peak memory,
   real/padded atom throughput, and cumulative sample presentations separately.
2. Establish a larger, sequence-clustered holdout with provenance and quality
   filters. A source split alone does not remove sequence-related leakage.
3. Replace JSON-per-source with incremental, versioned compact shards; keep
   coordinates, categorical identities, masks, and topology linear in size.
   Build quadratic pair features only after cropping. Bound worker prefetch
   and batch memory, preserving deterministic sampling and restart behavior.
4. Evaluate mixed precision and activation checkpointing against FP32 correctness.
   Measure pair-budget batching and padding efficiency before scaling crop length.
5. Add PyTorch distributed data parallel training with rank-aware sampling,
   globally weighted losses, and restart checks. Introduce state sharding only
   when model-state memory justifies it.

Diagnostic synchronization intentionally serializes timing boundaries. It is not
an overlapped production throughput measurement. Report sustained optimizer-step
throughput, evaluation/checkpoint overhead, and loader waits. Fused attention
does not eliminate the explicit dense pair representation's quadratic memory.

## Scientific capabilities still required

- Add diffusion as a distinct objective, noise schedule, prediction target, and
  sampler with independent tests. Record these choices in checkpoint contracts.
- Implement and evaluate motif clamping, classifier-free guidance, and geometry
  guidance. Missing required evaluation measurements must not count as passes.
- Build geometry metrics and calibrated verifier training on real held-out data;
  external refolding/designability checks need explicit budgets and provenance.
- Demonstrate full-structure generation with global topology and long-range
  contacts. Local coordinate patches cannot simply be joined into valid proteins.
  Explore global residue/anchor context with local atom refinement, and compare
  memory and quality across lengths. Cartesian projections with augmentation do
  not by themselves guarantee rotational equivariance.
- Expand modality and complex coverage deliberately. A multimodal tokenizer does
  not establish a trained model's multimodal generation capability.
- Add generate/verify/retrain only after calibrated quality gates, immutable
  provenance, replay, and regression checks are established.

## Capacity target

The long-term target is one billion crop/sample presentations, potentially
revisiting structures. Presentations are not independent experimental sources.

One billion presentations in 14 days requires 826.7/s sustained aggregate
throughput, or 1,033.4/s during training at an 80% duty cycle. Estimate wall time
from measured aggregate throughput and duty cycle; do not count scaling
inefficiency twice. Full proteins and bounded crops are different workloads.

For one billion 1,200-atom records, Float32 coordinates alone occupy 14.4 TB.
Persisting one Int64 pair-index matrix per record would require 11.52 PB.
Identities, masks, source files, replicas, checkpoints, and outputs are additional.
No hardware allocation or completion schedule is inferred from these arithmetic
examples. Scale only after correctness, representative throughput, and held-out
quality are measured together.
