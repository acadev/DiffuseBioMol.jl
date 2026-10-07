# Scaling and cleanup plan

Status: proposed, 2026-09-17. Based on current source inspection, not a measured
GPU run. This adds a plan only; existing working-tree edits are preserved.

2026-09-18 update: at the user's request, a PyTorch training baseline now lives in
`python/`, alongside Julia. See `python/README.md` for runnable commands and scope.
It uses exported Julia tokens, lazy source loading, bounded sequence crops,
flow-matching training, deterministic CPU resume and diagnostic timings. This
implements the initial comparison path, not the complete cleanup/distributed plan.
Measured CPU results and the LayerNorm correctness fix are recorded in
`docs/PYTHON_BASELINE_20260918.md`. Full-network forward and selected-gradient
agreement passed with maximum absolute error below 4e-7.

## Decision and scope

Immediate user priority: demonstrate reproducible training in the existing Julia
implementation before deciding on migration. A Python/PyTorch comparison remains
a later option; it is not a prerequisite for this baseline. Do not maintain two
production trainers indefinitely or start a wholesale rewrite before measurement.

The immediate objective is a correct, observable, resumable streaming trainer
with measured capacity. The user clarified the scale target as **one billion
training crop/sample presentations, revisiting sources**, not one billion unique
structures. Dataset provenance, accelerator allocation, storage and target crop
distribution remain open. Count presentations and optimizer updates explicitly;
do not equate an augmentation with a new independent experimental structure.

## Preserve both objectives and support full structures

Flow matching and diffusion are both required in the long-term design. Source
inspection currently finds a flow-matching objective and Euler sampler only;
there is no separate DDPM/score objective or diffusion sampler to baseline yet.
Preserve the working flow implementation. Add diffusion later as an explicit,
independently tested objective/schedule/prediction-target/sampler combination,
sharing data, conditioning, backbone and evaluation where mathematically valid.
Record objective and noise/path configuration in every checkpoint. A flow model
must not silently be loaded as a diffusion model or relabeled as one.

Crop training must ultimately support full-structure generation. This is a
scientific capability to demonstrate separately from local crop learning:
- Keep source identity, residue indices, sequence, chain breaks, crop membership
  and available global context. Preserve chain topology across spatial crops.
- Include full short proteins and progressively longer contexts in training.
- Investigate coarse global residue/anchor representations with local all-atom
  refinement; disconnected local crops alone cannot specify long-range contacts,
  domain arrangement or topology.
- Validate generation at lengths beyond the crop budget, including long-range
  contacts, chain continuity, clashes and external refolding/designability.
- Overlapping coordinate patches cannot simply be pasted together like image
  tiles: their frames and constraints must agree, and distant interactions must
  be modeled. A hierarchical or globally conditioned procedure needs its own
  quality and memory benchmarks.

These are proposed directions, not claims that the current architecture can
already generate reliable full proteins from crop-only training.

## Reproducible training proof

`scripts/verify_training_baseline.jl DATA_DIR NEW_RUN_DIR` exercises the existing
production baseline runner on six local coordinate sources: four training and
two held-out sources, 128-token maximum crops, three epochs, batch size two,
one Pairformer and one DiT block. It checks exact CPU checkpoint-resume
equivalence (parameters, optimizer, layer state and RNG), fixed-real-crop loss
reduction over 150 updates, and finite sampling. It writes `evidence.toml` and
`step_profile.csv` for each trajectory. No downloads happen during timing.

Set `training.profile_steps = true` in any baseline config for synchronized
diagnostic timings of host preparation, transfer, forward, backward and optimizer
work, plus crop time, loss, real atoms and padded atoms. This mode serializes
measurement boundaries; do not treat it as an overlapped throughput benchmark.

The small run establishes trainability and restart behavior only. It does not
establish held-out quality, full-structure generation, GPU performance, diffusion
training or billion-presentation capacity.

## Findings to address

| Priority | Evidence | Action | Acceptance criterion |
|---|---|---|---|
| P0 | `benchmark_throughput.jl:time_call` has no explicit accelerator synchronization | Separate cold start, host work, transfer, forward, backward, optimizer, and end-to-end timing; synchronize controlled GPU benchmarks | Repeated warmed measurements with hardware, precision, crop lengths, batch sizes, peak memory, p50/p95 latency and real/padded atom throughput |
| P0 | Baseline `gate_report` treats absent initial infill results as a pass; current runner sets these to `nothing` | Represent unavailable gates as not evaluated; establish a fixed small reference set and explicit acceptance policy | Missing measurements cannot produce a successful required gate; compare infill against a documented baseline |
| P0 | `load_corpus` parses serially and caches the whole token corpus as one object | Incremental preprocessing into bounded shards with a versioned portable schema and manifest | Cold/warm load benchmarks; restart processes only unfinished shards; training RAM independent of total corpus size |
| P0 | CUDA library stores per-source quadratic `relpos`; baseline rebuilds it after cropping | Store linear-size coordinates, atom/residue identities, masks, topology and provenance; derive pair features after cropping | No uncropped quadratic arrays on disk; bounded memory for giant sources; schema round-trip fixtures |
| P0 | Baseline evaluation runs full sampling on CPU and materializes all held-out crops | Stream validation batches; run model sampling on GPU; transfer only coordinates for CPU metrics initially | Validation peak RAM bounded by batch/prefetch budgets; independently reported evaluation time |
| P0 | Checkpoint follows expensive evaluation | Save resumable training state before evaluation; schedule evaluation separately against immutable checkpoint IDs | Kill/restart preserves optimizer, RNG, sampler position, split identity and update count; unfinished evaluation is retriable |
| P1 | Crop preparation and transfer run synchronously; CUDA batches shuffle unrelated lengths | Bounded worker prefetch, deterministic seeds, length buckets and a padded-pair budget per batch | Report loader wait, padding efficiency and transfer time; no duplicate/skipped sources across ranks |
| P1 | Dense atom pair features and attention scale quadratically | Sweep crop sizes and precision; measure activation checkpointing; investigate residue/anchor global context with local atom refinement | Compare memory, throughput and held-out quality on the same workload; treat architectural changes separately from refactors |
| P1 | `src/Distributed` is empty | Implement data parallelism in selected stack; use DDP first if model state fits, sharding only when justified | One-rank versus multi-rank gradient agreement within tolerance; globally weighted losses; rank-aware checkpoint/restart and throughput scaling |
| P1 | Training/cache/checkpoint behavior differs across scripts | Extract shared data, batching, training-state, evaluation and instrumentation interfaces; keep thin entry points during transition | One documented production path; existing commands retain compatibility until explicitly deprecated |
| P1 | README/PLAN and scripts disagree on GPU support, epoch-zero evaluation and implemented phases | Replace historical status claims with a capability matrix and reproducible evidence | Every capability marked implemented, tested locally, tested on GPU, or planned; environment and command attached to measurements |

Baseline exercise updates: per-step diagnostic instrumentation is implemented;
missing required infill comparisons now report `not_evaluated` and fail the
aggregate gate (disabled infill reports `not_requested`). The runner also imports
W&B unconditionally, initializing Python/Conda even when logging is disabled;
isolate that optional dependency in the next startup cleanup.

## Measurement protocol

1. Pin source revision plus working-tree patch, environment, GPU model/count,
   storage location, CPU workers, precision, seed and dataset manifest.
2. Use fixed local examples spanning roughly 300/600/1200 atoms and a realistic
   mixed-length subset. Test cold preprocessing, warm shards and synthetic
   already-resident tensors independently. Do not download during timing.
3. Distinguish useful atoms from virtual and batch padding. Report the padded
   pair workload `sum_batches(B * Nmax^2)` as well as structures/second.
4. Synchronize GPU timing boundaries in isolated measurements; use events or a
   profiler for overlapping pipelines rather than serializing normal training.
5. Include optimizer updates and data feeding in sustained throughput; report
   evaluation, checkpointing and downtime separately. Neither raw forward speed
   nor an asynchronous kernel-launch time predicts training capacity.
6. Run a stable 1,000-update test and an interrupted/resumed run before scale-up.
   Add tests for actual invariants and failures, not wrappers mirroring code.

## Stack decision experiment

Time box: two working days after representative data and hardware are available.
Export fixed features, coordinates, masks, conditioning and parameter fixtures.
Implement the same small network/loss and one optimizer step in PyTorch. Compare
forward results, losses, selected gradients and updates within explicit floating
point tolerances. Use the same prior draws and rotations for parity; random seeds
alone do not establish cross-language equivalence.

Compare end-to-end throughput and memory at identical architecture/precision,
then evaluate each stack's supported optimizations separately. Fused attention
eligibility must be measured with the actual pair bias/mask and dtype; it does not
remove the explicit dense pair representation. Select one production stack based
on correctness, throughput, distributed readiness and maintenance cost.

Relevant current primary documentation:
- [PyTorch distributed training](https://docs.pytorch.org/tutorials/distributed.html)
- [PyTorch FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [Lux distributed utilities](https://lux.csail.mit.edu/stable/manual/distributed_utils)
- [AtomWorks](https://github.com/RosettaCommons/atomworks)

The old assertion that Julia has no DDP equivalent is outdated: Lux documents
NCCL/MPI distributed utilities. Compatibility with this pinned environment and
target cluster must still be tested. PyTorch is recommended for reducing the
total integration burden, not because Julia cannot run fast GPU kernels.

## Two-week sequence (conditional milestones, not delivery guarantees)

- Days 1–2: profiling, valid evaluation gates, representative baseline and stack
  comparison. Freeze scope to protein crops for the scale experiment while
  retaining a schema capable of later multimodal complexes.
- Days 3–5: incremental compact shards, streaming input, bounded prefetch,
  checkpoint-before-evaluation, one production training entry point.
- Days 6–8: mixed-precision and activation-memory experiments; batched GPU
  validation; one-node data parallelism and restart checks on available hardware.
- Days 9–10: sustained test on 100k–1M eligible sources if available, quality
  comparison on a clustered holdout, and a measured storage/compute estimate.
  Multi-node expansion proceeds only after one-node correctness and efficiency.

If parity or quality fails, fix it before increasing corpus size. Success after
two weeks is a defensible production path and measured capacity, not an assumed
billion-source training completion.

## Capacity arithmetic

For one billion presentations in 14 days, minimum sustained aggregate throughput
is `1e9 / (14 * 86400) = 826.7 presentations/s`. If training occupies 80% of wall
time, active throughput must exceed 1,033.4/s. On eight GPUs that is 129.2/s/GPU,
before accounting for scaling losses. Fifty passes multiply required work by 50.

Use `wall_time = presentations / (GPUs * measured_per_GPU_rate * efficiency * duty)`.
Do not multiply efficiency twice when using measured aggregate throughput.
Full proteins and bounded crops are not interchangeable throughput units.

Illustrative uncompressed storage for one billion 1,200-atom records:
- Float32 XYZ alone: 14.4 TB, excluding identities, masks, topology and metadata.
- One Int64 pair-index matrix per record: 11.52 PB. Do not persist these.
- Raw source files, replicas, checkpoints and generated outputs are additional.

## Full-roadmap ordering

1. Establish dataset quality, provenance, sequence-clustered splits, stable loss,
   geometry and task-specific generalization before investing in scale.
2. Bring distributed training and streaming infrastructure forward now.
3. Validate conditioning and a separately trained/calibrated verifier on real
   holdouts; passing a geometry gate alone does not establish designability.
4. Add external verification on sampled outputs with explicit compute budgets.
5. Distill sampling only after a useful teacher exists; fewer sampling steps do
   not remove the training data pipeline cost.
6. Enable generate/verify/retrain only after verifier calibration, immutable
   provenance, replay and regression gates. Synthetic outputs do not create
   independent experimental evidence.
7. Scale modality/complex coverage deliberately: current largest-chain protein
   training does not validate the full any-modality roadmap.

The current model uses Cartesian coordinate projections plus augmentation; this
does not guarantee rotational equivariance. Architecture quality and local/global
attention design remain scientific decisions independent of implementation language.
