# Real-data training baseline — 2026-09-17

2026-09-18 correction: cross-framework validation found that the installed Lux
LayerNorm default normalized across tokens/batches. The model now specifies
`dims=1`, and the padding test uses a nonzero output head to expose leakage.
The measurements below are historical results from before that fix. Existing
weights have compatible shapes but changed normalization semantics; retrain a
baseline before using these checkpoints for quality or performance comparisons.

Result: the current Julia flow-matching model trains, learns a fixed real crop,
samples finite coordinates, and resumes exactly on CPU. Protein-quality gates
did not all pass. No GPU throughput or billion-presentation estimate is claimed.

## Reproduce

Stage the six public structures outside the timed run:

```sh
julia --project=. -e 'using DiffuseBioMol; for id in ("1CRN", "1UBQ", "5PTI", "1L2Y", "1VII", "2GB1"); fetch_pdb(id; dir="runs/baseline-smoke-data"); end'
julia --project=. scripts/verify_training_baseline.jl runs/baseline-smoke-data runs/new-training-proof
```

The output directory must be new. These commands download only during staging;
the training proof accepts local sources and does not require W&B credentials.
The runner currently imports W&B even when disabled, so Python/Conda environment
initialization can still occur. That happened before this run's timed `main` call.

## Environment and workload

- Apple M4 CPU, Julia 1.12.6, one Julia thread, two BLAS threads.
- Four training sources: 1L2Y, 1UBQ, 2GB1, 1CRN.
- Two source-disjoint held-out sources: 1VII, 5PTI. This tiny split is not a
  sequence-clustered scientific benchmark.
- Residue-complete crops of at most 128 atom tokens, batch size two.
- One Pairformer and one DiT block; single width 16, pair width 8, four heads.
- Three epochs: six optimizer updates and twelve crop presentations per complete
  training trajectory. A separate fixed-crop experiment performs 150 updates.
- Sampling uses three integration steps for a finite-output smoke check.

## Observed results

| Check | Result |
|---|---|
| Cold six-file parse/cache operation | 0.617 s |
| Warm cache operations in the same process | 0.074 s, then 0.003 s |
| First batch forward including AD setup/compilation | 29.910 s |
| First batch backward including compilation | 30.418 s |
| Subsequent forward + backward steps | 0.0126–0.0243 s |
| Uninterrupted three-epoch `main` call including compilation and evaluation | 87.649 s |
| Fixed real-crop loss before/after 150 updates | 132.5803 → 0.2609 |
| Resume after epoch two, finish epoch three | Parameters, layer state, optimizer and RNG match uninterrupted run exactly |
| Sample shape and finite coordinates | Passed |
| Held-out aggregate protein-quality gates | Failed |
| Baseline pipeline regression tests after gate correction | 48/48 passed |

Per-epoch stochastic training loss was 349.09, 474.90, 354.49. Fresh crops, prior
draws and times make this tiny series noisy; it is not evidence of generalization.
The separate fixed-example result is the controlled evidence that optimization
can learn. Its reduction is about 508-fold, not a held-out improvement claim.

At final held-out evaluation, mean model RMSD was 17.96 versus prior 18.62, but
mean clash count was 32 versus prior 27. Thus the overall quality gate correctly
failed. The brief run is too small to assess scientifically useful generation.
Infill was disabled. The infill-status correction was validated separately by
regression tests after this training process had loaded the script; the original
run's legacy `infill_reconstruction_pass` field is not evidence of an infill pass.

## Artifacts

Local artifacts are under `runs/baseline-proof-20260917/` (ignored by Git):
- `config.toml`, `evidence.toml`, and the reusable lightweight corpus cache.
- `uninterrupted/` and `resumed/`, each with manifest, checkpoint, metrics,
  gates and `step_profile.csv`.

The stage timings are opt-in diagnostics with explicit GPU synchronization when
the runner selects a GPU. They intentionally serialize boundaries. These small
CPU measurements must not be extrapolated to the H100 profile or large crops.

## Next baseline gate

Run the same correctness checks on the actual target GPU, then measure a realistic
crop-length distribution and a larger source-disjoint dataset. Separate a
preloaded-tensor compute benchmark from end-to-end input throughput and evaluation.
Only then size a billion-presentation run. Preserve flow matching, implement and
test the required diffusion path separately, and evaluate full-structure generation
as a distinct capability with long-range/global constraints.
