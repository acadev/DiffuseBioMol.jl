# DiffuseBioMol.jl

A Julia-native, all-atom biomolecular diffusion/flow model combining
[RFdiffusion3](https://github.com/RosettaCommons/foundry)'s conditioning/design
flexibility with [NeuralPLexer3](https://github.com/zrqiao/NeuralPLexer)'s
any-to-any co-folding generality — with physical correctness as an in-loop,
differentiable signal rather than a post-hoc filter, and a closed-loop
generate → verify → curate → retrain controller for continuous self-improvement.

**Provenance note**: this is a from-scratch Julia implementation *inspired by*
the published architectural descriptions of RFdiffusion3 and NeuralPLexer3
(papers/preprints/blog posts) — not a port of, or built against, their actual
source code. Neither codebase was available to reference directly while
building this. See `docs/PLAN.md`'s research summaries for what's confirmed
from primary sources vs. inferred.

Full design rationale, literature review, and phased roadmap:
[`docs/PLAN.md`](docs/PLAN.md).

## Status

Phases 0-3 are implemented and tested (see `test/` — unit tests with
hand-computed expected values, plus a dedicated `regression_test.jl` pinning
exact golden values for deterministic components):

- **Phase 0** (`src/Tokenization/`, `src/Data/`): a unified any-modality atom
  tokenizer — protein, RNA, DNA, ligand, and ion residues all become the same
  `AtomToken` representation, with fixed-slot padding for polymer residues and
  atom-by-atom tokenization for non-polymer ones — plus a PDB/mmCIF parser
  (via BioStructures.jl) producing the `ParsedResidue` records the tokenizer
  consumes, a `Data.fetch_pdb` helper for pulling real structures straight
  from the RCSB, and `Data.from_atom_array` — an adapter for data shaped like
  [AtomWorks](https://github.com/RosettaCommons/atomworks)/Biotite's
  `AtomArray` (the data layer RFdiffusion3/foundry trains against). This is
  *not* a Python bridge (this codebase has no Python dependency) — it's the
  Julia-side conversion logic for that data shape; see the function's
  docstring for the actual PythonCall.jl recipe to use it with a real
  `AtomArray`, which isn't exercised in this repo's test suite since it
  requires a Python environment with `atomworks`/`biotite` installed.
- **Phase 1** (`src/Model/`, `src/Sampling/`): a Pairformer-lite encoder
  (multi-head attention with a per-head pair-representation bias, no triangle
  attention, per RFdiffusion3's finding that this is sufficient) feeding a
  DiT-style decoder (time-conditioned, adaptive-layernorm multi-head
  attention blocks) that predicts a flow-matching velocity field; a
  physics-informed Langevin-polymer prior (`Sampling.Prior`, following
  NeuralPLexer3) replacing pure Gaussian noise at `t=0`; and a
  conditional-flow-matching training loss + Euler sampler
  (`Sampling.FlowMatching`). Validated both on synthetic fixtures (the
  network fits a single example to <20% of its initial loss) and on real PDB
  structures (`scripts/train_phase1_real_data.jl` trains across 1CRN/1UBQ/5PTI
  fetched live from the RCSB, 326-648 atoms each, with no errors and a finite,
  correctly-shaped sampled output).
- **Phase 2** (`src/Model/Conditioning.jl`): a generalized constraint-token
  system generalizing RFdiffusion3's design vocabulary onto the same backbone
  — `is_fixed` atoms (motif/hotspot scaffolding) are clamped exactly at every
  sampling step and excluded from the training loss; `is_hotspot`/
  `rasa_target` are soft conditioning subject to classifier-free-guidance
  dual-pass dropout (`Sampling.FlowMatching.sample_flow`'s
  `cond_features_uncond`/`guidance_scale`); `ChainCoMConstraint` is a
  closed-form sampling-time guidance potential for center-of-mass targeting.
- **Phase 3** (`src/Verification/`): the three-layer physical-correctness
  stack's first two layers. `Geometry` is pure, dependency-free, Zygote-
  differentiable coordinate math — clash energy (VDW-radius-based repulsion,
  excluding covalently-local atom pairs), bond-length energy (harmonic
  restraint against `backbone_bonds`' derived ideal-length pairs), and
  per-atom lDDT (the real AlphaFold-style local distance difference test,
  used as a self-supervised confidence label). `validity_guidance_step`
  turns `Geometry.validity_energy`'s negative gradient into a sampling-time
  correction, pluggable directly into `sample_flow`'s `post_step` hook — no
  changes to `Sampling.FlowMatching`'s API were needed. `Verifier` is a
  small, self-contained network (deliberately not sharing code with
  `Model.Network`, mirroring how `Sampling` stays decoupled from `Model`)
  predicting per-atom confidence + clash likelihood, trained against
  `Geometry`-derived labels without needing external tools. Layer 3
  (MolProbity/OpenMM relaxation/self-consistency refold) is still
  unimplemented — it's the one layer that will need a Python interop story.

v1 simplifications worth knowing about (documented in the relevant module
docstrings, not hidden): a pair representation that's built by the encoder
but not further updated by the decoder; the closed-vocabulary atom-identity
embedding only applies to polymer atoms (ligand/ion/PTM atoms are typed by
element + modality only); the flow-matching path is plain linear
interpolation, not yet NeuralPLexer3's optimal-transport permutation +
timestep shifting; symmetry conditioning (RFD3's pre-symmetrized noise for
Cn-symmetric design) and a richer motif vocabulary (unindexed motifs, H-bond
donor/acceptor typing) are not yet implemented; `backbone_bonds`/bond-length
guidance only covers protein backbone geometry (RNA/DNA backbone bonds are
future work); chirality/stereochemistry guidance isn't implemented yet —
only clash and bond-length terms are; and `Verification.Verifier` (the
learned verifier head) is not yet batched (only the main flow-matching model
is, see below) since it wasn't in this round's scope.

Everything else (`AgenticLoop`, `Distributed`, Layer 3 external verification)
is a documented stub module — see each module's docstring for what it's
responsible for and which roadmap phase it belongs to.

## Training pipeline / compute

CPU is fine for tests; real training needs GPU, and that surfaced a real bug.
Benchmarking a "modest production" config (2.5M params, 10 layers) found a
severe Zygote compile-time pathology — first backward-pass compile took
**47 minutes** at that scale. Root causes and fixes (both shipped, see
`docs/PLAN.md`'s training-pipeline section for the full diagnosis with
numbers):

1. Multi-head attention was a Julia-level loop over heads + `vcat` — fixed
   with a single batched `NNlib.batched_mul` (verified bit-identical output).
2. The encoder/decoder composed repeated blocks as a `Vector` iterated in a
   runtime loop inside one `Lux.@compact` — fixed by giving each block its
   own `Lux.@compact` layer and composing via `Lux.Chain`. Zygote handles
   `Lux.Chain`'s compile-time-recursive composition far better than a
   homogeneous-`Vector`-at-runtime loop, especially as width and depth grow
   *together* (neither alone was catastrophic; combined, it was).

Net result on the same config: **2822s → 80s first-compile, 514s → 1.16s/iter
steady-state.** This is now the standing rule for this codebase: never
compose repeated sub-layers as a `Vector` looped over inside `@compact` —
always `Lux.Chain` of standalone block layers.

**Hardware**: nothing here is NVIDIA-specific (no GPU code exists yet), but
the practical reality is that CUDA.jl is the only Julia GPU backend mature
enough for this today. Aurora's Intel GPUs (oneAPI.jl) aren't there yet —
Lux.jl labels that backend "experimental" with no confirmed NN-training track
record; AMDGPU.jl is more mature but Lux's own team recommends Reactant.jl
over it for production non-NVIDIA work, and Reactant's Intel/SYCL path is
unproven. **Near-term target: ALCF's Polaris (NVIDIA A100s, documented Julia+CUDA.jl support).** Aurora is a Reactant.jl R&D spike for later, not a
blocker.

**Sequencing decision**: cross-structure batching before CUDA.jl device
support, before Phase 4 — now executed (see below), gates passed.

### Batching (shipped)

`src/Model/Batching.jl`: every tensor through `Model.Network` now carries a
trailing batch dimension `B` (`s` is `d_single x N x B`, `z` is
`d_pair x N x N x B`, etc.). Padded atoms (from structures of different
length stacked into one batch) are masked out of attention via an additive
`pad_bias`, threaded through the `Lux.Chain` alongside `z`/`t_emb` — no other
op needed explicit masking (`Dense`/`LayerNorm` are per-atom; the only other
cross-atom op, the Pairformer's pair-update, can't leak padded garbage into
real atoms because the attention mask already overrides it where it'd
matter). The original single-structure API (`TrainingExample`,
`cfm_loss(model, ps, st, feat::TokenFeatures, ...)`, etc.) is **unchanged in
external behavior** — it auto-wraps to `B=1` internally — so nothing existing
broke; `BatchedTrainingExample` + the `BatchedFeatures`-based overloads are
the new `B>1` path.

One real bug found: `Lux.LayerNorm` expects 2D `(features, batch)` input,
not 3D `(d, N, B)` — fixed with a flatten/apply/restore wrapper (`apply_ln`),
same pattern already used for the pair-update step's `Dense` calls.

**Gate passed**: real atoms produce bit-identical output whether computed
alone or padded inside a batch (`test/batching_test.jl`); atoms/sec scales
clearly with batch size on CPU (1CRN, toy 77K-param config):

| Batch size | atoms/sec |
|---|---|
| 1 | 446-455 |
| 4 | 2695-2754 |
| 8 | 3267 |
| 16 | 3435 |

### CUDA.jl device support (shipped, not yet run on real GPU hardware)

Deliberately **zero new dependencies**: `Lux.gpu_device()`/`cpu_device()`
already ship with Lux and gracefully fall back to CPU when no GPU trigger
package (`CUDA.jl`/`AMDGPU.jl`/etc.) is loaded — confirmed directly in this
sandbox (no GPU here). `BatchedFeatures`/`BatchedTrainingExample` were made
parametric over their array types so they transparently hold either CPU or
GPU arrays; `Model.Batching.to_device` explicitly converts mask fields from
packed `BitMatrix` to dense `Array{Bool}` first (packed-bit storage has no
efficient GPU representation). `Verification.Geometry`'s `O(N²)` scalar-loop
clash/bond/lDDT checks were deliberately **not** ported to GPU — they only
run at sampling time, not the hot training loop, so the right move is
transferring the small sampled-coordinate array back to CPU first, not
vectorizing the loops.

### Polaris / H100 numbers — handoff (needs real hardware this session doesn't have)

```julia
using CUDA  # or AMDGPU, etc., whatever the node has
import DiffuseBioMol.Model.Network.Lux as Lux
include("scripts/benchmark_throughput.jl")
run_batch_size_sweep(device = Lux.gpu_device())
```
Run identically on a Polaris A100 node and the H100 cluster node; compare
measured atoms/sec and the H100:A100 ratio against ~1.9-2.6x (MLPerf
Training v2.1's BF16 range — not the 3.5x peak-FLOPS ratio or NVIDIA's
marketed up-to-9x figures, which need FP8/Transformer Engine paths current
Julia tooling doesn't expose). See `docs/PLAN.md` for the full hardware
research this expectation is based on.

## Benchmarking

`scripts/benchmark_validity.jl` is the Stage A benchmark — runnable today,
zero new dependencies, zero GPU. It does *not* reach for literature-standard
benchmarks (PoseBusters, CASP15, DockQ): those need either a meaningfully-
trained model or external tool bridging, both premature before batching/GPU
training exists. Instead it checks the thing `docs/PLAN.md` originally
specified as the Phase 1 gate — does the model produce physically sane
geometry on held-out real structures, and is it better than no learning at
all — using only this codebase's own `Geometry` machinery (`clash_count`,
`bond_length_rmsd`). Three conditions per held-out structure (`1L2Y`/`1VII`/
`2GB1`, disjoint from the training set `1CRN`/`1UBQ`/`5PTI`): a raw prior
sample (zero learning, the floor), an untrained model's `sample_flow` output
(isolates architecture/guidance effects from learning), and a briefly-trained
model. This is the number to track as training scales up — not a finished
evaluation. Once there's a real trained model at meaningful scale, the next
benchmarking rungs are: self-consistency/designability checks (RFD3-style,
need an external refold tool), then CASP15 RNA / DockQ protein-complex
accuracy against literature numbers.

## Roadmap

0. ✅ Core data & unified any-modality tokenization (+ AtomWorks-shaped data adapter)
1. ✅ Pairformer-lite + DiT backbone, SE(3) flow matching with a physics-informed prior (hardened: multi-head attention, real-PDB training)
2. ✅ Generalized constraint-token conditioning system (hotspot/motif/RASA/CoM; symmetry deferred)
3. ✅ Differentiable physical-validity guidance (clash + bond-length) + learned verifier head (confidence + clash); chirality guidance and Layer 3 external verification still open
4. Consistency-distilled fast sampler
5. Agentic generate→verify→curate→retrain loop
6. Multi-node distributed training (MPI.jl)

## Development

```sh
julia --project=. -e 'using Pkg; Pkg.test()'

# Real-PDB Phase 1 training smoke test (needs network access to RCSB):
julia --project=. scripts/train_phase1_real_data.jl

# Correctness benchmark (Stage A):
julia --project=. scripts/benchmark_validity.jl

# Performance benchmark (CPU baseline + batch-size sweep; add device=Lux.gpu_device()
# after `using CUDA` for a real GPU run):
julia --project=. scripts/benchmark_throughput.jl
```
