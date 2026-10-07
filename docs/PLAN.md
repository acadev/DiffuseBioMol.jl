# DiffuseBioMol.jl: A Modern, Self-Improving All-Atom Biomolecular Diffusion Model

## Context

RFdiffusion3 (RosettaCommons "foundry") and NeuralPLexer3 solve overlapping but distinct halves of the same problem: RFD3 is a **design** engine (generate novel structures/binders under sparse conditioning) built on residue-tokenized atom14 diffusion with classical DDPM and a thin Pairformer; NeuralPLexer3 is a **co-folding/prediction** engine (given sequence/graph inputs, predict the bound complex) built on flow matching with a physics-informed prior and an explicit clash/chirality-aware confidence score. Both are Python/PyTorch, both still rely on **external, post-hoc** validation (refold-and-filter with AF3/Chai-1, or a confidence score with no in-loop correction), and neither treats "improve the model from its own verified failures" as a first-class capability — verification is a filter, not a feedback signal back into the generator.

This project will build a Julia-native, all-atom diffusion/flow system that (a) unifies RFD3's conditioning/design flexibility with NeuralPLexer3's any-to-any co-folding generality and physics-informed sampling, (b) makes physical correctness a first-class, differentiable signal *during* sampling rather than a post-hoc gate, and (c) wraps the whole thing in an agentic generate→verify→curate→retrain loop so the model keeps improving from its own validated/invalidated outputs without a human manually re-running a pipeline each time.

Given the repo is currently empty, this plan covers the **full system design and a phased build-out**, not a diff against existing code.

---

## 1. What each base method actually contributes (research summary)

### RFdiffusion3 / foundry — confirmed from the Sept 2025 bioRxiv preprint + RosettaCommons/foundry repo
- **Representation**: unified residue-level "atom14" tokenization (4 backbone + 10 sidechain slots, virtual-atom padding) for proteins; nucleic acids handled by a *separate* model (RFdiffusion3NA), not one universal tokenizer.
- **Process/network**: classical Gaussian coordinate diffusion (not flow matching), denoised by a transformer U-Net (sparse, distance-gated local attention + atom-token cross-attention) with only a **2-layer** Pairformer (no triangle attention) — deliberately lightweight, ~168M params, ~10x faster sampling than RFdiffusion2.
- **Strengths**: rich conditioning vocabulary (hotspots, unindexed motifs, symmetry, H-bond donor/acceptor, RASA burial targets, CoM constraints, CFG-style dual-pass guidance) purpose-built for *design* tasks.
- **Weaknesses**: no in-loop confidence/energy signal; correctness is checked entirely externally (ProteinMPNN/LigandMPNN → AF3/Chai-1 refold → threshold filter); DNA-binder and complex-motif enzyme success rates are still low (6–15%); no nucleic-acid unification.

### NeuralPLexer3 — confirmed from arXiv:2412.10743
- **Representation**: direct all-heavy-atom coordinates via continuous normalizing flows, with PairFormer-style encoder, DiT-style decoder, anchor-atom global context + sliding-window all-atom refinement.
- **Process**: **flow matching** with a *physics-informed prior* (Langevin polymer model, not pure Gaussian noise) — 40 integration steps vs 100+ for diffusion, ~30s/prediction on one GPU, ~15x faster than AF3.
- **Strengths**: genuinely general across protein-protein, protein-peptide, covalent ligands, PTMs, RNA, DNA, protein-RNA/DNA (i.e., closer to NeuralPlexer3's stated "any biomolecular interaction" goal than RFD3); confidence score explicitly penalizes clashes and chirality violations (`pLDDT − 1000·(clash + chirality_violation)`), so *some* physical-validity signal exists at scoring time.
- **Weaknesses**: it's a predictor, not a conditional designer — no RFD3-style hotspot/motif/symmetry conditioning vocabulary; protein-RNA accuracy (32.7%) lags far behind protein-DNA (56.2%); generalization drops sharply off-distribution (80.2%→60.7% on dissimilar pockets); no in-loop energy guidance or relaxation, only a post-hoc penalized score.

### Complementarity (the actual thesis of this project)
| Axis | RFD3 wins | NeuralPLexer3 wins |
|---|---|---|
| Conditioning vocabulary for design | ✅ | — |
| Any-to-any modality generality (protein/RNA/DNA/ligand/PTM/covalent) | partial (NA is a bolt-on model) | ✅ |
| Sampling speed/steps | ✅ (10x vs RFD2) | ✅ (15x vs AF3, fewer steps via flow matching) |
| In-loop physical-validity signal | ✗ (post-hoc only) | partial (scoring only, not guidance) |
| Symmetry/motif/hotspot design primitives | ✅ | ✗ |

**Conclusion**: take NeuralPLexer3's representation/process (full-atom flow matching, physics-informed prior, any-modality tokenization) as the structural backbone, and graft RFD3's conditioning vocabulary (hotspots, motifs, symmetry, CoM/RASA constraints, CFG-style guidance) onto it as a generalized "constraint token" system. Neither model treats validity-guidance-during-sampling or closed-loop self-improvement as solved — that's the genuine novel contribution layer (Sections 3–4).

---

## 2. SOTA methodological choices to adopt (over either base method)

1. **SE(3)/full-atom flow matching, not DDPM** — straighter probability paths, fewer integration steps, simulation-free training (FoldFlow/FoldFlow-2 lineage, NeuralPLexer3's own justification). Use an OT-based path + a physics-informed prior (polymer/Langevin, per NeuralPLexer3) rather than pure Gaussian, since it measurably reduces required steps.
2. **Unified any-modality atom tokenization** — one tokenizer for protein/RNA/DNA/ligand/ion/PTM atoms (à la AF3/NeuralPLexer3), not RFD3's residue-vs-atom hybrid split. This is the mechanism that makes "any type of interaction at scale" tractable instead of bolting on per-modality models.
3. **Generalized constraint-token system** for conditioning — re-implement RFD3's hotspot/motif/symmetry/RASA/CoM primitives as typed constraint tokens consumed by the same Pairformer-lite + DiT-decoder NeuralPLexer3-style backbone, plus CFG-style dual-pass guidance at sample time.
4. **Differentiable physical-validity guidance during sampling**, not just post-hoc filtering: clash/bond-geometry/chirality terms (PoseBusters-style checks made differentiable) and optionally a lightweight differentiable force field (MadraX-style) added to the flow's velocity field at inference time — the Boltz-2 "steering potentials" precedent is the most directly reusable open design here.
5. **Consistency-distilled fast sampler** as a second inference mode for high-throughput screening/design-loop iterations (1–4 step sampling), trained by distilling the full flow-matching sampler — keeps the "at scale" promise honest.
6. **Confidence/verifier head** (pLDDT/PAE/PDE-style, extended with explicit clash+chirality penalty as in NeuralPLexer3) trained jointly, used both for output ranking and as the **reward signal for the agentic loop** below.

---

## 3. Physical correctness: reason, verify, validate

Three layers, increasingly expensive, increasingly authoritative:

1. **In-loop differentiable guidance** (cheap, runs every sampling step): bond-length/angle/clash/stereochemistry penalty terms added to the learned velocity field, same family as PoseBusters checks but implemented as smooth differentiable losses so they can backprop into the sampler, not just gate outputs afterward.
2. **Learned verifier head** (cheap, runs once per generated structure): jointly-trained confidence/PAE/PDE + clash/chirality classifier, giving a fast scalar reward without invoking external tools.
3. **External ground-truth verification** (expensive, runs on a sampled subset, also the supervision source for the loop): MolProbity/PoseBusters geometry validation, OpenMM/Amber relaxation + energy delta, and design-mode self-consistency (sequence design → refold with the verifier head or an external co-folder → RMSD/ipAE check) — this layer is what actually grounds the reward model in physics rather than the model's own learned opinion of itself.

Layer 3's results are the curated data that feeds back into Section 4 — this is what prevents the agentic loop from drifting into reward-hacking its own verifier.

---

## 4. The agentic self-improving loop

```
            ┌─────────────────────────────────────────────────────────┐
            │                                                         │
   generate │   1. SAMPLE (flow-matching or distilled fast sampler,   │
   ─────────┘      with in-loop physical-validity guidance)           │
            │                                                         │
            ▼                                                         │
   2. SCORE  — learned verifier head (confidence/PAE/clash/chirality) │
            │                                                         │
            ▼                                                         │
   3. TRIAGE — cheap rejects discarded; borderline/high-value cases   │
              routed to expensive external verification (MolProbity/  │
              OpenMM relaxation/self-consistency refold)              │
            │                                                         │
            ▼                                                         │
   4. CURATE — verified-pass and verified-fail structures both kept,  │
              labeled, added to a growing curated dataset (hard       │
              negatives matter as much as positives)                  │
            │                                                         │
            ▼                                                         │
   5. RETRAIN — periodic fine-tuning of (a) the generator on verified-│
              good samples + replay buffer of original training data │
              (avoid distribution collapse), and (b) the verifier     │
              head on the expensive-verification labels (closing the  │
              gap between learned and ground-truth validity signal)  │
            │                                                         │
            └─────────────► back to step 1 with updated weights ──────┘
```

Key design decisions:
- This is **not** an LLM-agent-orchestrated loop (the research above found that pattern still immature for all-atom co-folding) — it's a classical generate→verify→curate→retrain controller, closer to the BindCraft/AlphaProteo precedent, with the verifier head playing the role an LLM agent would play in a more speculative design. Keep the door open to adding an LLM planner later for *campaign-level* decisions (which targets/constraints to explore next), but don't make it the v1 critical path.
- Guard against reward hacking: always keep a slice of expensive external verification in every retrain cycle, and always retain a fixed replay buffer from the original curated training set so fine-tuning can't drift the generator away from physically realistic structures just to please its own verifier.
- Make the loop's cadence configurable and resumable (checkpoint generator + verifier + curated dataset state independently) so it can run as a long-lived background job, not a one-shot script.

---

## 5. Julia engineering plan

### Stack choices (with rationale)
- **NN framework**: Lux.jl (explicit/functional, composes better for custom equivariant/attention layers than Flux's implicit-mutation style).
- **AD**: Enzyme.jl as primary target for GPU-portable training; keep Zygote.jl as fallback for layers Enzyme doesn't yet support cleanly.
- **GPU/kernels**: KernelAbstractions.jl for backend-portable custom kernels (clash/distance/triangle-attention-style ops have no mature Julia package — these will need hand-written kernels), CUDA.jl as primary backend, structured so AMDGPU.jl/Metal.jl ports are possible later.
- **Distributed training**: MPI.jl wrapping vendor GPU-aware MPI for multi-node data/model parallel training; no Julia DDP/FSDP equivalent exists, so the training loop must implement gradient all-reduce explicitly.
- **No equivariant-NN library exists in Julia comparable to e3nn** — budget for hand-rolling the needed SE(3)/frame operations (rotation/translation-equivariant attention, local frame construction) rather than expecting to import one.
- **Pretrained-weight interop**: no tooling exists for loading AF3/NeuralPLexer3/RFD3 PyTorch checkpoints directly; plan to use PythonCall.jl for one-time weight extraction/conversion scripts (export to safetensors/JLD2), not as a runtime dependency.

### Phased roadmap
1. **Phase 0 — core data & tokenization**: PDB/PDBBind/CCD ingestion, unified any-modality atom tokenizer, dataset curation pipeline (mirrors RFD3/NeuralPLexer3 training-data curricula: PDB + AF distillation + DFT-level ligand conformers where available).
2. **Phase 1 — backbone model**: Pairformer-lite encoder + DiT-style decoder, full-atom SE(3) flow matching with physics-informed (Langevin polymer) prior, trained unconditionally first (structure prediction/co-folding only, no design conditioning) — this validates the core architecture against known benchmarks (PoseBusters, a CASP-RNA subset, protein-protein DockQ) before adding complexity.
3. **Phase 2 — constraint/conditioning system**: generalized constraint tokens (hotspot/motif/symmetry/RASA/CoM) + CFG-style dual-pass guidance, turning the Phase 1 predictor into a Phase 2 designer; validate against RFD3's own benchmarks (unconditional monomer refold, binder design, motif scaffolding).
4. **Phase 3 — physical-validity guidance + verifier head**: differentiable clash/bond/chirality guidance terms in the sampler, jointly-trained confidence/PAE/PDE+validity verifier head.
5. **Phase 4 — fast sampler**: consistency distillation of the flow-matching sampler for 1–4 step high-throughput inference.
6. **Phase 5 — agentic loop infrastructure**: triage/curation/retrain controller (Section 4), external verification integrations (MolProbity, OpenMM relaxation via a thin subprocess/Python bridge — this one external dependency is reasonable since these tools have no Julia equivalents), replay-buffer-protected fine-tuning, resumable long-running job orchestration.
7. **Phase 6 — distributed scale-up**: MPI.jl multi-node training, KernelAbstractions-backed custom kernels profiled/optimized, throughput benchmarking against the stated "at scale" goal.

Each phase should ship with its own benchmark harness reusing the same public benchmarks the base methods used (PoseBusters, CASP15 RNA subset, DockQ protein-protein sets, RFD3's enzyme/binder benchmarks where data is available) so progress is measured against the literature, not just internal metrics.

### Verification approach for this plan
Since there's no existing code, "verification" here means: after Phase 1 lands, confirm the unconditional flow-matching predictor reproduces sane bond geometry and beats a naive coordinate-diffusion baseline on a held-out PoseBusters-style validity check before any conditioning/design work begins. Each subsequent phase gets its own go/no-go benchmark gate before moving on, rather than building all six phases before testing anything.

---

## Decisions (resolved)
- **Compute**: multi-node HPC/cloud cluster is available. MPI.jl-based distributed training (originally Phase 6) is pulled forward into the near-term roadmap — the training loop and data pipeline should be designed for multi-node data parallelism from Phase 1 onward, not retrofitted later.
- **V1 scope**: full any-modality generality from the start. The Phase 0 tokenizer must handle protein/RNA/DNA/ligand/ion/PTM atoms in one unified scheme immediately, rather than starting protein+ligand-only and retrofitting nucleic acids later.
- **External-tool bridging** (Phase 5: MolProbity, OpenMM): accepted as a thin Python-interop dependency (PythonCall.jl) even in an otherwise pure-Julia codebase, since no Julia-native equivalents exist.
- **AtomWorks compatibility**: RFdiffusion3/foundry trains against [AtomWorks](https://github.com/RosettaCommons/atomworks) (BSD-3-Clause, github.com/RosettaCommons/atomworks), a general-purpose Python data-loading/featurization layer built on Biotite's `AtomArray`, *not* RFD3-specific code. Since this codebase has no Python dependency, "compatibility" means a Julia-side adapter for AtomArray's data shape (`Data.from_atom_array`), documented with the PythonCall.jl recipe for real interop — not a literal Python bridge exercised in this repo's tests.
- **Testing discipline**: every phase ships both unit tests (hand-derived expected values for varied scenarios) and, where the component is deterministic and dependency-light (pure math, vocab tables, index-bucketing formulas), a dedicated regression test pinning exact golden values. Neural-network components are regression-tested via logical invariants (loss decreases by an expected ratio, output shapes/ranges) rather than brittle exact-weight golden values, which break under routine dependency upgrades without any real logic change.

## Provenance
This implementation is **not** built from RFdiffusion3's or NeuralPLexer3's source code — neither was available to reference directly. Everything here is a from-scratch Julia implementation inspired by the published architectural descriptions in their papers/preprints/blog posts (see the research summaries above for what's confirmed from primary sources vs. inferred). Treat any specific implementation detail not explicitly cited to a primary source as this project's own design choice, not a faithful reproduction of either base method's actual code.

## Training pipeline: compute findings (resolved, with a real fix shipped)

**Hardware target**: nothing in this codebase is NVIDIA-specific — zero GPU code exists yet (confirmed by inspection). But the *available* Julia GPU ecosystem is: CUDA.jl (NVIDIA) is the only backend mature enough for production NN training today. Aurora's Intel GPUs (oneAPI.jl) are not viable for this yet — Lux.jl itself labels the oneAPI backend "experimental," there's no confirmed example anywhere of Lux.jl/Flux.jl training running on it, and Enzyme.jl has no confirmed Intel-GPU support. AMDGPU.jl is more mature but still "experimental" per Lux's own docs; Lux's team recommends Reactant.jl (MLIR/XLA) over native AMDGPU.jl/oneAPI.jl for non-NVIDIA hardware, but Reactant's Intel/SYCL path has no production track record. **Decision: target ALCF's Polaris (NVIDIA A100, documented CUDA.jl support) as the near-term compute target; treat Aurora as a Reactant.jl R&D spike to revisit later, not a blocking requirement.**

**A serious AD-scaling bug was found and fixed before any GPU work started.** Benchmarking a "modest production" config (2.5M params: `d_single=128`, 4 Pairformer + 6 DiT layers) on a real 326-atom structure (1CRN) surfaced a severe Zygote pathology:

1. Multi-head attention implemented as a Julia-level loop over heads (`for h in 1:n_heads ... vcat(heads...)`) made Zygote's first backward-pass compile take **2879s (48 min)** at this scale (vs. 0.5s for a plain forward pass) — fixed by rewriting attention as a single `NNlib.batched_mul` over a `(head_dim, N, n_heads)` batch (verified bit-identical output to the old version before trusting it). This alone dropped *steady-state* per-iteration time from ~514s to ~2s, but first-compile was still ~2822s.
2. The remaining cause: the encoder/decoder were a single `Lux.@compact` containing a runtime `for blk in blocks_vector` loop over a homogeneous `Vector` of blocks. Even though each block's logic was identical, Zygote's AD tracing through that pattern scaled combinatorially with **width × depth together** (neither alone was catastrophic: ~100-130s; both together: 2822s). Fixed by making each block its own standalone `Lux.@compact` layer and composing them via `Lux.Chain` (compile-time-recursive `Tuple`-of-layers application) instead of a `Vector` iterated at runtime. Confirmed directly: same exact config, **80s first-compile, 1.16s/iter steady-state** — ~35x and ~440x improvements respectively.

This is recorded as a load-bearing lesson for future phases: **never compose repeated sub-layers as a `Vector` iterated in a runtime loop inside `Lux.@compact` once the model is bigger than a toy.** Always give each repeated block its own `Lux.@compact`/layer type and compose via `Lux.Chain`. This pattern is also a prerequisite for the CUDA.jl port (Reactant/CUDA compilation would hit the same host-side AD tracing cost, independent of GPU FLOPs) and for Enzyme.jl (untested here, but `Lux.Chain` is the best-supported composition pattern for it too, per Lux's own docs).

**Sequencing decision (resolved): batching before CUDA.jl, CUDA.jl before revisiting Phase 4.** The model is currently unbatched (one structure per forward/backward pass) — a GPU would mostly idle on our structure sizes (N≈300-3000) without batching, so CUDA.jl device support would move the bottleneck onto more expensive hardware rather than fix it. And Phase 4 (consistency distillation) needs a trained model to distill against; building it now would only validate against an untrained toy model. Order: (1) Stage A validity benchmark harness (no new infra needed, ships now), (2) cross-structure batching, (3) CUDA.jl device management on top of batched tensors, (4) real Polaris throughput numbers, (5) revisit Phase 4 once there's a model worth distilling.

**Training-time estimates: explicitly not trustworthy yet, and why.** Measured: 1.16s/iter steady-state, 2.5M params, N=326, unbatched CPU. Extrapolating to ~200K PDB-scale structures gives ~65 CPU-hours/epoch — but unbatched GPU throughput on these structure sizes could plausibly be anywhere from 5x to 50x over CPU depending on how much of the GPU sits idle waiting on small matmuls, which is too wide a band to responsibly quote a single number from. NeuralPLexer3's published 1536 GPU-days (64×H100×24 days) is the best available literature anchor for "what comparable production training costs" but is for a different (batched) architecture at a different scale — treat it as a ballpark expectation-setter, not a prediction for this codebase. Get a real number after batching + CUDA.jl land.

**Benchmarking: where Stage A actually starts (resolved, shipped).** `scripts/benchmark_validity.jl` operationalizes the Phase 1 verification gate above as a recurring, runnable-today harness: compares a raw prior sample (zero learning), an untrained model's `sample_flow` output, and a briefly-trained model, on `Geometry.clash_count`/`Geometry.bond_length_rmsd`, across three held-out real structures (`1L2Y`/`1VII`/`2GB1`) disjoint from the training set. Deliberately does not reach for PoseBusters/CASP15/DockQ yet — those need either a meaningfully-trained model or external tool bridging. Next benchmarking rungs, in order: RFD3-style self-consistency/designability (needs an external refold tool), then CASP15 RNA / DockQ accuracy against literature numbers (needs real training at scale).

**Phase 3 chirality guidance (resolved, shipped).** `Geometry.chiral_centers`/`chiral_volume`/`chirality_energy`/`chirality_count` close the "differentiable clash/bond/chirality guidance terms" item from Section 3's Phase 3 line above (`verifier_loss`'s confidence/clash head predates this and is unchanged; a PAE/PDE-style verifier signal was never built — out of scope, same as before). The correct-handedness sign convention for a residue's `(CA, N, C, CB)` was **not** assumed from a textbook value — it was derived empirically by computing the raw signed volume `dot(N-CA, cross(C-CA, CB-CA))` across all 164 real CA stereocenters in 1CRN/1UBQ/5PTI: consistently positive, `[1.79, 3.13]`, never crossing zero. One real bug found running this end-to-end on real structures (not caught by the tiny hand-built test fixtures, which use 4-9 atoms): the raw signed volume scales as the *cube* of interatomic distance, so early in a sampling trajectory — where coordinates can sit far from realistic molecular scale — `chirality_energy`'s gradient exploded under `validity_guidance_step`'s default `step_size=0.05`, producing `Inf`/`NaN` bond RMSD on real ~150-435 atom held-out structures. Fixed by normalizing: `normalized_chiral_volume` divides by the product of the three substituent-vector norms, bounding the measure to `[-1, 1]` (empirically `[0.53, 0.87]` on the same 164 centers) and keeping the guidance gradient scale-invariant regardless of trajectory state. `scripts/benchmark_validity.jl` now reports a `trained+guided` condition (clash + bond + chirality guidance on the same trained model/seed) alongside prior/untrained/trained, operationalizing this as the Phase 3 go/no-go gate the way Stage A did for Phase 1.

## Benchmarking → batching → CUDA.jl → Polaris/H100 numbers (Sections 1-3 shipped; Section 4 needs real GPU hardware)

**Section 1 (performance benchmarking harness, shipped)**: `scripts/benchmark_throughput.jl`, distinct from the correctness-focused `benchmark_validity.jl` — measures atoms/sec, structures/sec, and forward/backward wall-clock with first-call compile time always reported separately from steady-state. CPU/unbatched (`B=1`) baseline measured: 454.3 atoms/sec (toy, 77K params) / 230.5 atoms/sec (modest, 2.5M params) on 1CRN (326 atoms).

**Section 2 (cross-structure batching, shipped, gate passed)**: `src/Model/Batching.jl` (new module: `BatchedFeatures`, `batch_features`/`batch_relpos`/`batch_coords`/`batch_cond_features`/`attention_pad_bias`/`to_device`), plus batched rewrites of `Model.Network` (every tensor gains a trailing `B` dim; attention masking via an additive `pad_bias` threaded through the `Lux.Chain` alongside `z`/`t_emb`; `t` is now a per-batch-element `Vector{Float32}`), `Sampling.Prior` (batched `sample_prior` dispatch), and `Sampling.FlowMatching` (`BatchedTrainingExample` + batched `prepare_training_example`/`cfm_loss`; the original single-structure API is unchanged in external behavior, internally auto-wrapping to `B=1`). One real bug found and fixed along the way: `Lux.LayerNorm` expects 2D `(features, batch)` input, not our 3D `(d, N, B)` — fixed with a `reshape`-apply-`reshape-back` wrapper (`apply_ln`), the same pattern already used for the pair-update step's `Dense` calls.

Verified: real atoms produce bit-for-bit identical output whether computed alone or padded inside a batch (`test/batching_test.jl`); `B=1` batched throughput (446-455 atoms/sec) matches the pre-batching baseline almost exactly; **gate passed** — atoms/sec scales clearly with batch size on CPU (1CRN, toy config): `B=1`→455, `B=4`→2695-2754, `B=8`→3267, `B=16`→3435 atoms/sec (6-7.6x).

**Batched real-data smoke test (shipped, CPU)**: `scripts/train_phase1_batched_real_data.jl` runs the full fetch → tokenize → `Model.Batching` → batched `cfm_loss` → unbatched `sample_flow` + validity-check path on 20 real, structurally diverse PDB structures (153-1252 atoms each, fetched live from RCSB), batch size 4, 10 epochs — the batched API's first real-data exercise beyond unit tests against tiny synthetic fixtures. Measured (this sandbox, CPU, toy `d_single=32` config): fetch+tokenize 20 structures 31.8s; epoch 1 (Zygote first-compile) 164.3s; steady-state epochs 2-10, 12-16s each; total run (fetch through sampling+validity-check on all 20) ~7 minutes. No errors, NaNs, or shape mismatches across the full size range — this is the relevant correctness signal, not the loss values or post-sampling clash/RMSD numbers, which are expected to look bad at 10 epochs with no Phase 3 guidance applied. One candidate structure (`2CI2`) was cleanly skipped (no tokens survived water/heteroatom filtering) rather than aborting the run, confirming `fetch_targets`'s per-structure try/catch works as intended.

**Local dataset directory support (shipped)**: real runs (Lambda or otherwise) generally shouldn't depend on live per-structure RCSB fetches — `Data.list_structure_files`/`Data.largest_chain` (new, generic, reusable beyond this script) let `scripts/train_phase1_batched_real_data.jl` load from a pre-staged local PDB/mmCIF directory instead: `julia --project=. scripts/train_phase1_batched_real_data.jl /path/to/pdbs`, or `main(; data_dir=...)`. Local files are restricted to their *largest* chain by residue count rather than a hardcoded `"A"`, since a real dataset mirror won't share the curated RCSB list's single-chain convention. A `max_atoms` keyword skips oversized structures before they reach the (still per-structure, unbatched) sampling/validity-check pass. Verified end-to-end against a small locally-cached directory (no network calls, same training/sampling output shape as the RCSB path) — not yet run against a real large local mirror.

**Section 3 (CUDA.jl device support, shipped as far as this sandbox allows)**: deliberately added **zero new dependencies** — `Lux.gpu_device()`/`cpu_device()` already ship with Lux and gracefully no-op to CPU when no GPU trigger package (`CUDA.jl`/`AMDGPU.jl`/etc.) is loaded by the caller (confirmed directly: calling `gpu_device()` with nothing loaded prints a warning and returns a CPU device, doesn't error). `BatchedFeatures`/`BatchedTrainingExample` were made parametric over their array types (rather than hardcoded `Matrix{Int}`/`BitMatrix`) specifically so they can hold either CPU or GPU arrays; `to_device` explicitly converts mask fields from packed `BitMatrix` to dense `Array{Bool}` first, since packed-bit storage has no efficient GPU representation. `scripts/benchmark_throughput.jl`'s `run_batch_size_sweep`/`benchmark_batched_step` take a `device` keyword (default `identity`, i.e. CPU) for exactly this purpose.

**What's NOT done and why**: this sandbox has no GPU, so the actual `device=Lux.gpu_device()` code path (after `using CUDA`) is written but **not executed or verified on real hardware**. `Geometry.jl`'s `O(N²)` scalar-loop clash/bond/lDDT functions were deliberately left CPU-only (not vectorized for GPU) per the plan — they only run at sampling-time, not in the hot training loop, so the right move is transferring the (small) sampled coordinates back to CPU before calling them, not porting the loops.

**Section 4 (Polaris/H100 numbers) — handoff, not yet executed**: requires running on real hardware this session doesn't have access to. To get real numbers:
```julia
using CUDA  # or AMDGPU, etc. — whatever the target node has
import DiffuseBioMol.Model.Network.Lux as Lux
include("scripts/benchmark_throughput.jl")
run_batch_size_sweep(device = Lux.gpu_device())
```
Run this identically on an ALCF Polaris A100 node and on the user's 8×H100 cluster node, compare measured atoms/sec and the H100:A100 ratio against the literature-derived expectation (~1.9-2.6x BF16 training, per MLPerf Training v2.1 — *not* the 3.5x peak-FLOPS ratio or NVIDIA's marketed up-to-9x figures, which rely on FP8/Transformer Engine paths current Julia tooling doesn't expose). A large deviation from that range in either direction is itself a useful finding, not just a number to report. Record the actual measured numbers here once run.

### Is this ready for a Lambda 8×H100 node? (single-GPU yes/untested; multi-GPU no/unbuilt)

Two different questions hide inside "ready for Lambda," and the plan should keep answering them separately:

1. **Single-GPU correctness + throughput on one of the 8 H100s.** Code-complete (`Lux.gpu_device()` dispatch, `Model.Batching.to_device`, parametric `BatchedFeatures`/`BatchedTrainingExample` array types — see Section 3 above), but genuinely never run on real GPU hardware — this sandbox has none. Treat the first Lambda run as the actual first execution of that code path, not a confirmed-working feature: budget time for `CUDA.jl` install/precompile friction, dtype/device-placement mismatches Lux's device dispatch doesn't catch automatically, and anything `NNlib.batched_mul`/broadcasting-based that behaves subtly differently on `CuArray` vs `Array`. The right first step is `run_batch_size_sweep(device=Lux.gpu_device())` from the snippet above — pure throughput/shape verification, no training decisions riding on it yet.
2. **True multi-GPU data-parallel training across all 8 H100s simultaneously.** Not implemented. Section 5's stack choices call for MPI.jl-wrapped gradient all-reduce (originally Phase 6, pulled forward by the "Decisions (resolved)" compute note below), but no actual distributed training loop exists in code yet — only single-device dispatch. Running on Lambda today means picking one of the 8 GPUs (`CUDA.device!(0)` or equivalent) and training on it alone; using all 8 for one training run needs the MPI.jl wiring built first.

Recommended order on Lambda: (1) single-GPU `benchmark_throughput.jl` sweep — confirms the device path works and gives a real H100 atoms/sec number against the MLPerf-derived expectation above; (2) single-GPU run of `scripts/train_phase1_batched_real_data.jl` (or a GPU-enabled variant) on a real dataset bigger than the 20-structure CPU smoke test, as the first real correctness+throughput data point on GPU; (3) only then decide whether multi-GPU data parallelism is worth building now or can wait — single-H100 throughput might already be enough for the next several phases' needs.

---

## Critical review (2026-06): SE(3) invariance gap (shipped) + literature-driven plan updates

A critical pass over the Phase 0-3 code, cross-checked against flow-matching/diffusion literature from the last several months, surfaced one bug-severity gap (now fixed) and several plan-level items worth recording before they're forgotten. None of these block Phase 1-3's current state, but they should inform what gets prioritized next.

### Fixed: no SE(3) invariance, and nothing compensating for it
`Model.Network.build_model`'s `coord_in = Dense(3 => d_single)` (and `Verifier.build_verifier`'s identical `coord_in`) consume raw Cartesian coordinates directly, with no rotation/translation-equivariant architecture (per Section 5's "no equivariant-NN library exists in Julia" note) — and, until this fix, no data augmentation compensating for that either. Every training example was shown to the network in its arbitrary, as-deposited PDB/mmCIF orientation. Recent benchmarking directly on this question ([Protein-SE(3) benchmarking, arXiv:2507.20243](https://arxiv.org/html/2507.20243v1)) confirms model-based equivariance beats augmentation, but also that *removing* augmentation entirely (the bug here) "degrades performance substantially across all metrics" — i.e. this wasn't a minor inefficiency, it was a real generalization-correctness bug.

**Shipped fix** (`src/Sampling/Augmentation.jl`, wired into `Sampling.FlowMatching.prepare_training_example` and `sample_flow`): translation invariance is obtained exactly, for free, by always centering a structure on its real atoms' centroid before it reaches the model (no randomness needed — centering removes the translation degree of freedom from the training distribution entirely); rotation invariance is approximated statistically by drawing a fresh Haar-random `SO(3)` rotation per training example (and independently per batch element in the batched path). `fixed_coord` (motif/hotspot constraints) gets the *same* rotation/centroid draw as the ground truth it's drawn from, so the two stay geometrically consistent; `sample_flow` centers on the fixed atoms' centroid at inference time and shifts the result back, so callers see output in their original frame. See `Sampling.Augmentation`'s and `Sampling.FlowMatching`'s module docstrings for the full mechanism, and `test/augmentation_test.jl` for the isometry/round-trip regression tests.

**Not yet done**: `Verifier.build_verifier` has no training script at all yet (`verifier_loss` is only exercised by hand-built unit fixtures), so it has nothing to wire augmentation into. Whoever writes the real verifier-training pipeline must apply `Augmentation.random_se3_transform`/`apply_se3_transform` the same way `FlowMatching.prepare_training_example` does — noted directly in `Verifier.jl`'s docstring so it isn't missed. Longer-term, per the benchmarking above, a frame-based/equivariant architecture (IPA-style, as AF2/RFdiffusion use) would be a stronger fix than augmentation alone; that's a bigger lift (hand-rolled, since no Julia e3nn-equivalent exists) and should wait until augmentation's effect on real benchmarks is measured and found insufficient.

### Other items from the literature review, recorded for future phases (not yet implemented)
- **Phase 3/5 guidance API should be designed for particle-based steering, not just single-trajectory blending.** The current `sample_flow` API (CFG dual-pass blend + a `post_step` callback for gradient/proportional nudges) is single-trajectory. [Feynman-Kac steering for protein design (arXiv:2511.09216)](https://arxiv.org/abs/2511.09216) reports +89.5% binder designability over plain diffusion guidance using particle resampling against arbitrary reward functions, and [inference-time compute scaling for flow matching (arXiv:2510.17786)](https://arxiv.org/pdf/2510.17786) generalizes this further. Boltz-2 has likewise moved steering potentials from an opt-in add-on to baked-into-default-inference. Before building out Phase 3/5's guidance machinery in earnest, design `sample_flow`'s API to support multi-particle batched rollout + resampling against a reward function (the current single-trajectory `post_step` hook would need to become a different control-flow shape, not just a new term) — this is a sequencing note, not new work to do now.
- **Re-scope Phase 4 from "consistency distillation" to the broader "flow-map" family.** Section 2 item 5 and the Phase 4 roadmap line currently say "consistency-distilled fast sampler." Current literature ([flow-map self-distillation survey, arXiv:2505.18825](https://arxiv.org/html/2505.18825v2); [Distilled Protein Backbone Generation, arXiv:2510.03095](https://arxiv.org/html/2510.03095v1)) frames classic teacher-student consistency distillation as one special case of a broader flow-map family (MeanFlow, shortcut models, consistency trajectory models) that generally trains a single network end-to-end rather than needing a separate distillation pass — simpler and more robust. When Phase 4 actually starts (after there's a real trained Phase 1-3 model worth accelerating, per the existing sequencing decision), default to a flow-map-style method rather than literal teacher-student distillation.
- **OT-path/timestep-shifting remains a real, not-yet-claimed win for Phase 1.** Already flagged as a TODO in `Sampling.FlowMatching`'s module docstring; FoldFlow-SFM's Riemannian-OT lineage is the concrete reference to implement against once the plain-linear-path baseline is validated (per this plan's existing Phase 1 verification gate) — no change to the existing sequencing, just confirming it's still the right next step rather than something superseded.
- **AD backend: plan says Enzyme.jl-primary, code is Zygote-only.** Section 5's stack-choices line ("Enzyme.jl as primary target ... Zygote.jl as fallback") has silently diverged from reality — every module (`Network`, `FlowMatching`, `Verification`) uses Zygote exclusively, and no Enzyme port has been attempted. Resolve this divergence explicitly before CUDA.jl/Polaris work goes further: either spend a session attempting the Enzyme port now (while the model is still small enough that a failure is cheap to diagnose), or update Section 5 to say Zygote is the actual primary backend and downgrade Enzyme to "future investigation." Leaving the plan and code disagreeing is the actual problem, independent of which choice is correct.
- **Geometry's `O(N²)` scalar loops will need a neighbor-list rewrite eventually, not yet.** `Verification.Geometry.clash_energy`/`clash_count`/`lddt` are fine at today's scale (sampling-time only, hundreds to low thousands of atoms) but will need a spatial neighbor-list rewrite once batched sampling or large complexes push N higher. Recorded here as a known future cost, not urgent — re-evaluate when Stage A benchmarking (already shipped) starts showing this as the bottleneck rather than the model forward/backward pass.
