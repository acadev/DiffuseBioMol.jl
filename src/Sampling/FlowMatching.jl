"""
Conditional flow matching: the linear-path training objective (predict the
constant velocity `x1 - x0` along the straight-line interpolant
`x_t = (1-t) x0 + t x1`) and an Euler-integration sampler, both operating on
top of `Network.build_model`'s velocity field and `Prior.sample_prior`'s
physics-informed `t=0` endpoint. Optimal-transport atom permutation and
timestep-shifting (both used by NeuralPLexer3 to further reduce required
integration steps) are not yet implemented — the plain linear path is the
correct starting point for validating the architecture itself.

Phase 2 conditioning (fixed/clamped atoms for motif scaffolding, CFG dual-pass
guidance, center-of-mass guidance) is supported here through plain arrays and
optional callbacks rather than by importing `Model.Conditioning` directly —
`Sampling` stays decoupled from `Model`'s types; callers build the
`AtomConstraints`/`ChainCoMConstraint` values in `Model.Conditioning` and pass
the derived arrays/closures in.

**SE(3) augmentation**: `Model.Network` feeds raw coordinates through a plain
`Dense` layer, with no built-in rotation/translation equivariance (see
`Augmentation`'s module docstring). `prepare_training_example` therefore
centers the ground-truth structure on its own centroid and applies a fresh
random rotation (via `Augmentation.random_se3_transform`/`apply_se3_transform`)
to every training example, applying the *same* draw to `fixed_coord` so
motif/hotspot atoms stay geometrically consistent with the rest of the
structure. `sample_flow` mirrors this at inference time: when `fixed_coord` is
given, it centers on the fixed atoms' centroid before sampling and adds that
centroid back to the returned coordinates, so the output is in the caller's
original frame despite sampling happening in the centered frame the model was
trained on.

**Two parallel APIs since `docs/PLAN.md`'s batching section**: `Network.build_model`'s
call convention is now always batched (every tensor carries a trailing `B`
dimension — see `Network`'s docstring). The single-structure functions below
(`prepare_training_example(feat::NamedTuple-like, ...)`,
`cfm_loss`/`sample_flow` taking a plain `Matrix`) are unchanged in their
*external* signature/behavior — internally, they wrap into a trivial `B=1`
batch before calling the model and unwrap the result, so every existing
caller keeps working exactly as before. The batched counterparts
(`BatchedTrainingExample`, `cfm_loss`/`sample_flow` taking a
`Model.Batching.BatchedFeatures`) are the genuinely new, `B>1`-capable path.
"""
module FlowMatching

using Random
using ..Prior: sample_prior
using ..Augmentation: random_se3_transform, apply_se3_transform

# `feat` below is a `Model.Features.TokenFeatures` (duck-typed rather than
# imported, to avoid a cross-module path back into `Model` from `Sampling`;
# only its `element_idx`/`modality_idx`/`polymer_atom_idx`/`chain_idx`/
# `is_virtual` fields are used). `batched_feat` is a
# `Model.Batching.BatchedFeatures`, also duck-typed, with the same fields
# plus `pad_mask`.

export TrainingExample, prepare_training_example, cfm_loss, sample_flow
export BatchedTrainingExample

# --- single-structure API (unchanged external behavior; internally wraps to B=1) ---

"""
    TrainingExample

A precomputed (non-differentiable-step) conditional-flow-matching example:
the interpolated input `x_t`, the time `t` it was evaluated at, the target
velocity `x1 - x0`, a boolean mask selecting which atoms have a real
ground-truth target (virtual/padding and fixed/clamped atoms are excluded),
and the per-atom conditioning feature matrix shown to the model. Splitting
this out from `cfm_loss` keeps the RNG-based prior sampling outside of the
Zygote-differentiated closure — `Prior.sample_prior` mutates a buffer
in-place, which Zygote cannot differentiate through, but there is no need to:
`x0` does not depend on the model's parameters.
"""
struct TrainingExample
    x_t::Matrix{Float32}
    t::Float32
    target_v::Matrix{Float32}
    mask::BitVector
    cond_features::Matrix{Float32}
end

"""
    prepare_training_example(feat, x1, cond_features, rng; is_fixed=nothing, fixed_coord=nothing) -> TrainingExample

`x1` is the `3 x N` ground-truth coordinate matrix (virtual/padding atom
columns are ignored regardless of their contents). `cond_features` is the
`N_COND_FEATURES x N` matrix the model should be shown for this example (use
`Conditioning.no_constraints` + `Conditioning.constraint_features` for plain
unconditional examples, or a randomly soft-dropped-out version of a real
constraint set for CFG training).

`is_fixed`/`fixed_coord` mark motif/hotspot atoms whose coordinates are given
rather than generated: they're clamped in `x0` (the flow's start point, not
just its end point) and excluded from the loss `mask`, so the network is
never asked to predict a velocity for them.

`x1` (and `fixed_coord`, if given) are centered on the real atoms' centroid
and rotated by a fresh random rotation before anything else happens — see
the module docstring's "SE(3) augmentation" note. `fixed_coord` is rotated by
the *same* draw as `x1` so the two stay geometrically consistent.
"""
function prepare_training_example(feat, x1::AbstractMatrix, cond_features::AbstractMatrix, rng::AbstractRNG;
    is_fixed::Union{Nothing,BitVector}=nothing, fixed_coord::Union{Nothing,AbstractMatrix}=nothing)
    n = length(feat.element_idx)
    fixed = is_fixed === nothing ? falses(n) : is_fixed
    real_atoms = .!feat.is_virtual

    x1f = Float32.(x1)
    fixed_coord_t = fixed_coord === nothing ? nothing : Float32.(fixed_coord)
    if any(real_atoms)
        R, centroid = random_se3_transform(x1f, real_atoms, rng)
        x1f = apply_se3_transform(x1f, R, centroid)
        fixed_coord_t !== nothing && (fixed_coord_t = apply_se3_transform(fixed_coord_t, R, centroid))
    end

    x0 = Float32.(sample_prior(rng, feat.chain_idx))
    any(fixed) && (x0[:, fixed] .= fixed_coord_t[:, fixed])

    mask = real_atoms .& .!fixed
    x1f[:, .!mask] .= x0[:, .!mask]  # neutralize: zero target velocity where there's no ground truth (virtual or fixed)

    t = rand(rng, Float32)
    x_t = (1 - t) .* x0 .+ t .* x1f
    target_v = x1f .- x0
    TrainingExample(x_t, t, target_v, mask, Float32.(cond_features))
end

"""
    zero_pad_bias(n) -> Array{Float32,3}  # n x n x 1

The trivial (no padding) attention bias for a genuine single structure
wrapped as a `B=1` batch.
"""
zero_pad_bias(n::Int) = zeros(Float32, n, n, 1)

as_batch(m::AbstractMatrix) = reshape(m, size(m, 1), size(m, 2), 1)
as_batch(v::AbstractVector) = reshape(v, length(v), 1)

"""
    cfm_loss(model, ps, st, feat, relpos_idx, example::TrainingExample) -> (loss, st)

Pure (RNG-free) conditional-flow-matching loss for a precomputed
`TrainingExample` — safe to differentiate w.r.t. `ps` with Zygote. Wraps
inputs into a trivial `B=1` batch before calling `model` (always-batched
call convention, see `Network`'s docstring) and unwraps the result; the
external behavior here is unchanged from before batching existed.
"""
function cfm_loss(model, ps, st, feat, relpos_idx::AbstractMatrix, example::TrainingExample)
    n = length(feat.element_idx)
    v_pred3, st = model(
        (
            as_batch(feat.element_idx), as_batch(feat.modality_idx), as_batch(feat.polymer_atom_idx),
            as_batch(relpos_idx), as_batch(example.x_t), Float32[example.t], as_batch(example.cond_features),
            zero_pad_bias(n),
        ),
        ps, st,
    )
    v_pred = dropdims(v_pred3; dims=3)
    n_real = max(count(example.mask), 1)
    loss = sum(abs2, (v_pred .- example.target_v)[:, example.mask]) / (3 * n_real)
    (loss, st)
end

"""
    cfm_loss(model, ps, st, feat, relpos_idx, x1, cond_features, rng) -> (loss, st)

Convenience wrapper that draws a fresh `TrainingExample` and computes the
loss in one call. Do not use this form inside a `Zygote.pullback`/gradient
closure — call `prepare_training_example` outside the closure and use the
`TrainingExample` form instead (see module docstring).
"""
function cfm_loss(model, ps, st, feat, relpos_idx::AbstractMatrix, x1::AbstractMatrix, cond_features::AbstractMatrix, rng::AbstractRNG)
    example = prepare_training_example(feat, x1, cond_features, rng)
    cfm_loss(model, ps, st, feat, relpos_idx, example)
end

"""
    sample_flow(model, ps, st, feat, relpos_idx, cond_features, rng; n_steps=40,
                is_fixed=nothing, fixed_coord=nothing,
                cond_features_uncond=nothing, guidance_scale=1.0,
                post_step=nothing) -> (x1_hat, st)

Draw a structure by Euler-integrating the learned velocity field from a prior
sample (`t=0`) to `t=1` in `n_steps` (NeuralPLexer3 reports good results with
40 flow-matching steps vs. 100+ for diffusion, hence the default).

- `is_fixed`/`fixed_coord`: clamp these atoms to `fixed_coord` at `t=0` and
  re-clamp after every step, regardless of what the velocity field predicts
  (motif/hotspot scaffolding).
- `cond_features_uncond` + `guidance_scale != 1`: classifier-free-guidance
  dual pass — blends the conditioned velocity with the velocity from the
  "soft conditioning dropped" feature matrix
  (`guidance_scale > 1` sharpens adherence to the soft conditioning,
  `guidance_scale = 1` recovers the plain conditioned pass).
- `post_step(x, t) -> x`: optional callback applied after each step (e.g.
  `Conditioning.apply_com_guidance!`), for guidance terms that aren't
  expressed as a velocity-field blend.

When `is_fixed`/`fixed_coord` are given, sampling happens in the frame
centered on the fixed atoms' centroid (matching the centering
`prepare_training_example` applies at training time — see the module
docstring's "SE(3) augmentation" note), and that centroid is added back to
the returned coordinates, so the result is in `fixed_coord`'s original frame
and `x1_hat[:, is_fixed] == fixed_coord[:, is_fixed]` exactly. With no fixed
atoms (plain unconditional sampling), there is nothing to center on and this
is a no-op.
"""
function sample_flow(model, ps, st, feat, relpos_idx::AbstractMatrix, cond_features::AbstractMatrix, rng::AbstractRNG;
    n_steps::Int=40, is_fixed::Union{Nothing,BitVector}=nothing, fixed_coord::Union{Nothing,AbstractMatrix}=nothing,
    cond_features_uncond::Union{Nothing,AbstractMatrix}=nothing, guidance_scale::Real=1.0,
    post_step=nothing)
    n = length(feat.element_idx)

    centroid = zeros(Float32, 3)
    fixed_coord_c = fixed_coord === nothing ? nothing : Float32.(fixed_coord)
    if is_fixed !== nothing && any(is_fixed)
        centroid = vec(sum(view(fixed_coord_c, :, is_fixed); dims=2)) ./ count(is_fixed)
        fixed_coord_c = fixed_coord_c .- centroid
    end

    x = Float32.(sample_prior(rng, feat.chain_idx))
    is_fixed !== nothing && (x[:, is_fixed] .= fixed_coord_c[:, is_fixed])

    pad_bias = zero_pad_bias(n)
    dt = 1.0f0 / n_steps
    for step in 0:n_steps-1
        t = step * dt
        v_cond3, st = model(
            (as_batch(feat.element_idx), as_batch(feat.modality_idx), as_batch(feat.polymer_atom_idx),
                as_batch(relpos_idx), as_batch(x), Float32[t], as_batch(cond_features), pad_bias),
            ps, st,
        )
        v_cond = dropdims(v_cond3; dims=3)
        if cond_features_uncond !== nothing && guidance_scale != 1
            v_uncond3, st = model(
                (as_batch(feat.element_idx), as_batch(feat.modality_idx), as_batch(feat.polymer_atom_idx),
                    as_batch(relpos_idx), as_batch(x), Float32[t], as_batch(cond_features_uncond), pad_bias),
                ps, st,
            )
            v_uncond = dropdims(v_uncond3; dims=3)
            v = v_uncond .+ Float32(guidance_scale) .* (v_cond .- v_uncond)
        else
            v = v_cond
        end

        x = x .+ dt .* v
        is_fixed !== nothing && (x[:, is_fixed] .= fixed_coord_c[:, is_fixed])
        post_step !== nothing && (x = post_step(x, t + dt))
    end
    (x .+ centroid, st)
end

# --- genuinely-batched API (B > 1) -----------------------------------------

"""
    BatchedTrainingExample

The batched counterpart to `TrainingExample`: `x_t`/`target_v` are
`3 x N x B`, `t` is a length-`B` `Vector{Float32}` (one independently-sampled
flow time per batch element, standard CFM practice), `mask` is `N x B`
(combining padding, virtual atoms, and fixed/clamped atoms — all three
reasons a slot is excluded from the loss collapse into one mask), and
`cond_features` is `N_COND_FEATURES x N x B`.
"""
struct BatchedTrainingExample{A<:AbstractArray{Float32,3},V<:AbstractVector{Float32},Bo<:AbstractMatrix{Bool}}
    x_t::A
    t::V
    target_v::A
    mask::Bo
    cond_features::A
end

"""
    prepare_training_example(batched_feat, x1, cond_features, rng; is_fixed=nothing, fixed_coord=nothing) -> BatchedTrainingExample

Batched counterpart of the single-structure `prepare_training_example`.
`batched_feat` is a `Model.Batching.BatchedFeatures`; `x1`/`cond_features`
are already-batched `3 x N x B`/`N_COND_FEATURES x N x B` arrays (see
`Model.Batching.batch_coords`/`batch_cond_features`). `batched_feat.pad_mask`
is folded into the loss mask alongside `is_virtual`/`is_fixed`.

Each batch element gets its own independent centering + random rotation (see
the module docstring's "SE(3) augmentation" note and the single-structure
`prepare_training_example` above) — a plain Julia loop over `b` for this is
fine, same reasoning as `Prior.sample_prior`'s batched form: this isn't
called inside a `Zygote.pullback` closure.
"""
function prepare_training_example(batched_feat, x1::AbstractArray{<:Real,3}, cond_features::AbstractArray{<:Real,3},
    rng::AbstractRNG; is_fixed::Union{Nothing,BitMatrix}=nothing, fixed_coord::Union{Nothing,AbstractArray{<:Real,3}}=nothing)
    n, b_size = size(batched_feat.element_idx)
    fixed = is_fixed === nothing ? falses(n, b_size) : is_fixed
    real_atoms = batched_feat.pad_mask .& .!batched_feat.is_virtual

    x1f = Float32.(x1)
    fixed_coord_t = fixed_coord === nothing ? nothing : Float32.(fixed_coord)
    for b in 1:b_size
        mask_b = view(real_atoms, :, b)
        any(mask_b) || continue
        R, centroid = random_se3_transform(view(x1f, :, :, b), mask_b, rng)
        x1f[:, :, b] .= apply_se3_transform(view(x1f, :, :, b), R, centroid)
        fixed_coord_t !== nothing && (fixed_coord_t[:, :, b] .= apply_se3_transform(view(fixed_coord_t, :, :, b), R, centroid))
    end

    x0 = Float32.(sample_prior(rng, batched_feat.chain_idx))
    any(fixed) && (x0[:, fixed] .= fixed_coord_t[:, fixed])

    mask = real_atoms .& .!fixed
    x1f[:, .!mask] .= x0[:, .!mask]

    t = rand(rng, Float32, b_size)
    t_broadcast = reshape(t, 1, 1, b_size)
    x_t = (1 .- t_broadcast) .* x0 .+ t_broadcast .* x1f
    target_v = x1f .- x0
    BatchedTrainingExample(x_t, t, target_v, mask, Float32.(cond_features))
end

"""
    cfm_loss(model, ps, st, batched_feat, relpos_idx, pad_bias, example::BatchedTrainingExample) -> (loss, st)

Batched counterpart of the single-structure `cfm_loss`. Loss normalization
is **pooled**: sum of squared error over every real `(atom, batch)` entry,
divided by the total real-atom count across the whole batch — the natural
generalization of the per-structure normalization, which is this function's
`B=1` special case.
"""
function cfm_loss(model, ps, st, batched_feat, relpos_idx::AbstractArray{<:Integer,3},
    pad_bias::AbstractArray{Float32,3}, example::BatchedTrainingExample)
    v_pred, st = model(
        (batched_feat.element_idx, batched_feat.modality_idx, batched_feat.polymer_atom_idx,
            relpos_idx, example.x_t, example.t, example.cond_features, pad_bias),
        ps, st,
    )
    n_real = max(count(example.mask), 1)
    mask3 = reshape(example.mask, 1, size(example.mask)...)  # 1 x N x B, broadcasts against the 3 coordinate channels
    loss = sum(abs2, (v_pred .- example.target_v) .* mask3) / (3 * n_real)
    (loss, st)
end

# Extend Model.Batching.to_device so calling to_device(::BatchedTrainingExample, dev)
# dispatches correctly from the DiffuseBioMol top-level scope — same function object,
# new method, no export clash with the BatchedFeatures method already there.
import ...Model.Batching: to_device

"""
    to_device(example::BatchedTrainingExample, device) -> BatchedTrainingExample

Moves every field to `device` (e.g. `Lux.gpu_device()` or `Lux.cpu_device()`).
`mask` is converted from `BitMatrix` to dense `Array{Bool}` before transfer for
the same reason `Model.Batching.to_device` does for `BatchedFeatures`.
"""
function to_device(example::BatchedTrainingExample, dev)
    BatchedTrainingExample(
        dev(example.x_t), dev(example.t), dev(example.target_v),
        dev(Array{Bool}(example.mask)), dev(example.cond_features),
    )
end

end # module
