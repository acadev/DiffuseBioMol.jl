"""
Phase 2: the generalized constraint-token system, generalizing RFdiffusion3's
design-conditioning vocabulary (hotspots, motif scaffolding, RASA burial
targets, center-of-mass constraints) onto the same Pairformer-lite + DiT
backbone used for unconditional co-folding (Phase 1), rather than a separate
design-specific model.

Two different mechanisms are deliberately used for different constraint
kinds, matching how "hard" vs. "soft" they are:

- `is_fixed` (motif/hotspot scaffolding): a *hard* structural fact about the
  sampling state — these atoms' coordinates are given, not generated. They
  are clamped exactly at every sampling step (`FlowMatching.sample_flow`),
  excluded from the training loss, and always shown to the network (zeroing
  this out for a CFG "unconditional" pass would just be lying to the network
  about which coordinates are noisy).
- `is_hotspot` / `rasa_target` (descriptive *soft* conditioning — "this fixed
  atom matters as a binding contact," "burn this residue this much"): these
  are what classifier-free guidance's dual pass drops out, both during
  training (`cond_dropout_prob`) and at sample time (`guidance_scale`).

Symmetry conditioning (RFD3's pre-symmetrized noise for Cn-symmetric design)
and a fully general motif vocabulary (unindexed motifs, H-bond donor/acceptor
typing) are not yet implemented — see `docs/PLAN.md` Phase 2 notes. Center-
of-mass constraints are implemented as a sampling-time *guidance* potential
(`ChainCoMConstraint`), not a learned conditioning signal, since they're a
closed-form geometric quantity that doesn't need the network to represent it.
"""
module Conditioning

export AtomConstraints, no_constraints, constraint_features, N_COND_FEATURES
export ChainCoMConstraint, apply_com_guidance!

const N_COND_FEATURES = 4  # [is_fixed, is_hotspot, rasa_target, has_rasa]

"""
    AtomConstraints

Per-atom design-conditioning annotations, parallel to `Features.TokenFeatures`.

- `is_fixed`: this atom's coordinate is given (motif/hotspot scaffolding,
  binder-design target atoms) rather than generated.
- `is_hotspot`: this fixed atom is a flagged binding contact point (soft —
  CFG-droppable).
- `rasa_target` / `has_rasa`: target relative solvent-accessible surface area
  for burial control, and whether it's set for this atom (soft).
- `fixed_coord`: `3 x N`; only the columns where `is_fixed` is true are
  meaningful.
"""
struct AtomConstraints
    is_fixed::BitVector
    is_hotspot::BitVector
    rasa_target::Vector{Float32}
    has_rasa::BitVector
    fixed_coord::Matrix{Float32}
end

"""
    no_constraints(n) -> AtomConstraints

The fully-unconditional placeholder for `n` atoms — no fixed atoms, no
hotspots, no RASA targets. Used both as a default for plain co-folding
(Phase 1 behavior recovered exactly) and as the "unconditional" half of a CFG
pass at sample time.
"""
no_constraints(n::Int) =
    AtomConstraints(falses(n), falses(n), zeros(Float32, n), falses(n), zeros(Float32, 3, n))

"""
    constraint_features(c::AtomConstraints; drop_soft=false) -> Matrix{Float32}  # N_COND_FEATURES x N

Raw (pre-embedding) per-atom constraint feature matrix. `is_fixed` is always
included (see module docstring); setting `drop_soft=true` zeroes the
hotspot/RASA rows, giving the "unconditional" feature matrix CFG needs.
"""
function constraint_features(c::AtomConstraints; drop_soft::Bool=false)::Matrix{Float32}
    n = length(c.is_fixed)
    feats = Matrix{Float32}(undef, N_COND_FEATURES, n)
    feats[1, :] .= c.is_fixed
    if drop_soft
        feats[2, :] .= 0
        feats[3, :] .= 0
        feats[4, :] .= 0
    else
        feats[2, :] .= c.is_hotspot
        feats[3, :] .= c.rasa_target
        feats[4, :] .= c.has_rasa
    end
    feats
end

"""
    ChainCoMConstraint(moving_chain, reference_chain, target_offset, weight)

A sampling-time guidance potential pulling `moving_chain`'s center of mass
toward `target_offset` (Å, a 3-vector) relative to `reference_chain`'s center
of mass — e.g. for steering a designed binder toward a target pocket.
`chain_idx` values match `Features.TokenFeatures.chain_idx`.
"""
struct ChainCoMConstraint
    moving_chain::Int
    reference_chain::Int
    target_offset::NTuple{3,Float32}
    weight::Float32
end

"""
    apply_com_guidance!(x, chain_idx, constraints, dt)

In-place corrective nudge applied after an Euler step: for each
`ChainCoMConstraint`, moves every atom in `moving_chain` by
`weight * dt * (desired_com - current_com)`, a simple proportional pull
rather than a hard snap (so it composes smoothly with the learned velocity
field instead of fighting it).
"""
function apply_com_guidance!(x::AbstractMatrix, chain_idx::AbstractVector{Int},
    constraints::AbstractVector{ChainCoMConstraint}, dt::Real)
    for c in constraints
        moving = findall(==(c.moving_chain), chain_idx)
        reference = findall(==(c.reference_chain), chain_idx)
        (isempty(moving) || isempty(reference)) && continue
        com_moving = vec(sum(x[:, moving]; dims=2)) ./ length(moving)
        com_reference = vec(sum(x[:, reference]; dims=2)) ./ length(reference)
        desired = com_reference .+ collect(Float32, c.target_offset)
        correction = c.weight * Float32(dt) .* (desired .- com_moving)
        x[:, moving] .+= correction
    end
    x
end

end # module
