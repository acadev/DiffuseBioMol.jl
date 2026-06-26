"""
Cross-structure batching: pads a list of independently-featurized structures
(from `Features.featurize`/`Features.relpos_buckets`/`Features.target_coordinates`/
`Conditioning.constraint_features`) into stacked, padded `N_max x B`-shaped
tensors, plus the `pad_mask` that records which slots are real atoms vs.
padding.

Sits on top of the existing per-structure pipeline rather than rewriting it —
`Features`/`Conditioning` stay exactly as they are, well-tested, operating on
one structure at a time; this module is purely an assembly layer.

Two distinct masks are tracked, on purpose, because they answer different
questions and are used in different places:
- `pad_mask` (true = a real atom slot exists here *in this structure*,
  whether or not its coordinate is known): used for **attention masking**
  (`attention_pad_bias`) — a padded slot must never be attended to/from, but
  a real-but-virtual (missing-density) atom should still be attended to.
- `is_virtual` (true = this real atom slot has no known coordinate): used,
  combined with `pad_mask` and any `is_fixed` constraint, for **loss
  masking** in `Sampling.FlowMatching` — unchanged from the unbatched logic,
  just with `pad_mask` added as a third reason to exclude a slot.

See `docs/PLAN.md`'s batching section for the full design rationale,
including why no other op needs explicit masking (only attention and the
Pairformer's pair-update are cross-atom, and the pair-update's padded
garbage never reaches a real atom's output because attention masking
overrides it before it matters).
"""
module Batching

using ..Features: TokenFeatures

export BatchedFeatures, batch_features, batch_relpos, batch_coords, batch_cond_features, attention_pad_bias
export to_device

"""
    BatchedFeatures

The batched counterpart to `Features.TokenFeatures`: every field is
`N_max x B`. Padded slots get index `1` (an arbitrary valid index — never
observed downstream once `pad_mask` masks them out of attention) and
`pad_mask = false`.

Parametric over the underlying array type (`M`/`Bo`) rather than hardcoding
`Matrix{Int}`/`BitMatrix`, so the same struct holds either CPU arrays
(`Matrix{Int}`/`BitMatrix`, what `batch_features` constructs) or GPU arrays
(e.g. `CuArray{Int}`/`CuArray{Bool}`, what `to_device` produces) — see
`to_device`'s docstring for why the mask fields become a dense `Bool` array
rather than staying packed-bit on GPU.
"""
struct BatchedFeatures{M<:AbstractMatrix{Int},Bo<:AbstractMatrix{Bool}}
    element_idx::M
    modality_idx::M
    polymer_atom_idx::M
    chain_idx::M
    res_index::M
    is_virtual::Bo
    pad_mask::Bo
end

"""
    batch_features(examples::AbstractVector{TokenFeatures}) -> BatchedFeatures
"""
function batch_features(examples::AbstractVector{TokenFeatures})::BatchedFeatures
    b_size = length(examples)
    n_max = maximum(length(ex.element_idx) for ex in examples)

    element_idx = ones(Int, n_max, b_size)
    modality_idx = ones(Int, n_max, b_size)
    polymer_atom_idx = ones(Int, n_max, b_size)
    chain_idx = ones(Int, n_max, b_size)
    res_index = zeros(Int, n_max, b_size)
    is_virtual = trues(n_max, b_size)
    pad_mask = falses(n_max, b_size)

    for (b, ex) in enumerate(examples)
        n = length(ex.element_idx)
        element_idx[1:n, b] .= ex.element_idx
        modality_idx[1:n, b] .= ex.modality_idx
        polymer_atom_idx[1:n, b] .= ex.polymer_atom_idx
        chain_idx[1:n, b] .= ex.chain_idx
        res_index[1:n, b] .= ex.res_index
        is_virtual[1:n, b] .= ex.is_virtual
        pad_mask[1:n, b] .= true
    end

    BatchedFeatures(element_idx, modality_idx, polymer_atom_idx, chain_idx, res_index, is_virtual, pad_mask)
end

"""
    batch_relpos(relpos_matrices) -> Array{Int,3}  # N_max x N_max x B

Pads each `N_b x N_b` relative-position bucket matrix to `N_max x N_max`
(arbitrary bucket index `1` for padded entries — masked out of attention,
see module docstring) and stacks along a new trailing batch dimension.
"""
function batch_relpos(relpos_matrices::AbstractVector{<:AbstractMatrix{<:Integer}})::Array{Int,3}
    b_size = length(relpos_matrices)
    n_max = maximum(size(m, 1) for m in relpos_matrices)
    out = ones(Int, n_max, n_max, b_size)
    for (b, m) in enumerate(relpos_matrices)
        n = size(m, 1)
        out[1:n, 1:n, b] .= m
    end
    out
end

"""
    batch_coords(coord_matrices) -> Array{Float32,3}  # 3 x N_max x B

Pads each `3 x N_b` coordinate matrix to `3 x N_max` (zeros for padded
columns) and stacks along a new trailing batch dimension. Used for both
ground-truth coordinates (`Features.target_coordinates` output) and
conditioning-constraint fixed-coordinate matrices.
"""
function batch_coords(coord_matrices::AbstractVector{<:AbstractMatrix})::Array{Float32,3}
    b_size = length(coord_matrices)
    n_max = maximum(size(m, 2) for m in coord_matrices)
    out = zeros(Float32, 3, n_max, b_size)
    for (b, m) in enumerate(coord_matrices)
        n = size(m, 2)
        out[:, 1:n, b] .= Float32.(m)
    end
    out
end

"""
    batch_cond_features(cond_features_list) -> Array{Float32,3}  # N_COND_FEATURES x N_max x B

Pads each `N_COND_FEATURES x N_b` conditioning-feature matrix to
`N_COND_FEATURES x N_max` (zeros — "no constraint" for padded columns,
matching `Conditioning.no_constraints`'s all-zero convention) and stacks
along a new trailing batch dimension.
"""
function batch_cond_features(cond_features_list::AbstractVector{<:AbstractMatrix})::Array{Float32,3}
    b_size = length(cond_features_list)
    n_max = maximum(size(m, 2) for m in cond_features_list)
    n_cond = size(first(cond_features_list), 1)
    out = zeros(Float32, n_cond, n_max, b_size)
    for (b, m) in enumerate(cond_features_list)
        n = size(m, 2)
        out[:, 1:n, b] .= Float32.(m)
    end
    out
end

"""
    attention_pad_bias(pad_mask::BitMatrix) -> Array{Float32,3}  # N_max x N_max x B

`bias[i, j, b] = 0` if atom `j` is real in structure `b` (`pad_mask[j, b]`),
else a large negative number — added to attention logits *before* the
per-head split (see `Network.multihead_attention_with_bias`), masking
padded **key** positions only (column-masking, not row-masking: a padded
query's output is discarded anyway, and masking only columns guarantees no
softmax row is ever all-masked, which would produce `NaN`).
"""
function attention_pad_bias(pad_mask::BitMatrix)::Array{Float32,3}
    n_max, b_size = size(pad_mask)
    bias = zeros(Float32, n_max, n_max, b_size)
    for b in 1:b_size, j in 1:n_max
        pad_mask[j, b] || (bias[:, j, b] .= -1.0f9)
    end
    bias
end

"""
    to_device(batched::BatchedFeatures, device) -> BatchedFeatures

Moves every field of a `BatchedFeatures` to `device` (e.g. `Lux.gpu_device()`
or `Lux.cpu_device()` — see `Lux`'s device-management docs; no CUDA.jl
dependency is added by this package itself, since `Lux.gpu_device()`
gracefully falls back to CPU when no GPU trigger package
(`CUDA.jl`/`AMDGPU.jl`/etc.) is loaded by the caller).

`is_virtual`/`pad_mask` are explicitly converted from `BitMatrix` to
`Array{Bool}` first — packed-bit storage has no efficient GPU
representation, so transferring a `BitMatrix` directly is the kind of thing
that works on one backend and silently misbehaves (or errors) on another;
converting to a plain dense `Bool` array first is the safe, explicit choice.
Plain tensors that aren't wrapped in a custom struct (coordinates, `relpos`,
`cond_features`, `attention_pad_bias`'s output) don't need this helper — just
call `device(tensor)` directly.
"""
function to_device(batched::BatchedFeatures, device)
    BatchedFeatures(
        device(batched.element_idx), device(batched.modality_idx), device(batched.polymer_atom_idx),
        device(batched.chain_idx), device(batched.res_index),
        device(Array{Bool}(batched.is_virtual)), device(Array{Bool}(batched.pad_mask)),
    )
end

end # module
