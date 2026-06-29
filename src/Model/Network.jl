"""
The Phase 1 backbone network: a Pairformer-lite encoder (multi-head attention
with a per-head pair-representation bias, plus a pair update from the single
representation — deliberately skipping triangle multiplication/attention,
since RFdiffusion3 found a 2-layer Pairformer-lite sufficient) feeding a
DiT-style decoder (time-conditioned, adaptive-layernorm multi-head self-
attention blocks) that predicts the flow-matching velocity field for every
atom.

Each block is its own standalone `Lux.@compact` layer, and the encoder/
decoder are `Lux.Chain`s of those blocks — *not* a single `@compact` with a
runtime `for blk in blocks_vector` loop, which was the original (and much
simpler-looking) design. That distinction matters a lot in practice: the
runtime-loop-over-a-`Vector` version measured a **47-minute** first-compile
time for Zygote's backward pass at a 10-block, `d_single=128` scale (vs. 28s
for the equivalent `Lux.Chain` composition, confirmed by direct
benchmarking) — `Lux.Chain`'s compile-time-recursive `Tuple`-of-layers
application is dramatically friendlier to Zygote (and, per Lux's own
guidance, to Enzyme and GPU compilation) than a homogeneous `Vector`
iterated at runtime. See `docs/PLAN.md`'s training-pipeline notes for the
full diagnosis. The multi-head attention itself is a single batched
`NNlib.batched_mul` op (not a per-head Julia loop) for the same underlying
reason — that was the first half of this fix, found before the `Lux.Chain`
restructuring.

**Batched (cross-structure) since `docs/PLAN.md`'s batching section**: every
tensor carries a trailing batch dimension `B` (`s` is `d_single x N x B`, `z`
is `d_pair x N x N x B`, etc. — see `build_model`'s docstring for the full
call convention). `B=1` reproduces the unbatched behavior exactly (verified
bit-identical in `test/batching_test.jl`). Padded atoms (from
`Model.Batching.batch_features` et al.) are excluded from attention via an
additive `pad_bias` term threaded alongside `z`/`t_emb` through the
`Lux.Chain`s — see `Model.Batching`'s module docstring for why no other op
needs explicit masking.

v1 simplification, noted explicitly: the pair representation is fixed (not
further updated) once it reaches the decoder. Generalizing that is
straightforward once needed (see `docs/PLAN.md`, Phase 1 verification gate).

**No built-in SE(3) equivariance**: `coord_in = Dense(3 => d_single)` consumes
raw Cartesian coordinates with no rotation/translation invariance baked into
the architecture (no mature Julia equivalent of e3nn exists, see
`docs/PLAN.md`'s engineering-plan section). This is compensated for at the
data/training level, not here — see `Sampling.Augmentation`'s module
docstring and `Sampling.FlowMatching.prepare_training_example`'s use of it.
Any new caller that feeds this model real coordinates outside that path
(e.g. a future verifier-training script calling `Verifier.build_verifier`
directly) needs the same centering + random-rotation treatment.
"""
module Network

using Lux
using NNlib: softmax, gelu, batched_mul, batched_transpose
using Random
using ..Features: N_ELEMENTS, N_MODALITIES, N_POLYMER_ATOM_TYPES, N_RELPOS_BUCKETS
using ..Conditioning: N_COND_FEATURES

export ModelConfig, build_model

"""
    ModelConfig(; d_single=32, d_pair=16, n_heads=4, n_pairformer_layers=2,
                  n_dit_layers=2, d_time=32, d_hidden_mult=4)

`d_single` must be divisible by `n_heads`.
"""
struct ModelConfig
    d_single::Int
    d_pair::Int
    n_heads::Int
    n_pairformer_layers::Int
    n_dit_layers::Int
    d_time::Int
    d_hidden_mult::Int
end

function ModelConfig(; d_single=32, d_pair=16, n_heads=4, n_pairformer_layers=2, n_dit_layers=2, d_time=32, d_hidden_mult=4)
    d_single % n_heads == 0 || throw(ArgumentError("d_single ($d_single) must be divisible by n_heads ($n_heads)"))
    ModelConfig(d_single, d_pair, n_heads, n_pairformer_layers, n_dit_layers, d_time, d_hidden_mult)
end

# --- shared helpers -------------------------------------------------------

"""
    multihead_attention_with_bias(q, k, v, bias, n_heads) -> d x N x B

Multi-head, batched self-attention with an additive per-head pairwise bias,
computed as a single batched operation (`NNlib.batched_mul`, with `n_heads`
and `B` merged into one combined batch dimension — `NNlib.batched_mul` only
supports a single batch dim) rather than a Julia-level loop over heads or
batch elements.

`q`/`k`/`v` are `d x N x B` (`d` divisible by `n_heads`), split into
contiguous `head_dim`-sized chunks per head (head `h` occupies rows
`(h-1)*head_dim+1:h*head_dim`). `bias` is `N x N x n_heads x B` (see
`pair_bias_heads`).
"""
function multihead_attention_with_bias(q::AbstractArray{T,3}, k::AbstractArray{T,3}, v::AbstractArray{T,3},
    bias::AbstractArray{T,4}, n_heads::Int) where {T}
    d, n, b_size = size(q)
    head_dim = d ÷ n_heads

    qh = reshape(permutedims(reshape(q, head_dim, n_heads, n, b_size), (1, 3, 2, 4)), head_dim, n, n_heads * b_size)
    kh = reshape(permutedims(reshape(k, head_dim, n_heads, n, b_size), (1, 3, 2, 4)), head_dim, n, n_heads * b_size)
    vh = reshape(permutedims(reshape(v, head_dim, n_heads, n, b_size), (1, 3, 2, 4)), head_dim, n, n_heads * b_size)
    bias_merged = reshape(bias, n, n, n_heads * b_size)

    logits = batched_mul(batched_transpose(qh), kh) ./ sqrt(Float32(head_dim)) .+ bias_merged  # N x N x (n_heads*B)
    attn = softmax(logits; dims=2)
    out = batched_mul(vh, batched_transpose(attn))  # head_dim x N x (n_heads*B)

    out4 = reshape(out, head_dim, n, n_heads, b_size)
    reshape(permutedims(out4, (1, 3, 2, 4)), d, n, b_size)
end

"""
    pair_bias_heads(pair_bias_dense, z, n_heads) -> N x N x n_heads x B

Projects the pair representation `z` (`d_pair x N x N x B`) to a per-head
scalar bias for every atom pair, via a `Dense(d_pair => n_heads)` layer
applied pointwise across pairs and batch elements.
"""
function pair_bias_heads(pair_bias_dense, z::AbstractArray{T,4}, n_heads::Int) where {T}
    n, b_size = size(z, 2), size(z, 4)
    permutedims(reshape(pair_bias_dense(reshape(z, size(z, 1), :)), n_heads, n, n, b_size), (2, 3, 1, 4))
end

"""
    apply_ln(ln, x) -> Array  # d x N x B

`Lux.LayerNorm` expects a 2D `(features, batch)` input; `s` here is
`d x N x B` (features, sequence position, batch). Reshapes to `(d, N*B)`,
applies `ln`, reshapes back — the same flatten-apply-restore pattern already
used for the pair-update step's `Dense` calls.
"""
function apply_ln(ln, x::AbstractArray{T,3}) where {T}
    d, n, b_size = size(x)
    reshape(ln(reshape(x, d, n * b_size)), d, n, b_size)
end

"""
    adaln_modulate(ln_out, ada_params) -> Array  # d x N x B

`ada_params` is `2d x B`; splits into per-channel, per-batch-element
scale/shift and applies `scale .* ln_out .+ shift` (DiT-style adaptive layer
norm, simplified to scale+shift without the additional output gate for v1).
"""
function adaln_modulate(ln_out::AbstractArray{T,3}, ada_params::AbstractMatrix) where {T}
    d, b_size = size(ln_out, 1), size(ada_params, 2)
    scale = reshape(ada_params[1:d, :], d, 1, b_size)
    shift = reshape(ada_params[d+1:2d, :], d, 1, b_size)
    (1 .+ scale) .* ln_out .+ shift
end

"""
    sinusoidal_embedding(t, d) -> Matrix{Float32}  # d x B

Standard transformer-style sinusoidal embedding of a length-`B` vector of
flow times in `[0, 1]` (one per batch element, independently sampled — see
`docs/PLAN.md`) into a `d x B` matrix (no learned parameters).
"""
function sinusoidal_embedding(t::AbstractVector{<:Real}, d::Int)
    half = d ÷ 2
    freqs = exp.(-log(10000.0f0) .* (0:half-1) ./ half)  # half
    args = reshape(freqs, half, 1) .* reshape(Float32.(t), 1, :)  # half x B
    vcat(sin.(args), cos.(args))  # d x B
end

# --- Pairformer-lite block -------------------------------------------------

"""
    pairformer_block(cfg::ModelConfig) -> Lux layer

One standalone Pairformer-lite block. Call convention: `block((s, z,
pad_bias), ps, st) -> ((s', z', pad_bias), st)`, `s` is `d_single x N x B`,
`z` is `d_pair x N x N x B`, `pad_bias` is `N x N x B` (threaded through
unchanged — see `Model.Batching`). Updates the single representation via
multi-head, pair-biased self-attention + a transition MLP, and updates the
pair representation via an outer-sum of the (post-attention) single
representation, matching the RFdiffusion3 finding that this lightweight
scheme (no triangle multiplication/attention) is sufficient for a 2-layer
Pairformer.
"""
function pairformer_block(cfg::ModelConfig)
    d, dp, dh, n_heads = cfg.d_single, cfg.d_pair, cfg.d_single * cfg.d_hidden_mult, cfg.n_heads
    Lux.@compact(
        wq=Lux.Dense(d => d), wk=Lux.Dense(d => d), wv=Lux.Dense(d => d), wo=Lux.Dense(d => d),
        pair_bias=Lux.Dense(dp => n_heads),
        ln_attn=Lux.LayerNorm((d,)),
        pair_a=Lux.Dense(d => dp), pair_b=Lux.Dense(d => dp), pair_mix=Lux.Dense(dp => dp, gelu),
        ln_pair=Lux.LayerNorm((dp,)),
        mlp1=Lux.Dense(d => dh, gelu), mlp2=Lux.Dense(dh => d),
        ln_mlp=Lux.LayerNorm((d,)),
        n_heads=n_heads,
    ) do szp
        s, z, pad_bias = szp
        n, b_size = size(s, 2), size(s, 3)

        bias = pair_bias_heads(pair_bias, z, n_heads) .+ reshape(pad_bias, n, n, 1, b_size)
        q, k, v = wq(s), wk(s), wv(s)
        attn_out = wo(multihead_attention_with_bias(q, k, v, bias, n_heads))
        s = apply_ln(ln_attn, s .+ attn_out)
        s = apply_ln(ln_mlp, s .+ mlp2(mlp1(s)))

        a, b = pair_a(s), pair_b(s)
        dp_ = size(a, 1)
        pair_update = pair_mix(reshape(reshape(a, dp_, n, 1, b_size) .+ reshape(b, dp_, 1, n, b_size), dp_, :))
        z = ln_pair(reshape(z .+ reshape(pair_update, dp_, n, n, b_size), dp_, :))
        z = reshape(z, dp_, n, n, b_size)

        @return (s, z, pad_bias)
    end
end

# --- DiT-style decoder block ------------------------------------------------

"""
    dit_block(cfg::ModelConfig) -> Lux layer

One standalone DiT-style decoder block. Call convention: `block((s, z,
t_emb, pad_bias), ps, st) -> ((s', z, t_emb, pad_bias), st)` — `z`/`t_emb`/
`pad_bias` are threaded through unchanged (each block only updates `s`) so
the block can be composed into a `Lux.Chain` alongside its siblings.
"""
function dit_block(cfg::ModelConfig)
    d, dh, n_heads = cfg.d_single, cfg.d_single * cfg.d_hidden_mult, cfg.n_heads
    Lux.@compact(
        wq=Lux.Dense(d => d), wk=Lux.Dense(d => d), wv=Lux.Dense(d => d), wo=Lux.Dense(d => d),
        pair_bias=Lux.Dense(cfg.d_pair => n_heads),
        ada1=Lux.Dense(cfg.d_time => 2d),  # time -> (scale, shift) for pre-attention norm
        ada2=Lux.Dense(cfg.d_time => 2d),  # time -> (scale, shift) for pre-MLP norm
        ln1=Lux.LayerNorm((d,); affine=false),
        ln2=Lux.LayerNorm((d,); affine=false),
        mlp1=Lux.Dense(d => dh, gelu), mlp2=Lux.Dense(dh => d),
        n_heads=n_heads,
    ) do sztp
        s, z, t_emb, pad_bias = sztp
        n, b_size = size(s, 2), size(s, 3)
        bias = pair_bias_heads(pair_bias, z, n_heads) .+ reshape(pad_bias, n, n, 1, b_size)

        h = adaln_modulate(apply_ln(ln1, s), ada1(t_emb))
        q, k, v = wq(h), wk(h), wv(h)
        s = s .+ wo(multihead_attention_with_bias(q, k, v, bias, n_heads))

        h2 = adaln_modulate(apply_ln(ln2, s), ada2(t_emb))
        s = s .+ mlp2(mlp1(h2))

        @return (s, z, t_emb, pad_bias)
    end
end

# --- full network ------------------------------------------------------

"""
    build_model(cfg::ModelConfig) -> Lux layer

Call convention: `model((element_idx, modality_idx, polymer_atom_idx,
relpos_idx, x_t, t, cond_features, pad_bias), ps, st)`, all batched (trailing
`B` dimension):
- `element_idx`/`modality_idx`/`polymer_atom_idx`: `N x B` `Matrix{Int}`
  (see `Model.Batching.batch_features`).
- `relpos_idx`: `N x N x B` (see `Model.Batching.batch_relpos`).
- `x_t`: `3 x N x B` current noised/flowed coordinates (`Float32`).
- `t`: length-`B` `Vector{Float32}`, one independently-sampled flow time per
  batch element.
- `cond_features`: `N_COND_FEATURES x N x B` (see
  `Model.Batching.batch_cond_features`; pass an all-zero array, or batch
  `Conditioning.constraint_features(Conditioning.no_constraints(N))` per
  structure, to recover plain unconditional behavior).
- `pad_bias`: `N x N x B` additive attention mask (see
  `Model.Batching.attention_pad_bias`; pass an all-zero array if every
  structure in the batch is the same length, i.e. no padding).

`B=1` for every tensor reproduces the pre-batching behavior exactly (see
`test/batching_test.jl`). Returns the predicted velocity, `3 x N x B`.
"""
function build_model(cfg::ModelConfig)
    encoder = Lux.Chain([pairformer_block(cfg) for _ in 1:cfg.n_pairformer_layers]...)
    decoder = Lux.Chain([dit_block(cfg) for _ in 1:cfg.n_dit_layers]...)

    Lux.@compact(
        elem_emb=Lux.Embedding(N_ELEMENTS => cfg.d_single),
        mod_emb=Lux.Embedding(N_MODALITIES => cfg.d_single),
        poly_emb=Lux.Embedding(N_POLYMER_ATOM_TYPES => cfg.d_single),
        relpos_emb=Lux.Embedding(N_RELPOS_BUCKETS => cfg.d_pair),
        cond_in=Lux.Dense(N_COND_FEATURES => cfg.d_single),
        encoder=encoder,
        decoder=decoder,
        coord_in=Lux.Dense(3 => cfg.d_single),
        time_mlp1=Lux.Dense(cfg.d_time => cfg.d_time, gelu),
        time_mlp2=Lux.Dense(cfg.d_time => cfg.d_time),
        head=Lux.Dense(cfg.d_single => 3; init_weight=Lux.zeros32, init_bias=Lux.zeros32),
        d_time=cfg.d_time,
    ) do input
        element_idx, modality_idx, polymer_atom_idx, relpos_idx, x_t, t, cond_features, pad_bias = input

        s = elem_emb(element_idx) .+ mod_emb(modality_idx) .+ poly_emb(polymer_atom_idx) .+ cond_in(cond_features)
        z = relpos_emb(relpos_idx)
        s, z, pad_bias = encoder((s, z, pad_bias))

        t_emb = time_mlp2(time_mlp1(sinusoidal_embedding(t, d_time)))
        s = s .+ coord_in(x_t)
        s, z, t_emb, pad_bias = decoder((s, z, t_emb, pad_bias))

        @return head(s)
    end
end

end # module
