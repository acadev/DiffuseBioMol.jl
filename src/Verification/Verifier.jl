"""
The learned verifier head (Layer 2 of `docs/PLAN.md`'s three-layer physical-
correctness stack): a small, self-contained Lux network that reads a
structure's coordinates and per-atom categorical features and predicts, per
atom, (a) a confidence score (trained against `Geometry.lddt` labels) and
(b) a clash logit (trained against `Geometry.clash_energy`-derived labels) —
a fast scalar reward that doesn't require invoking external tools
(MolProbity/OpenMM, Layer 3) at every candidate-scoring step.

Deliberately *not* built by importing `Model.Network`'s internals: this is a
separate, smaller network (no weight sharing intended), and keeping
`Verification` decoupled from `Model`'s implementation details mirrors how
`Sampling` stays decoupled from `Model` (see those modules' docstrings).
Vocabulary sizes (element/modality/polymer-atom-type/relpos-bucket counts)
are passed in explicitly by the caller rather than imported, for the same
reason.

Each attention block is its own standalone `Lux.@compact` layer, composed via
`Lux.Chain` — see `Model.Network`'s module docstring for why this matters
(a homogeneous-`Vector`-iterated-at-runtime version of this same pattern hit
a 47-minute Zygote first-compile time at production scale; `Lux.Chain`'s
compile-time-recursive composition does not).
"""
module Verifier

using Lux
using NNlib: softmax, gelu, sigmoid, batched_mul, batched_transpose

export VerifierConfig, build_verifier

"""
    VerifierConfig(; d_single=24, n_heads=4, n_layers=2, d_hidden_mult=4,
                      n_elements, n_modalities, n_polymer_atom_types, n_relpos_buckets)
"""
struct VerifierConfig
    d_single::Int
    n_heads::Int
    n_layers::Int
    d_hidden_mult::Int
    n_elements::Int
    n_modalities::Int
    n_polymer_atom_types::Int
    n_relpos_buckets::Int
end

function VerifierConfig(; d_single=24, n_heads=4, n_layers=2, d_hidden_mult=4,
    n_elements, n_modalities, n_polymer_atom_types, n_relpos_buckets)
    d_single % n_heads == 0 || throw(ArgumentError("d_single ($d_single) must be divisible by n_heads ($n_heads)"))
    VerifierConfig(d_single, n_heads, n_layers, d_hidden_mult, n_elements, n_modalities, n_polymer_atom_types, n_relpos_buckets)
end

"""
Batched (not per-head-looped) multi-head attention — see
`Model.Network.multihead_attention_with_bias`'s docstring for why this
matters. `bias` is `N x N x n_heads` (batch last, matching
`NNlib.batched_mul`'s convention).
"""
function multihead_attention_with_bias(q::AbstractMatrix, k::AbstractMatrix, v::AbstractMatrix,
    bias::AbstractArray{T,3}, n_heads::Int) where {T}
    d, n = size(q)
    head_dim = d ÷ n_heads

    qh = permutedims(reshape(q, head_dim, n_heads, n), (1, 3, 2))
    kh = permutedims(reshape(k, head_dim, n_heads, n), (1, 3, 2))
    vh = permutedims(reshape(v, head_dim, n_heads, n), (1, 3, 2))

    logits = batched_mul(batched_transpose(qh), kh) ./ sqrt(Float32(head_dim)) .+ bias
    attn = softmax(logits; dims=2)
    out = batched_mul(vh, batched_transpose(attn))

    reshape(permutedims(out, (1, 3, 2)), d, n)
end

"""
    verifier_block(cfg::VerifierConfig) -> Lux layer

One standalone attention block. Call convention: `block((s, bias), ps, st)
-> ((s', bias), st)` — `bias` (the fixed per-head pairwise bias, computed
once outside the chain) is threaded through unchanged so the block composes
into a `Lux.Chain`.
"""
function verifier_block(cfg::VerifierConfig)
    d, dh = cfg.d_single, cfg.d_single * cfg.d_hidden_mult
    n_heads = cfg.n_heads
    Lux.@compact(
        wq=Lux.Dense(d => d), wk=Lux.Dense(d => d), wv=Lux.Dense(d => d), wo=Lux.Dense(d => d),
        ln_attn=Lux.LayerNorm((d,)),
        mlp1=Lux.Dense(d => dh, gelu), mlp2=Lux.Dense(dh => d),
        ln_mlp=Lux.LayerNorm((d,)),
        n_heads=n_heads,
    ) do sbias
        s, bias = sbias
        attn_out = wo(multihead_attention_with_bias(wq(s), wk(s), wv(s), bias, n_heads))
        s = ln_attn(s .+ attn_out)
        s = ln_mlp(s .+ mlp2(mlp1(s)))
        @return (s, bias)
    end
end

"""
    build_verifier(cfg::VerifierConfig) -> Lux layer

Call convention: `model((element_idx, modality_idx, polymer_atom_idx,
relpos_idx, coords), ps, st)` -> `(confidence, clash_logit)`, each a
length-`N` vector. `confidence` is already passed through a sigmoid (`[0,
1]`); `clash_logit` is a raw logit (use `NNlib.sigmoid` or a
binary-cross-entropy-with-logits loss, don't apply sigmoid twice).
"""
function build_verifier(cfg::VerifierConfig)
    blocks = Lux.Chain([verifier_block(cfg) for _ in 1:cfg.n_layers]...)

    Lux.@compact(
        elem_emb=Lux.Embedding(cfg.n_elements => cfg.d_single),
        mod_emb=Lux.Embedding(cfg.n_modalities => cfg.d_single),
        poly_emb=Lux.Embedding(cfg.n_polymer_atom_types => cfg.d_single),
        relpos_emb=Lux.Embedding(cfg.n_relpos_buckets => cfg.d_single),
        coord_in=Lux.Dense(3 => cfg.d_single),
        pair_bias=Lux.Dense(cfg.d_single => cfg.n_heads),
        blocks=blocks,
        confidence_head=Lux.Dense(cfg.d_single => 1, sigmoid),
        clash_head=Lux.Dense(cfg.d_single => 1),
        n_heads=cfg.n_heads,
    ) do input
        element_idx, modality_idx, polymer_atom_idx, relpos_idx, coords = input
        n = length(element_idx)

        s = elem_emb(element_idx) .+ mod_emb(modality_idx) .+ poly_emb(polymer_atom_idx) .+ coord_in(coords)
        z = relpos_emb(relpos_idx)
        bias = permutedims(reshape(pair_bias(reshape(z, size(z, 1), :)), n_heads, n, n), (2, 3, 1))

        s, bias = blocks((s, bias))

        confidence = vec(confidence_head(s))
        clash_logit = vec(clash_head(s))
        @return (confidence, clash_logit)
    end
end

end # module
