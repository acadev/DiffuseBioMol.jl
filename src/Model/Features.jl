"""
Turns a `Tokenizer.AtomToken` sequence into the categorical feature arrays the
embedding layer consumes. Polymer atoms (protein/RNA/DNA) get an identity
embedding keyed on `(res_name, atom_name)` from the same closed vocabulary as
`AtomVocab.RESIDUE_ATOMS`, since that's a small, fixed set; non-polymer atoms
(ligand/ion/PTM) have no closed naming vocabulary, so they fall back to
element + modality only. This is a deliberate v1 simplification — extending
ligand atoms to richer typing (formal charge, hybridization, ring membership)
is later work, not required to validate the core flow-matching architecture.
"""
module Features

using ..AtomVocab
using ..Tokenizer: AtomToken

export TokenFeatures, featurize, relpos_buckets, target_coordinates
export N_ELEMENTS, N_MODALITIES, N_POLYMER_ATOM_TYPES, N_RELPOS_BUCKETS

const COMMON_ELEMENTS = [:C, :N, :O, :S, :P, :H, :Fe, :Zn, :Mg, :Ca, :Na, :Cl, :Mn, :Cu, :K, :Se]
const ELEMENT_INDEX = Dict(e => i for (i, e) in enumerate(COMMON_ELEMENTS))
const N_ELEMENTS = length(COMMON_ELEMENTS) + 1  # +1 catch-all "other" bucket

element_index(e::Symbol) = get(ELEMENT_INDEX, e, N_ELEMENTS)

const MODALITIES = (PROTEIN, RNA, DNA, LIGAND, ION, PTM)
const N_MODALITIES = length(MODALITIES)

modality_index(m::Modality) = Int(m) + 1  # enums are 0-indexed

const POLYMER_ATOM_VOCAB = let
    pairs = Tuple{String,String}[]
    for (res_name, atoms) in AtomVocab.RESIDUE_ATOMS
        for atom_name in atoms
            push!(pairs, (res_name, atom_name))
        end
    end
    Dict(p => i for (i, p) in enumerate(pairs))
end
const N_POLYMER_ATOM_TYPES = length(POLYMER_ATOM_VOCAB) + 1  # +1 catch-all for non-polymer atoms

polymer_atom_index(res_name::AbstractString, atom_name::AbstractString) =
    get(POLYMER_ATOM_VOCAB, (res_name, atom_name), N_POLYMER_ATOM_TYPES)

"""
    TokenFeatures

Parallel categorical-index arrays (1-indexed, ready for embedding lookups) plus
bookkeeping needed to build relative-position pair features and to mask
virtual (padding) atoms out of losses/attention pooling.
"""
struct TokenFeatures
    element_idx::Vector{Int}
    modality_idx::Vector{Int}
    polymer_atom_idx::Vector{Int}
    chain_idx::Vector{Int}      # integer-coded chain id, for relative-position pair features
    res_index::Vector{Int}      # residue index within chain, for relative-position pair features
    is_virtual::Vector{Bool}
end

"""
    featurize(tokens::Vector{AtomToken}) -> TokenFeatures
"""
function featurize(tokens::Vector{AtomToken})::TokenFeatures
    n = length(tokens)
    chain_lookup = Dict{String,Int}()
    chain_idx = Vector{Int}(undef, n)
    for (i, t) in enumerate(tokens)
        chain_idx[i] = get!(chain_lookup, t.chain_id, length(chain_lookup) + 1)
    end

    TokenFeatures(
        [element_index(t.element) for t in tokens],
        [modality_index(t.modality) for t in tokens],
        [polymer_atom_index(t.res_name, t.atom_name) for t in tokens],
        chain_idx,
        [t.res_index for t in tokens],
        [t.is_virtual for t in tokens],
    )
end

const RELPOS_CLAMP = 32
const N_RELPOS_BUCKETS = 2 * RELPOS_CLAMP + 1 + 1  # +1 for the cross-chain bucket

"""
    relpos_buckets(feat::TokenFeatures) -> Matrix{Int}  # N x N, 1-indexed

Pairwise relative-position bucket for the encoder's initial pair
representation: atoms in different chains all share one bucket (chain
topology, not residue distance, is what matters across chains); atoms in the
same chain are bucketed by their clamped residue-index difference.
"""
function relpos_buckets(feat::TokenFeatures)::Matrix{Int}
    n = length(feat.chain_idx)
    buckets = Matrix{Int}(undef, n, n)
    cross_chain_bucket = N_RELPOS_BUCKETS
    for i in 1:n, j in 1:n
        if feat.chain_idx[i] != feat.chain_idx[j]
            buckets[i, j] = cross_chain_bucket
        else
            d = clamp(feat.res_index[i] - feat.res_index[j], -RELPOS_CLAMP, RELPOS_CLAMP)
            buckets[i, j] = d + RELPOS_CLAMP + 1
        end
    end
    buckets
end

"""
    target_coordinates(tokens::Vector{AtomToken}) -> Matrix{Float64}  # 3 x N

Ground-truth coordinate matrix for `FlowMatching.prepare_training_example`.
Virtual (padding) atoms have no real coordinate; their column is filled with
zeros here as a placeholder — harmless, since `prepare_training_example`
overwrites those columns with the prior sample before computing the flow
target.
"""
function target_coordinates(tokens::Vector{AtomToken})::Matrix{Float64}
    n = length(tokens)
    x1 = zeros(Float64, 3, n)
    for (i, t) in enumerate(tokens)
        t.coord === nothing || (x1[:, i] .= t.coord)
    end
    x1
end

end # module
