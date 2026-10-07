"""
Phase 3: the three-layer physical-correctness stack from `docs/PLAN.md`.

- Layer 1 (`Geometry`): differentiable in-loop guidance — pure coordinate
  math (clash, bond-length, and chirality penalties), usable both as
  hand-verifiable unit tests and as a sampling-time correction via
  `validity_guidance_step` (plugs into `Sampling.FlowMatching.sample_flow`'s
  `post_step` hook). The chirality term's sign convention
  (`Geometry.chiral_volume`) was derived empirically from real structures
  (1CRN/1UBQ/5PTI), not a memorized textbook constant — see its docstring.
- Layer 2 (`Verifier`): the learned verifier head — a small network scoring
  confidence (trained against `Geometry.lddt` labels) and clash likelihood
  (trained against `Geometry.clash_energy`-derived labels) without invoking
  external tools.
- Layer 3 (expensive, ground-truth external verification — MolProbity,
  OpenMM/Amber relaxation, self-consistency refold) is not yet implemented;
  bridging it in will need a Python interop story (PythonCall.jl), per
  `docs/PLAN.md`'s accepted dependency.

`backbone_bonds`/`chiral_centers` are the one place this module depends on
`Tokenizer`/`AtomVocab` (to derive the bonded-pair/chiral-center lists from
real residue/atom-name information) — everything else in
`Geometry`/`Verifier` is deliberately generic over plain arrays, decoupled
from `Model`'s types (see those modules' docstrings for why).
"""
module Verification

using ..AtomVocab: PROTEIN
using ..Tokenizer: AtomToken
using Zygote

include("Geometry.jl")
include("Verifier.jl")

using .Geometry
using .Verifier

export Geometry, Verifier
export VDW_RADII, BACKBONE_BOND_LENGTHS, clash_energy, bond_energy, validity_energy, lddt, clash_count, bond_length_rmsd
export chiral_volume, normalized_chiral_volume, chirality_energy, chirality_count
export kabsch_align, aligned_rmsd
export VerifierConfig, build_verifier
export backbone_bonds, chiral_centers, validity_guidance_step, verifier_loss

"""
    backbone_bonds(tokens::Vector{AtomToken}) -> Vector{Tuple{Int,Int,Float64}}

Derive the standard protein-backbone bonded-pair list (intra-residue N-CA,
CA-C, C-O, CA-CB, plus inter-residue peptide bonds C(i)-N(i+1)) from a token
sequence, for use with `Geometry.bond_energy`/`validity_energy`. Only
`PROTEIN`-modality atoms are considered (RNA/DNA backbone bonds are future
work, see `docs/PLAN.md`).
"""
function backbone_bonds(tokens::Vector{AtomToken})::Vector{Tuple{Int,Int,Float64}}
    residue_atoms = Dict{Tuple{String,Int},Dict{String,Int}}()
    for (i, t) in enumerate(tokens)
        t.modality == PROTEIN || continue
        key = (t.chain_id, t.res_index)
        d = get!(residue_atoms, key, Dict{String,Int}())
        d[t.atom_name] = i
    end

    bonds = Tuple{Int,Int,Float64}[]
    for (key, len) in BACKBONE_BOND_LENGTHS
        key isa Tuple || continue
        a, b = key
        for atoms in values(residue_atoms)
            haskey(atoms, a) && haskey(atoms, b) && push!(bonds, (atoms[a], atoms[b], len))
        end
    end

    peptide_len = BACKBONE_BOND_LENGTHS["peptide"]
    for ((chain, resi), atoms) in residue_atoms
        haskey(atoms, "C") || continue
        next_atoms = get(residue_atoms, (chain, resi + 1), nothing)
        next_atoms === nothing && continue
        haskey(next_atoms, "N") || continue
        push!(bonds, (atoms["C"], next_atoms["N"], peptide_len))
    end
    bonds
end

"""
    chiral_centers(tokens::Vector{AtomToken}) -> Vector{NTuple{4,Int}}

Derive `(ca, n, c, cb)` index quadruples for each residue with all four of
N/CA/C/CB present, for use with `Geometry.chirality_energy`/`chiral_volume`.
Glycine (no CB) is skipped — it has no chiral CA. Only `PROTEIN`-modality
atoms are considered (RNA/DNA have no analogous check here; future work
alongside `backbone_bonds`' RNA/DNA gap, see `docs/PLAN.md`).
"""
function chiral_centers(tokens::Vector{AtomToken})::Vector{NTuple{4,Int}}
    residue_atoms = Dict{Tuple{String,Int},Dict{String,Int}}()
    for (i, t) in enumerate(tokens)
        t.modality == PROTEIN || continue
        key = (t.chain_id, t.res_index)
        d = get!(residue_atoms, key, Dict{String,Int}())
        d[t.atom_name] = i
    end

    centers = NTuple{4,Int}[]
    for atoms in values(residue_atoms)
        if haskey(atoms, "N") && haskey(atoms, "CA") && haskey(atoms, "C") && haskey(atoms, "CB")
            push!(centers, (atoms["CA"], atoms["N"], atoms["C"], atoms["CB"]))
        end
    end
    centers
end

"""
    validity_guidance_step(elements, chain_idx, res_index, bonded_pairs, chiral_centers=[];
                            step_size=0.05) -> (x, t) -> x

Returns a closure suitable for `Sampling.FlowMatching.sample_flow`'s
`post_step` callback: a single gradient-descent step on
`Geometry.validity_energy` w.r.t. the current coordinates, computed with
Zygote. `step_size` trades off correction strength against the risk of the
guidance term fighting the learned velocity field too aggressively.
`chiral_centers` defaults to empty (no chirality correction) so existing
callers are unaffected.
"""
function validity_guidance_step(elements::AbstractVector{Symbol}, chain_idx::AbstractVector, res_index::AbstractVector,
    bonded_pairs::AbstractVector{<:Tuple}, chiral_centers::AbstractVector{<:NTuple{4,Int}}=NTuple{4,Int}[];
    step_size::Real=0.05, clash_weight::Real=1.0, bond_weight::Real=1.0, chirality_weight::Real=1.0, chirality_margin::Real=0.3)
    return (x, _t) -> begin
        grad = only(Zygote.gradient(
            c -> validity_energy(c, elements, chain_idx, res_index, bonded_pairs, chiral_centers;
                clash_weight=clash_weight, bond_weight=bond_weight, chirality_weight=chirality_weight, chirality_margin=chirality_margin),
            x,
        ))
        x .- Float32(step_size) .* grad
    end
end

"""
    verifier_loss(model, ps, st, element_idx, modality_idx, polymer_atom_idx, relpos_idx,
                   coords, confidence_target, clash_target) -> (loss, st)

Joint training loss for the learned verifier head: MSE on the confidence
score against `confidence_target` (e.g. `Geometry.lddt` labels) plus
binary-cross-entropy-with-logits on the clash head against `clash_target`
(e.g. a thresholded `Geometry.clash_energy`-derived per-atom label).
"""
function verifier_loss(model, ps, st, element_idx, modality_idx, polymer_atom_idx, relpos_idx::AbstractMatrix,
    coords::AbstractMatrix, confidence_target::AbstractVector, clash_target::AbstractVector)
    (confidence, clash_logit), st = model((element_idx, modality_idx, polymer_atom_idx, relpos_idx, coords), ps, st)
    confidence_loss = sum(abs2, confidence .- confidence_target) / length(confidence_target)
    # Binary cross-entropy with logits, numerically stable form.
    clash_loss = sum(
        max.(clash_logit, 0) .- clash_logit .* clash_target .+ log1p.(exp.(-abs.(clash_logit))),
    ) / length(clash_target)
    (confidence_loss + clash_loss, st)
end

end # module
