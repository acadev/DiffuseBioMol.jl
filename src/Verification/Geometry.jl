"""
Pure, differentiable geometric validity checks — no Lux/network dependency,
just coordinate math — so they can serve double duty as (a) hand-verifiable
unit-tested functions and (b) a sampling-time guidance potential whose
gradient w.r.t. coordinates is taken directly with Zygote (Boltz-2's
"steering potential" precedent, see `docs/PLAN.md` Phase 3).

Approximate ideal bond lengths and van der Waals radii below are standard
organic-chemistry textbook values, not a real force field — sufficient for a
v1 differentiable guidance term, not a substitute for MolProbity/OpenMM
relaxation (Layer 3 in `docs/PLAN.md`'s verification stack, not yet
implemented).
"""
module Geometry

using LinearAlgebra: svd, Diagonal, det, cross, dot

export VDW_RADII, BACKBONE_BOND_LENGTHS
export clash_energy, bond_energy, validity_energy, lddt
export clash_count, bond_length_rmsd
export chiral_volume, normalized_chiral_volume, chirality_energy, chirality_count
export kabsch_align, aligned_rmsd

const VDW_RADII = Dict{Symbol,Float64}(
    :C => 1.70, :N => 1.55, :O => 1.52, :S => 1.80, :P => 1.80,
    :H => 1.10, :Fe => 1.94, :Zn => 1.39, :Mg => 1.73, :Ca => 2.31,
    :Na => 2.27, :Cl => 1.75, :Mn => 1.97, :Cu => 1.96, :K => 2.75, :Se => 1.90,
)
const DEFAULT_VDW_RADIUS = 1.70

vdw_radius(element::Symbol) = get(VDW_RADII, element, DEFAULT_VDW_RADIUS)

"""
    BACKBONE_BOND_LENGTHS

Approximate ideal protein-backbone bond lengths (Å): intra-residue N-CA,
CA-C, C-O, CA-CB, and the inter-residue peptide bond C(i)-N(i+1).
"""
const BACKBONE_BOND_LENGTHS = Dict(
    ("N", "CA") => 1.458,
    ("CA", "C") => 1.525,
    ("C", "O") => 1.231,
    ("CA", "CB") => 1.530,
    "peptide" => 1.329,  # C(i) -> N(i+1)
)

"""
    clash_energy(coords, elements, chain_idx, res_index; tol=0.4) -> Real

`coords` is `3 x N`. Sums a smooth repulsive penalty
`relu(r_vdw_i + r_vdw_j - tol - dist_ij)^2` over atom pairs that are *not*
excluded as covalently-local (same chain with `|res_index_i - res_index_j| <= 1`,
i.e. within a residue or its immediate peptide-bonded neighbor — those
distances are governed by bond/angle constraints, not a free-atom VDW clash
term). `tol` softens the cutoff slightly below the literal VDW-radius sum,
since real structures routinely pack atoms a little closer than the textbook
sum suggests.
"""
function clash_energy(coords::AbstractMatrix, elements::AbstractVector{Symbol},
    chain_idx::AbstractVector, res_index::AbstractVector; tol::Real=0.4)
    n = size(coords, 2)
    total = zero(eltype(coords))
    for i in 1:n, j in (i+1):n
        excluded = chain_idx[i] == chain_idx[j] && abs(res_index[i] - res_index[j]) <= 1
        excluded && continue
        dist = sqrt(sum(abs2, coords[:, i] .- coords[:, j]))
        threshold = vdw_radius(elements[i]) + vdw_radius(elements[j]) - tol
        violation = max(threshold - dist, zero(dist))
        total = total + violation^2
    end
    total
end

"""
    bond_energy(coords, bonded_pairs) -> Real

`bonded_pairs` is a vector of `(i, j, ideal_length)`. Sums
`(dist_ij - ideal_length)^2` — a harmonic bond-length restraint.
"""
function bond_energy(coords::AbstractMatrix, bonded_pairs::AbstractVector{<:Tuple})
    total = zero(eltype(coords))
    for (i, j, ideal) in bonded_pairs
        dist = sqrt(sum(abs2, coords[:, i] .- coords[:, j]))
        total = total + (dist - ideal)^2
    end
    total
end

"""
    clash_count(coords, elements, chain_idx, res_index; tol=0.4) -> Int

The number of atom pairs with a nonzero clash violation (same exclusion rule
as `clash_energy`) — a more directly interpretable reporting metric for
benchmarks than the squared-penalty energy ("3 clashing pairs" vs. an
arbitrary energy unit).
"""
function clash_count(coords::AbstractMatrix, elements::AbstractVector{Symbol},
    chain_idx::AbstractVector, res_index::AbstractVector; tol::Real=0.4)
    n = size(coords, 2)
    count = 0
    for i in 1:n, j in (i+1):n
        excluded = chain_idx[i] == chain_idx[j] && abs(res_index[i] - res_index[j]) <= 1
        excluded && continue
        dist = sqrt(sum(abs2, coords[:, i] .- coords[:, j]))
        threshold = vdw_radius(elements[i]) + vdw_radius(elements[j]) - tol
        dist < threshold && (count += 1)
    end
    count
end

"""
    bond_length_rmsd(coords, bonded_pairs) -> Real

`sqrt(bond_energy(coords, bonded_pairs) / length(bonded_pairs))` — bond
deviation in Å (the natural unit), rather than `bond_energy`'s squared-error
sum, for benchmark reporting. Returns `0.0` for an empty bond list.
"""
function bond_length_rmsd(coords::AbstractMatrix, bonded_pairs::AbstractVector{<:Tuple})
    isempty(bonded_pairs) && return zero(eltype(coords))
    sqrt(bond_energy(coords, bonded_pairs) / length(bonded_pairs))
end

"""
    chiral_volume(coords, ca, n, c, cb) -> Real

Signed volume `dot(n - ca, cross(c - ca, cb - ca))` of the tetrahedron formed
by a candidate chiral center `ca` and its three substituents `n`, `c`, `cb`
(atom indices into `coords`, `3 x N`). Positive for correct L-amino-acid
backbone stereochemistry at a residue's CA — confirmed empirically here
(not a memorized textbook constant) across 164 real CA centers spanning
1CRN/1UBQ/5PTI: consistently in `[1.79, 3.13]`, never crossing zero. A
mirrored (D-form) or badly distorted center drives this toward zero or
negative.
"""
function chiral_volume(coords::AbstractMatrix, ca::Int, n::Int, c::Int, cb::Int)
    b1 = coords[:, n] .- coords[:, ca]
    b2 = coords[:, c] .- coords[:, ca]
    b3 = coords[:, cb] .- coords[:, ca]
    dot(b1, cross(b2, b3))
end

"""
    normalized_chiral_volume(coords, ca, n, c, cb; eps=1e-6) -> Real

`chiral_volume(...)` divided by `norm(b1)*norm(b2)*norm(b3)` (the three
substituent-offset vectors), bounding it to `[-1, 1]` by the scalar-triple-
-product/Cauchy-Schwarz inequality — a dimensionless handedness measure,
independent of how far apart the four atoms happen to be. Confirmed
empirically across the same 164 real CA centers (1CRN/1UBQ/5PTI) used for
`chiral_volume`: consistently in `[0.53, 0.87]`.

This normalization matters for sampling-time guidance specifically: early in
a flow-matching trajectory, coordinates can sit far from any realistic
molecular scale, and the raw `chiral_volume` grows as the *cube* of that
scale — so a fixed guidance step size that's stable for `clash_energy`/
`bond_energy` (whose violations are capped by physical distances) can send
`chirality_energy`'s gradient into a runaway explosion. The normalized
version's gradient stays bounded regardless of coordinate scale, which is
what actually makes it safe to use as a `post_step` guidance term.
"""
function normalized_chiral_volume(coords::AbstractMatrix, ca::Int, n::Int, c::Int, cb::Int; eps::Real=1e-6)
    b1 = coords[:, n] .- coords[:, ca]
    b2 = coords[:, c] .- coords[:, ca]
    b3 = coords[:, cb] .- coords[:, ca]
    dot(b1, cross(b2, b3)) / (sqrt(sum(abs2, b1)) * sqrt(sum(abs2, b2)) * sqrt(sum(abs2, b3)) + eps)
end

"""
    chirality_energy(coords, centers; margin=0.3) -> Real

`centers` is a vector of `(ca, n, c, cb)` index tuples (see
`Verification.chiral_centers`). Sums `relu(margin - normalized_chiral_volume(...))^2`
over each center: zero once a center's normalized handedness comfortably
clears `margin`, growing as it shrinks toward zero or flips sign.
`margin=0.3` sits well below the empirical real-structure range (~0.53-0.87,
see `normalized_chiral_volume`), so ordinary structural flexibility isn't
penalized — only genuine wrong-handedness or near-planarity is. Built on the
normalized (not raw) volume so the term stays numerically stable as a
sampling-time guidance potential regardless of coordinate scale.
"""
function chirality_energy(coords::AbstractMatrix, centers::AbstractVector{<:NTuple{4,Int}}; margin::Real=0.3)
    total = zero(eltype(coords))
    for (ca, n, c, cb) in centers
        violation = max(margin - normalized_chiral_volume(coords, ca, n, c, cb), zero(eltype(coords)))
        total = total + violation^2
    end
    total
end

"""
    chirality_count(coords, centers; margin=0.0) -> Int

Number of centers whose normalized chiral volume falls at or below `margin`
(default `0.0`, i.e. actually wrong-handed or degenerate) — an interpretable
reporting metric for benchmarks, analogous to `clash_count` vs.
`clash_energy`.
"""
function chirality_count(coords::AbstractMatrix, centers::AbstractVector{<:NTuple{4,Int}}; margin::Real=0.0)
    count(t -> normalized_chiral_volume(coords, t...) <= margin, centers)
end

"""
    validity_energy(coords, elements, chain_idx, res_index, bonded_pairs,
                     chiral_centers=[]; clash_weight=1.0, bond_weight=1.0,
                     chirality_weight=1.0, tol=0.4, chirality_margin=0.3) -> Real

Combined differentiable physical-validity penalty: `clash_weight *
clash_energy(...) + bond_weight * bond_energy(...) + chirality_weight *
chirality_energy(...)`. This is the function whose negative gradient (w.r.t.
`coords`) is used as a sampling-time guidance correction. `chiral_centers`
defaults to empty (no chirality term) so existing callers that only pass
`bonded_pairs` are unaffected.
"""
function validity_energy(coords::AbstractMatrix, elements::AbstractVector{Symbol},
    chain_idx::AbstractVector, res_index::AbstractVector, bonded_pairs::AbstractVector{<:Tuple},
    chiral_centers::AbstractVector{<:NTuple{4,Int}}=NTuple{4,Int}[];
    clash_weight::Real=1.0, bond_weight::Real=1.0, chirality_weight::Real=1.0, tol::Real=0.4, chirality_margin::Real=0.3)
    clash_weight * clash_energy(coords, elements, chain_idx, res_index; tol=tol) +
        bond_weight * bond_energy(coords, bonded_pairs) +
        chirality_weight * chirality_energy(coords, chiral_centers; margin=chirality_margin)
end

"""
    lddt(coords_model, coords_ref; cutoff=15.0, thresholds=(0.5, 1.0, 2.0, 4.0)) -> Vector{Float64}

Per-atom local distance difference test (lDDT, as used for AlphaFold-style
pLDDT confidence labels): for each atom `i`, considers every other atom `j`
within `cutoff` Å in the *reference* structure, and scores the fraction of
those `(i, j)` distances that are preserved (within each tolerance in
`thresholds`) in the model structure, averaged over thresholds. Returns
values in `[0, 1]`; an atom with no neighbors within `cutoff` scores `1.0`
(vacuously — there is nothing to get wrong).
"""
function lddt(coords_model::AbstractMatrix, coords_ref::AbstractMatrix;
    cutoff::Real=15.0, thresholds=(0.5, 1.0, 2.0, 4.0))
    n = size(coords_ref, 2)
    scores = ones(Float64, n)
    for i in 1:n
        neighbors = Int[]
        for j in 1:n
            j == i && continue
            d_ref = sqrt(sum(abs2, coords_ref[:, i] .- coords_ref[:, j]))
            d_ref <= cutoff && push!(neighbors, j)
        end
        isempty(neighbors) && continue
        preserved = 0.0
        for j in neighbors
            d_ref = sqrt(sum(abs2, coords_ref[:, i] .- coords_ref[:, j]))
            d_model = sqrt(sum(abs2, coords_model[:, i] .- coords_model[:, j]))
            delta = abs(d_model - d_ref)
            preserved += count(t -> delta <= t, thresholds) / length(thresholds)
        end
        scores[i] = preserved / length(neighbors)
    end
    scores
end

"""
    kabsch_align(coords, reference) -> Matrix{Float64}  # 3 x N

Rigidly superposes `coords` (`3 x N`) onto `reference` (`3 x N`) via the
Kabsch algorithm (the optimal least-squares rotation + translation, with a
reflection correction so the result is never mirrored) and returns the
superposed `coords`.

Use this — never a raw coordinate-wise difference — when comparing a
sampled/generated structure against a reference: nothing in this codebase
guarantees the two share a common global frame. `Sampling.FlowMatching`'s
training-time SE(3) augmentation (`Sampling.Augmentation`) deliberately
randomizes the orientation the network is shown, precisely so the network
isn't forced to memorize the arbitrary frame of any one structure — which
means a trained model's *unconditional* samples come out in an essentially
arbitrary frame too, by design, not as a bug. A raw difference conflates
"wrong rigid pose" with "wrong shape"; superposing first isolates the latter,
which is what's actually informative.
"""
function kabsch_align(coords::AbstractMatrix, reference::AbstractMatrix)
    centroid_c = vec(sum(coords; dims=2)) ./ size(coords, 2)
    centroid_r = vec(sum(reference; dims=2)) ./ size(reference, 2)
    p = Float64.(coords) .- centroid_c
    q = Float64.(reference) .- centroid_r

    f = svd(p * q')
    d = sign(det(f.V * f.U'))
    r = f.V * Diagonal([1.0, 1.0, d]) * f.U'
    r * p .+ centroid_r
end

"""
    aligned_rmsd(coords, reference) -> Real

RMSD between `coords` and `reference` *after* optimal rigid superposition
(`kabsch_align`) — the standard way to compare two structures' shape
independent of their relative orientation/position. See `kabsch_align`'s
docstring for why this, not a raw coordinate-wise RMSE, is the right
comparison here.
"""
function aligned_rmsd(coords::AbstractMatrix, reference::AbstractMatrix)
    aligned = kabsch_align(coords, reference)
    sqrt(sum(abs2, aligned .- Float64.(reference)) / size(reference, 2))
end

end # module
