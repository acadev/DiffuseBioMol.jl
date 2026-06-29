"""
Random SE(3) (rotation + translation) augmentation for training.

`Model.Network.build_model`'s `coord_in = Dense(3 => d_single)` consumes raw
Cartesian coordinates directly — there is no SE(3)-equivariant architecture
here (no mature Julia equivalent of e3nn exists, see `docs/PLAN.md`'s
engineering-plan section), so the network has no structural guarantee of
producing the same output (up to the same rigid motion) for a rotated/
translated copy of the same structure. Without correcting for this, the
network would have to learn invariance purely from how many distinct global
orientations happen to appear across the training set, which is a much
harder and more data-hungry thing to learn than the actual task.

This is the standard fix used by AF2/RFdiffusion-style pipelines:
translation invariance is obtained *exactly*, for free, by always centering
a structure on the centroid of its real atoms before it's shown to the
network (so the training distribution simply never contains translation
variance to overfit to); rotation invariance has no such free lunch (there's
no canonical orientation to center on), so it's instead approximated
statistically by drawing a fresh uniformly-random rotation for every
training example.
"""
module Augmentation

using LinearAlgebra: qr, Diagonal, det
using Random: AbstractRNG

export random_rotation, random_se3_transform, apply_se3_transform, random_se3_augment

"""
    random_rotation(rng) -> Matrix{Float64}

A uniformly random (Haar-distributed) proper 3x3 rotation matrix (`det = 1`):
QR-decompose a random Gaussian 3x3 matrix, then correct signs so the result
is never a reflection (`det = -1`).
"""
function random_rotation(rng::AbstractRNG)
    q, r = qr(randn(rng, 3, 3))
    d = [sign(r[i, i]) for i in 1:3]
    q = Matrix(q) * Diagonal(d)
    det(q) < 0 && (q[:, end] .*= -1)
    q
end

"""
    random_se3_transform(coords, mask, rng) -> (R, centroid)

Draws a `(rotation, centroid)` pair: `centroid` is the centroid of the
`mask`-selected columns of `coords` (`3 x N`), `R` is a fresh
`random_rotation`. Exposed separately from `apply_se3_transform` so the
*same* draw can be applied consistently to multiple coordinate matrices that
belong to the same structure (e.g. ground-truth coordinates and motif/
hotspot fixed coordinates, which must stay mutually consistent under the
transform).
"""
function random_se3_transform(coords::AbstractMatrix, mask::AbstractVector{Bool}, rng::AbstractRNG)
    centroid = vec(sum(view(coords, :, mask); dims=2)) ./ count(mask)
    (random_rotation(rng), Float64.(centroid))
end

"""
    apply_se3_transform(coords, R, centroid) -> Matrix{Float32}

`R * (coords .- centroid)`, i.e. center on `centroid` then rotate by `R`.
"""
function apply_se3_transform(coords::AbstractMatrix, R::AbstractMatrix, centroid::AbstractVector)
    Float32.(R * (Float64.(coords) .- centroid))
end

"""
    random_se3_augment(coords, mask, rng) -> Matrix{Float32}

Convenience wrapper: `apply_se3_transform(coords, random_se3_transform(coords, mask, rng)...)`.
Use `random_se3_transform`/`apply_se3_transform` directly instead when more
than one coordinate matrix from the same structure needs the same transform.
"""
function random_se3_augment(coords::AbstractMatrix, mask::AbstractVector{Bool}, rng::AbstractRNG)
    R, centroid = random_se3_transform(coords, mask, rng)
    apply_se3_transform(coords, R, centroid)
end

end # module
