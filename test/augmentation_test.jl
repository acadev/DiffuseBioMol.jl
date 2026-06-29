using Test
using DiffuseBioMol
using Random, LinearAlgebra

@testset "Augmentation.random_rotation: produces proper rotations" begin
    rng = Random.Xoshiro(1)
    for _ in 1:20
        R = random_rotation(rng)
        @test size(R) == (3, 3)
        @test isapprox(R' * R, I; atol=1e-10)  # orthogonal
        @test isapprox(det(R), 1.0; atol=1e-10)  # proper (not a reflection)
    end
end

@testset "Augmentation.apply_se3_transform: preserves pairwise distances (isometry)" begin
    rng = Random.Xoshiro(2)
    coords = Float32[0.0 1.5 2.9 5.0; 0.0 0.0 0.4 1.0; 0.0 0.0 0.0 -0.3]
    mask = trues(4)
    R, centroid = random_se3_transform(coords, mask, rng)
    transformed = apply_se3_transform(coords, R, centroid)

    n = size(coords, 2)
    for i in 1:n, j in (i+1):n
        d_orig = norm(coords[:, i] .- coords[:, j])
        d_new = norm(transformed[:, i] .- transformed[:, j])
        @test isapprox(d_orig, d_new; atol=1e-4)
    end

    # Centroid of the transformed (mask-selected) atoms is at the origin.
    com = vec(sum(transformed[:, mask]; dims=2)) ./ count(mask)
    @test isapprox(com, zeros(3); atol=1e-4)
end

@testset "Augmentation: random_se3_transform applied consistently to two related matrices" begin
    rng = Random.Xoshiro(3)
    x1 = Float32[0.0 3.0; 0.0 0.0; 0.0 0.0]
    fixed_coord = x1[:, [1]]  # the first atom is "fixed", sharing x1's frame

    R, centroid = random_se3_transform(x1, trues(2), rng)
    x1_t = apply_se3_transform(x1, R, centroid)
    fixed_t = apply_se3_transform(fixed_coord, R, centroid)

    # The transformed fixed coordinate must still coincide with the
    # transformed x1's corresponding column (same rigid motion applied).
    @test isapprox(fixed_t[:, 1], x1_t[:, 1]; atol=1e-4)
end

@testset "FlowMatching.sample_flow: fixed-atom output is exact in the caller's original frame" begin
    rng = Random.Xoshiro(4)
    res1 = ParsedResidue("A", 1, "ALA", PROTEIN, Dict("N" => (10.0, 20.0, -5.0), "CA" => (11.5, 20.0, -5.0), "C" => (12.9, 20.4, -5.0), "O" => (13.4, 21.5, -5.0), "CB" => (11.9, 19.3, -3.8)))
    res2 = ParsedResidue("A", 2, "GLY", PROTEIN, Dict("N" => (13.6, 19.4, -5.3), "CA" => (15.0, 19.5, -5.2), "C" => (15.6, 18.1, -5.4), "O" => (15.0, 17.1, -5.1)))
    tokens = tokenize_structure([res1, res2])
    feat = featurize(tokens)
    relpos = relpos_buckets(feat)
    n = length(tokens)
    x1 = Matrix{Float64}(undef, 3, n)
    for (i, t) in enumerate(tokens)
        x1[:, i] .= t.coord
    end

    is_fixed = falses(n)
    is_fixed[1:5] .= true  # residue 1 (far from the origin) is the fixed motif
    fixed_coord = zeros(Float32, 3, n)
    fixed_coord[:, is_fixed] .= Float32.(x1[:, is_fixed])
    cond_features = constraint_features(AtomConstraints(is_fixed, is_fixed, zeros(Float32, n), falses(n), fixed_coord))

    cfg = ModelConfig(d_single=16, d_pair=8, n_heads=4, n_pairformer_layers=2, n_dit_layers=2, d_time=16)
    model = build_model(cfg)
    ps, st = DiffuseBioMol.Model.Network.Lux.setup(rng, model)

    x_sample, _ = sample_flow(model, ps, st, feat, relpos, cond_features, rng; n_steps=8, is_fixed=is_fixed, fixed_coord=fixed_coord)

    # Sampling happens in a centered frame internally, but the returned
    # coordinates must be shifted back to fixed_coord's original (far from
    # the origin) frame, not left near zero.
    @test x_sample[:, is_fixed] ≈ fixed_coord[:, is_fixed]
    @test all(isfinite, x_sample)
end
