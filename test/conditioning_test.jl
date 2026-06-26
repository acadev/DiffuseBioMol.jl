using Test
using DiffuseBioMol
using Random, Zygote, Optimisers

@testset "Conditioning: feature matrix construction" begin
    n = 5
    c = no_constraints(n)
    feats = constraint_features(c)
    @test size(feats) == (N_COND_FEATURES, n)
    @test all(feats .== 0)

    c2 = AtomConstraints(
        BitVector([true, false, false, false, false]),
        BitVector([true, false, false, false, false]),
        Float32[0.8, 0, 0, 0, 0],
        BitVector([true, false, false, false, false]),
        zeros(Float32, 3, n),
    )
    full = constraint_features(c2)
    @test full[1, 1] == 1  # is_fixed
    @test full[2, 1] == 1  # is_hotspot
    @test full[3, 1] == 0.8f0  # rasa_target
    @test full[4, 1] == 1  # has_rasa
    @test all(full[:, 2:end] .== 0)

    dropped = constraint_features(c2; drop_soft=true)
    @test dropped[1, 1] == 1       # is_fixed still shown
    @test all(dropped[2:4, :] .== 0)  # soft conditioning zeroed
end

@testset "Conditioning: motif clamping during sampling" begin
    rng = Random.Xoshiro(7)

    res1 = ParsedResidue("A", 1, "ALA", PROTEIN, Dict("N" => (0.0, 0.0, 0.0), "CA" => (1.5, 0.0, 0.0), "C" => (2.9, 0.4, 0.0), "O" => (3.4, 1.5, 0.0), "CB" => (1.9, -0.7, 1.2)))
    res2 = ParsedResidue("A", 2, "GLY", PROTEIN, Dict("N" => (3.6, -0.6, -0.3), "CA" => (5.0, -0.5, -0.2), "C" => (5.6, -1.9, -0.4), "O" => (5.0, -2.9, -0.1)))
    res3 = ParsedResidue("A", 3, "ALA", PROTEIN, Dict("N" => (6.9, -1.9, -0.8), "CA" => (7.6, -3.2, -1.0), "C" => (9.1, -3.0, -1.2), "O" => (9.7, -1.9, -1.1), "CB" => (7.2, -4.0, -2.3)))

    tokens = tokenize_structure([res1, res2, res3])
    feat = featurize(tokens)
    relpos = relpos_buckets(feat)
    n = length(tokens)
    x1 = Matrix{Float64}(undef, 3, n)
    for (i, t) in enumerate(tokens)
        x1[:, i] .= t.coord
    end

    # Treat residue 1 (the first 5 atoms: N, CA, C, O, CB of ALA) as a fixed
    # motif that must be scaffolded exactly; residues 2-3 are to be generated.
    is_fixed = falses(n)
    is_fixed[1:5] .= true
    fixed_coord = zeros(Float32, 3, n)
    fixed_coord[:, is_fixed] .= Float32.(x1[:, is_fixed])

    constraints = AtomConstraints(is_fixed, is_fixed, zeros(Float32, n), falses(n), fixed_coord)
    cond_features = constraint_features(constraints)

    cfg = ModelConfig(d_single=16, d_pair=8, n_heads=4, n_pairformer_layers=2, n_dit_layers=2, d_time=16)
    model = build_model(cfg)
    ps, st = DiffuseBioMol.Model.Network.Lux.setup(rng, model)

    x_sample, _ = sample_flow(
        model, ps, st, feat, relpos, cond_features, rng;
        n_steps=10, is_fixed=is_fixed, fixed_coord=fixed_coord,
    )

    @test size(x_sample) == size(x1)
    @test all(isfinite, x_sample)
    @test x_sample[:, is_fixed] ≈ fixed_coord[:, is_fixed]  # motif atoms stayed exactly put
    @test !(x_sample[:, .!is_fixed] ≈ fixed_coord[:, .!is_fixed])  # other atoms actually moved/generated
end

@testset "Conditioning: training excludes fixed atoms from the loss" begin
    rng = Random.Xoshiro(11)
    res1 = ParsedResidue("A", 1, "ALA", PROTEIN, Dict("N" => (0.0, 0.0, 0.0), "CA" => (1.5, 0.0, 0.0), "C" => (2.9, 0.4, 0.0), "O" => (3.4, 1.5, 0.0), "CB" => (1.9, -0.7, 1.2)))
    tokens = tokenize_structure([res1])
    feat = featurize(tokens)
    n = length(tokens)
    x1 = Matrix{Float64}(undef, 3, n)
    for (i, t) in enumerate(tokens)
        x1[:, i] .= t.coord
    end

    is_fixed = trues(n)  # the whole (tiny) structure is "given"
    fixed_coord = Float32.(x1)
    cond_features = constraint_features(AtomConstraints(is_fixed, is_fixed, zeros(Float32, n), falses(n), fixed_coord))

    example = prepare_training_example(feat, x1, cond_features, rng; is_fixed=is_fixed, fixed_coord=fixed_coord)
    @test count(example.mask) == 0  # nothing left to predict
    @test all(example.target_v .== 0)
end

@testset "Conditioning: CoM guidance pulls a chain toward its target offset" begin
    rng = Random.Xoshiro(0)
    # Two single-atom "chains" far apart; pull chain B toward a fixed offset from chain A.
    chain_idx = [1, 2]
    x = Float32[0.0 20.0; 0.0 0.0; 0.0 0.0]  # chain A at origin, chain B at (20,0,0)
    target_offset = (5.0f0, 0.0f0, 0.0f0)
    com_constraint = ChainCoMConstraint(2, 1, target_offset, 1.0f0)

    apply_com_guidance!(x, chain_idx, [com_constraint], 1.0)
    # With weight*dt = 1, the proportional pull moves chain B exactly onto the target.
    @test x[:, 2] ≈ Float32[5.0, 0.0, 0.0]
    @test x[:, 1] == Float32[0.0, 0.0, 0.0]  # reference chain untouched
end
