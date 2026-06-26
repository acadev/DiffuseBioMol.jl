using Test
using DiffuseBioMol
using Random, Zygote, Optimisers

@testset "Model+Sampling: architecture smoke test (overfit one example)" begin
    rng = Random.Xoshiro(42)

    res1 = ParsedResidue("A", 1, "ALA", PROTEIN, Dict("N" => (0.0, 0.0, 0.0), "CA" => (1.5, 0.0, 0.0), "C" => (2.9, 0.4, 0.0), "O" => (3.4, 1.5, 0.0), "CB" => (1.9, -0.7, 1.2)))
    res2 = ParsedResidue("A", 2, "GLY", PROTEIN, Dict("N" => (3.6, -0.6, -0.3), "CA" => (5.0, -0.5, -0.2), "C" => (5.6, -1.9, -0.4), "O" => (5.0, -2.9, -0.1)))
    res3 = ParsedResidue("A", 3, "ALA", PROTEIN, Dict("N" => (6.9, -1.9, -0.8), "CA" => (7.6, -3.2, -1.0), "C" => (9.1, -3.0, -1.2), "O" => (9.7, -1.9, -1.1), "CB" => (7.2, -4.0, -2.3)))

    tokens = tokenize_structure([res1, res2, res3])
    feat = featurize(tokens)
    relpos = relpos_buckets(feat)
    @test count(feat.is_virtual) == 0  # every atom in this fixture has a real coordinate

    x1 = Matrix{Float64}(undef, 3, length(tokens))
    for (i, t) in enumerate(tokens)
        x1[:, i] .= t.coord
    end

    cfg = ModelConfig(d_single=16, d_pair=8, n_heads=4, n_pairformer_layers=2, n_dit_layers=2, d_time=16)
    model = build_model(cfg)
    ps, st = DiffuseBioMol.Model.Network.Lux.setup(rng, model)
    cond_features = constraint_features(no_constraints(length(tokens)))

    # Fix a single training example (no per-iteration randomness) so the test
    # checks the architecture's capacity to fit, not how fast a noisy
    # stochastic objective converges.
    example = prepare_training_example(feat, x1, cond_features, rng)
    loss_initial, _ = cfm_loss(model, ps, st, feat, relpos, example)
    @test isfinite(loss_initial)

    opt_state = Optimisers.setup(Optimisers.Adam(1.0f-2), ps)
    for _ in 1:150
        l, back = Zygote.pullback(p -> cfm_loss(model, p, st, feat, relpos, example)[1], ps)
        grad = back(1.0f0)[1]
        opt_state, ps = Optimisers.update(opt_state, ps, grad)
    end

    loss_final, _ = cfm_loss(model, ps, st, feat, relpos, example)
    @test loss_final < loss_initial / 5  # the network has the capacity to fit a single example

    x_sample, _ = sample_flow(model, ps, st, feat, relpos, cond_features, rng; n_steps=10)
    @test size(x_sample) == size(x1)
    @test all(isfinite, x_sample)
end
