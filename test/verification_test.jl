using Test
using DiffuseBioMol
using Random, Zygote, Optimisers

function two_residue_fixture()
    res1 = ParsedResidue("A", 1, "ALA", PROTEIN, Dict("N" => (0.0, 0.0, 0.0), "CA" => (1.5, 0.0, 0.0), "C" => (2.9, 0.4, 0.0), "O" => (3.4, 1.5, 0.0), "CB" => (1.9, -0.7, 1.2)))
    res2 = ParsedResidue("A", 2, "GLY", PROTEIN, Dict("N" => (3.6, -0.6, -0.3), "CA" => (5.0, -0.5, -0.2), "C" => (5.6, -1.9, -0.4), "O" => (5.0, -2.9, -0.1)))
    tokens = tokenize_structure([res1, res2])
    feat = featurize(tokens)
    relpos = relpos_buckets(feat)
    x1 = Matrix{Float64}(undef, 3, length(tokens))
    for (i, t) in enumerate(tokens)
        x1[:, i] .= t.coord
    end
    (tokens=tokens, feat=feat, relpos=relpos, x1=x1)
end

@testset "backbone_bonds: derives the correct bonded-pair set" begin
    ex = two_residue_fixture()
    bonds = backbone_bonds(ex.tokens)
    # Atom order from tokenize_structure(ALA, GLY): ALA[N,CA,C,O,CB]=1:5, GLY[N,CA,C,O]=6:9
    expected = Set([
        (1, 2, 1.458), (2, 3, 1.525), (3, 4, 1.231), (2, 5, 1.530),  # ALA backbone + CB
        (6, 7, 1.458), (7, 8, 1.525), (8, 9, 1.231),                # GLY backbone (no CB)
        (3, 6, 1.329),                                              # peptide bond ALA.C -> GLY.N
    ])
    @test Set(bonds) == expected
    @test length(bonds) == length(expected)  # no duplicates
end

@testset "backbone_bonds: no spurious peptide bond across different chains" begin
    res1 = ParsedResidue("A", 1, "ALA", PROTEIN, Dict("N" => (0.0, 0.0, 0.0), "CA" => (1.5, 0.0, 0.0), "C" => (2.9, 0.4, 0.0), "O" => (3.4, 1.5, 0.0)))
    res2 = ParsedResidue("B", 1, "GLY", PROTEIN, Dict("N" => (3.6, -0.6, -0.3), "CA" => (5.0, -0.5, -0.2), "C" => (5.6, -1.9, -0.4), "O" => (5.0, -2.9, -0.1)))
    tokens = tokenize_structure([res1, res2])
    bonds = backbone_bonds(tokens)
    @test !any(b -> b[1] == 3 && b[2] == 5, bonds)  # ALA.C (chain A) -> GLY.N (chain B) must not bond
    # ALA = N,CA,C,O,CB (5 tokens; CB is a virtual atom14 slot here, but
    # backbone_bonds derives bonds from the token's atom_name regardless of
    # is_virtual, so CA-CB is still included) -> 4 intra-residue bonds.
    # GLY = N,CA,C,O (no CB slot at all) -> 3 intra-residue bonds. No peptide
    # bond since the two residues are on different chains.
    @test length(bonds) == 7
end

@testset "validity_guidance_step: strictly decreases validity energy" begin
    rng = Random.Xoshiro(99)
    ex = two_residue_fixture()
    bonds = backbone_bonds(ex.tokens)
    elements = [t.element for t in ex.tokens]
    n = length(ex.tokens)

    x_perturbed = Float32.(ex.x1) .+ 0.5f0 .* Float32.(randn(rng, 3, n))
    e_before = validity_energy(x_perturbed, elements, ex.feat.chain_idx, ex.feat.res_index, bonds)

    post = validity_guidance_step(elements, ex.feat.chain_idx, ex.feat.res_index, bonds; step_size=0.01)
    x_after = post(x_perturbed, 0.5f0)
    e_after = validity_energy(x_after, elements, ex.feat.chain_idx, ex.feat.res_index, bonds)

    @test e_after < e_before
    @test size(x_after) == size(x_perturbed)
end

@testset "chiral_centers: derives CA stereocenters, skips glycine" begin
    ex = two_residue_fixture()
    centers = chiral_centers(ex.tokens)
    # Atom order (see two_residue_fixture): ALA[N,CA,C,O,CB]=1:5, GLY[N,CA,C,O]=6:9.
    # ALA has a CB -> one center (CA=2, N=1, C=3, CB=5); GLY has none -> no center.
    @test centers == [(2, 1, 3, 5)]
end

@testset "validity_guidance_step: also decreases chirality energy when centers are passed" begin
    ex = two_residue_fixture()
    bonds = backbone_bonds(ex.tokens)
    centers = chiral_centers(ex.tokens)
    elements = [t.element for t in ex.tokens]

    x0 = Float32.(ex.x1)
    e_before = validity_energy(x0, elements, ex.feat.chain_idx, ex.feat.res_index, bonds, centers)

    post = validity_guidance_step(elements, ex.feat.chain_idx, ex.feat.res_index, bonds, centers; step_size=0.01)
    x_after = post(x0, 0.5f0)
    e_after = validity_energy(x_after, elements, ex.feat.chain_idx, ex.feat.res_index, bonds, centers)

    @test e_after < e_before
end

@testset "build_verifier: output shapes and ranges are correct" begin
    rng = Random.Xoshiro(1)
    ex = two_residue_fixture()
    n = length(ex.tokens)

    cfg = VerifierConfig(d_single=16, n_heads=4, n_layers=2,
        n_elements=N_ELEMENTS, n_modalities=N_MODALITIES,
        n_polymer_atom_types=N_POLYMER_ATOM_TYPES, n_relpos_buckets=N_RELPOS_BUCKETS)
    model = build_verifier(cfg)
    ps, st = DiffuseBioMol.Verification.Verifier.Lux.setup(rng, model)

    x1f = Float32.(ex.x1)
    (confidence, clash_logit), _ = model((ex.feat.element_idx, ex.feat.modality_idx, ex.feat.polymer_atom_idx, ex.relpos, x1f), ps, st)

    @test length(confidence) == n
    @test length(clash_logit) == n
    @test all(0 .<= confidence .<= 1)  # passed through sigmoid
    @test all(isfinite, clash_logit)
end

@testset "verifier_loss: trains to near-zero on a single fixed example" begin
    rng = Random.Xoshiro(2)
    ex = two_residue_fixture()
    n = length(ex.tokens)

    cfg = VerifierConfig(d_single=16, n_heads=4, n_layers=2,
        n_elements=N_ELEMENTS, n_modalities=N_MODALITIES,
        n_polymer_atom_types=N_POLYMER_ATOM_TYPES, n_relpos_buckets=N_RELPOS_BUCKETS)
    model = build_verifier(cfg)
    ps, st = DiffuseBioMol.Verification.Verifier.Lux.setup(rng, model)

    x1f = Float32.(ex.x1)
    conf_target = lddt(x1f, x1f)  # self-comparison: every atom should score 1.0
    @test all(conf_target .== 1.0)
    clash_target = zeros(Float32, n)

    loss_initial, _ = verifier_loss(model, ps, st, ex.feat.element_idx, ex.feat.modality_idx, ex.feat.polymer_atom_idx, ex.relpos, x1f, conf_target, clash_target)
    @test isfinite(loss_initial)

    opt_state = Optimisers.setup(Optimisers.Adam(1.0f-2), ps)
    for _ in 1:80
        l, back = Zygote.pullback(
            p -> verifier_loss(model, p, st, ex.feat.element_idx, ex.feat.modality_idx, ex.feat.polymer_atom_idx, ex.relpos, x1f, conf_target, clash_target)[1],
            ps,
        )
        grad = back(1.0f0)[1]
        opt_state, ps = Optimisers.update(opt_state, ps, grad)
    end

    loss_final, _ = verifier_loss(model, ps, st, ex.feat.element_idx, ex.feat.modality_idx, ex.feat.polymer_atom_idx, ex.relpos, x1f, conf_target, clash_target)
    @test loss_final < loss_initial / 10
end

@testset "Phase 3 integration: in-loop guidance reduces sampled-structure validity energy" begin
    rng = Random.Xoshiro(5)
    ex = two_residue_fixture()
    bonds = backbone_bonds(ex.tokens)
    centers = chiral_centers(ex.tokens)
    elements = [t.element for t in ex.tokens]
    n = length(ex.tokens)
    cond_features = constraint_features(no_constraints(n))

    cfg = ModelConfig(d_single=16, d_pair=8, n_heads=4, n_pairformer_layers=2, n_dit_layers=2, d_time=16)
    model = build_model(cfg)
    ps, st = DiffuseBioMol.Model.Network.Lux.setup(rng, model)

    post = validity_guidance_step(elements, ex.feat.chain_idx, ex.feat.res_index, bonds, centers; step_size=0.02)

    x_guided, _ = sample_flow(model, ps, st, ex.feat, ex.relpos, cond_features, Random.Xoshiro(123); n_steps=20, post_step=post)
    x_unguided, _ = sample_flow(model, ps, st, ex.feat, ex.relpos, cond_features, Random.Xoshiro(123); n_steps=20)

    @test all(isfinite, x_guided)
    e_guided = validity_energy(x_guided, elements, ex.feat.chain_idx, ex.feat.res_index, bonds, centers)
    e_unguided = validity_energy(x_unguided, elements, ex.feat.chain_idx, ex.feat.res_index, bonds, centers)
    @test e_guided < e_unguided
end
