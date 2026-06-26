using Test
using DiffuseBioMol
using Random
import DiffuseBioMol.Model.Network.Lux as Lux

function tiny_example(n_atoms::Int, seed::Int)
    rng = Random.Xoshiro(seed)
    # Build a minimal but valid protein chain directly via real residues so
    # featurize/relpos_buckets behave exactly as in the rest of the test suite.
    residues = ParsedResidue[]
    natoms = 0
    ri = 1
    while natoms < n_atoms
        atoms = Dict("N" => (Float64(ri), 0.0, 0.0), "CA" => (Float64(ri) + 1.5, 0.0, 0.0), "C" => (Float64(ri) + 2.9, 0.4, 0.0), "O" => (Float64(ri) + 3.4, 1.5, 0.0))
        push!(residues, ParsedResidue("A", ri, "GLY", PROTEIN, atoms))
        natoms += 4
        ri += 1
    end
    tokens = tokenize_structure(residues)
    tokens = tokens[1:n_atoms]  # trim to exactly n_atoms
    feat = featurize(tokens)
    relpos = relpos_buckets(feat)
    cond_features = constraint_features(no_constraints(n_atoms))
    x_t = Float32.(randn(rng, 3, n_atoms))
    (feat=feat, relpos=relpos, cond_features=cond_features, x_t=x_t, n_atoms=n_atoms)
end

@testset "Batching: real atoms produce identical output alone vs. inside a padded batch" begin
    rng = Random.Xoshiro(1)
    cfg = ModelConfig(d_single=16, d_pair=8, n_heads=4, n_pairformer_layers=2, n_dit_layers=2, d_time=16)
    model = build_model(cfg)
    ps, st = Lux.setup(rng, model)

    ex_small = tiny_example(3, 11)
    ex_large = tiny_example(5, 22)
    t_val = 0.37f0

    # "Alone" computation: exactly the B=1 wrapping FlowMatching.cfm_loss uses internally.
    as_batch_mat(m) = reshape(m, size(m, 1), size(m, 2), 1)
    as_batch_vec(v) = reshape(v, length(v), 1)
    zero_bias(n) = zeros(Float32, n, n, 1)

    v_alone, _ = model(
        (as_batch_vec(ex_small.feat.element_idx), as_batch_vec(ex_small.feat.modality_idx), as_batch_vec(ex_small.feat.polymer_atom_idx),
            as_batch_mat(ex_small.relpos), as_batch_mat(ex_small.x_t), Float32[t_val], as_batch_mat(ex_small.cond_features), zero_bias(3)),
        ps, st,
    )

    # Batched computation: ex_small padded to N_max=5 alongside ex_large.
    batched_feat = batch_features([ex_small.feat, ex_large.feat])
    relpos_batched = batch_relpos([ex_small.relpos, ex_large.relpos])
    cond_batched = batch_cond_features([ex_small.cond_features, ex_large.cond_features])
    x_t_padded = zeros(Float32, 3, 5)
    x_t_padded[:, 1:3] .= ex_small.x_t
    x_t_large_padded = ex_large.x_t  # already n_max=5, no padding needed
    x_t_batched = cat(reshape(x_t_padded, 3, 5, 1), reshape(x_t_large_padded, 3, 5, 1); dims=3)
    pad_bias = attention_pad_bias(batched_feat.pad_mask)

    v_batched, _ = model(
        (batched_feat.element_idx, batched_feat.modality_idx, batched_feat.polymer_atom_idx,
            relpos_batched, x_t_batched, Float32[t_val, t_val], cond_batched, pad_bias),
        ps, st,
    )

    @test size(v_batched) == (3, 5, 2)
    @test all(isfinite, v_batched)
    # The real atoms of the small (padded) structure must match the alone computation exactly.
    @test v_batched[:, 1:3, 1] ≈ v_alone[:, :, 1] atol = 1e-5
end

@testset "Batching: B=1 batch_features/batch_relpos/etc. are no-ops vs. single-structure data" begin
    ex = tiny_example(4, 3)
    batched_feat = batch_features([ex.feat])
    @test batched_feat.element_idx[:, 1] == ex.feat.element_idx
    @test batched_feat.modality_idx[:, 1] == ex.feat.modality_idx
    @test all(batched_feat.pad_mask[:, 1])  # no padding when B=1 and this is the only (longest) structure

    relpos_batched = batch_relpos([ex.relpos])
    @test relpos_batched[:, :, 1] == ex.relpos

    pad_bias = attention_pad_bias(batched_feat.pad_mask)
    @test all(pad_bias .== 0)  # nothing padded -> no masking applied
end

@testset "Batching: attention_pad_bias masks exactly the padded columns" begin
    pad_mask = BitMatrix([true true; true false; false false])  # N=3, B=2; structure 1 has 2 atoms, structure 2 has 1
    bias = attention_pad_bias(pad_mask)
    @test size(bias) == (3, 3, 2)
    # Structure 1: atom 3 is padding -> column 3 fully masked, columns 1-2 unmasked.
    @test all(bias[:, 1, 1] .== 0) && all(bias[:, 2, 1] .== 0)
    @test all(bias[:, 3, 1] .== -1.0f9)
    # Structure 2: only atom 1 is real -> columns 2-3 masked.
    @test all(bias[:, 1, 2] .== 0)
    @test all(bias[:, 2, 2] .== -1.0f9) && all(bias[:, 3, 2] .== -1.0f9)
end

@testset "Batching: prepare_training_example/cfm_loss run end-to-end at B>1" begin
    rng = Random.Xoshiro(5)
    cfg = ModelConfig(d_single=16, d_pair=8, n_heads=4, n_pairformer_layers=2, n_dit_layers=2, d_time=16)
    model = build_model(cfg)
    ps, st = Lux.setup(rng, model)

    ex1 = tiny_example(4, 1)
    ex2 = tiny_example(8, 2)
    batched_feat = batch_features([ex1.feat, ex2.feat])
    relpos_batched = batch_relpos([ex1.relpos, ex2.relpos])
    cond_batched = batch_cond_features([ex1.cond_features, ex2.cond_features])
    pad_bias = attention_pad_bias(batched_feat.pad_mask)

    x1_1 = Float64.(ex1.x_t)  # stand-in "ground truth" coordinates for this smoke test
    x1_2 = Float64.(ex2.x_t)
    x1_batched = batch_coords([x1_1, x1_2])

    example = prepare_training_example(batched_feat, x1_batched, cond_batched, rng)
    @test size(example.x_t) == (3, 8, 2)
    @test length(example.t) == 2
    @test count(example.mask) == 4 + 8  # all real atoms in both structures (none virtual/fixed/padded... except ex1 is padded to 8)

    loss, _ = cfm_loss(model, ps, st, batched_feat, relpos_batched, pad_bias, example)
    @test isfinite(loss)
    @test loss >= 0
end
