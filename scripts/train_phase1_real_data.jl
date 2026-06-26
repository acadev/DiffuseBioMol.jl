"""
Phase 1 hardening: train the Pairformer-lite + DiT flow-matching backbone on
real PDB structures (via `Data.fetch_pdb`) rather than synthetic fixtures.

This is a small multi-structure training run, not production-scale training —
each structure is a separate, unbatched forward/backward pass (current v1
limitation noted in `docs/PLAN.md`), and a handful of small single-chain
proteins is enough to confirm the architecture trains on real geometry and
real atom-type/element distributions, not just hand-built synthetic examples.
Run with: `julia --project=. scripts/train_phase1_real_data.jl`
"""

using DiffuseBioMol
using Random, Zygote, Optimisers

const PDB_TARGETS = [
    ("1CRN", "A"),  # crambin, 46 residues, no missing density
    ("1UBQ", "A"),  # ubiquitin, 76 residues
    ("5PTI", "A"),  # BPTI, 58 residues
]

function load_example(pdb_id, chain_id)
    residues = restrict_to_chain(fetch_pdb(pdb_id), chain_id)
    tokens = tokenize_structure(residues)
    feat = featurize(tokens)
    relpos = relpos_buckets(feat)
    x1 = target_coordinates(tokens)
    cond_features = constraint_features(no_constraints(length(tokens)))
    (pdb_id=pdb_id, feat=feat, relpos=relpos, x1=x1, cond_features=cond_features, n_atoms=length(tokens))
end

function main(; n_epochs=5, lr=1.0f-3, seed=0)
    rng = Random.Xoshiro(seed)

    println("Fetching and tokenizing real structures...")
    examples = [load_example(id, chain) for (id, chain) in PDB_TARGETS]
    for ex in examples
        println("  $(ex.pdb_id): $(ex.n_atoms) atoms")
    end

    cfg = ModelConfig(d_single=32, d_pair=16, n_heads=4, n_pairformer_layers=2, n_dit_layers=2, d_time=32)
    model = build_model(cfg)
    ps, st = DiffuseBioMol.Model.Network.Lux.setup(rng, model)
    opt_state = Optimisers.setup(Optimisers.Adam(lr), ps)

    for epoch in 1:n_epochs
        epoch_losses = Float64[]
        for ex in shuffle(rng, examples)
            training_example = prepare_training_example(ex.feat, ex.x1, ex.cond_features, rng)
            loss, back = Zygote.pullback(p -> cfm_loss(model, p, st, ex.feat, ex.relpos, training_example)[1], ps)
            grad = back(1.0f0)[1]
            opt_state, ps = Optimisers.update(opt_state, ps, grad)
            push!(epoch_losses, loss)
        end
        println("epoch $epoch: losses = ", round.(epoch_losses; digits=2))
    end

    println("\nSampling each structure after training:")
    for ex in examples
        x_sample, _ = sample_flow(model, ps, st, ex.feat, ex.relpos, ex.cond_features, rng; n_steps=20)
        rmse = sqrt(sum(abs2, x_sample .- Float32.(ex.x1)) / length(ex.x1))
        println("  $(ex.pdb_id): sample RMSE vs. native coords = $(round(rmse; digits=2)) A")
    end
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
