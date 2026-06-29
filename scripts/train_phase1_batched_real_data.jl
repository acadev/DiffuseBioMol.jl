"""
Phase 1 hardening, batched: train the Pairformer-lite + DiT flow-matching
backbone on ~20 real PDB structures using `Model.Batching`'s cross-structure
batching, rather than `train_phase1_real_data.jl`'s one-structure-at-a-time
loop.

Still a small-scale smoke test, not production training — the point is to
exercise the *batched* real-data path end-to-end (fetch -> tokenize -> batch
-> train -> sample -> validity-check) at least once before trusting it on a
larger run, since the batched API has so far only been exercised by unit
tests against tiny synthetic fixtures (see `test/batching_test.jl`) and the
unbatched path's real-data run (`train_phase1_real_data.jl`).

Run with: `julia --project=. scripts/train_phase1_batched_real_data.jl`
"""

using DiffuseBioMol
using Random, Zygote, Optimisers

# A few more candidates than we need (~24 for ~20 targets), since not every
# PDB ID is guaranteed to fetch/parse cleanly (missing chain A, no residues
# left after water/heteroatom filtering, etc.) -- failures are skipped, not
# fatal, so the run degrades gracefully rather than aborting partway through.
const CANDIDATE_TARGETS = [
    ("1CRN", "A"), ("1UBQ", "A"), ("5PTI", "A"), ("1L2Y", "A"), ("1VII", "A"), ("2GB1", "A"),
    ("1ENH", "A"), ("1PGB", "A"), ("1SHG", "A"), ("2CI2", "A"), ("1IGD", "A"), ("1TEN", "A"),
    ("2TRX", "A"), ("1AHO", "A"), ("1BTA", "A"), ("1CSP", "A"), ("1FNA", "A"), ("1MJC", "A"),
    ("1PIN", "A"), ("1HHP", "A"), ("2PTL", "A"), ("1UBI", "A"), ("1COA", "A"), ("1FXD", "A"),
]
const N_TARGETS = 20
const BATCH_SIZE = 4
const N_EPOCHS = 10

function load_example(pdb_id, chain_id)
    residues = restrict_to_chain(fetch_pdb(pdb_id), chain_id)
    tokens = tokenize_structure(residues)
    isempty(tokens) && error("no tokens after parsing/filtering")
    feat = featurize(tokens)
    relpos = relpos_buckets(feat)
    x1 = target_coordinates(tokens)
    cond_features = constraint_features(no_constraints(length(tokens)))
    (pdb_id=pdb_id, tokens=tokens, feat=feat, relpos=relpos, x1=x1,
        cond_features=cond_features, n_atoms=length(tokens))
end

function fetch_targets()
    examples = typeof(load_example("1CRN", "A"))[]
    for (id, chain) in CANDIDATE_TARGETS
        length(examples) >= N_TARGETS && break
        try
            push!(examples, load_example(id, chain))
        catch e
            println("  skipping $id ($e)")
        end
    end
    examples
end

function make_batches(examples, batch_size, rng)
    order = shuffle(rng, 1:length(examples))
    [examples[order[i:min(i + batch_size - 1, end)]] for i in 1:batch_size:length(order)]
end

function train_on_batch(model, ps, st, opt_state, batch, rng)
    batched_feat = batch_features([ex.feat for ex in batch])
    relpos_batched = batch_relpos([ex.relpos for ex in batch])
    cond_batched = batch_cond_features([ex.cond_features for ex in batch])
    pad_bias = attention_pad_bias(batched_feat.pad_mask)
    x1_batched = batch_coords([Float64.(ex.x1) for ex in batch])

    example = prepare_training_example(batched_feat, x1_batched, cond_batched, rng)
    loss, back = Zygote.pullback(p -> cfm_loss(model, p, st, batched_feat, relpos_batched, pad_bias, example)[1], ps)
    grad = back(1.0f0)[1]
    opt_state, ps = Optimisers.update(opt_state, ps, grad)
    (ps, opt_state, loss)
end

function main(; n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, lr=1.0f-3, seed=0)
    rng = Random.Xoshiro(seed)

    println("Fetching and tokenizing real structures (target: $N_TARGETS)...")
    t_fetch = @elapsed examples = fetch_targets()
    println("got $(length(examples)) structures:")
    for ex in examples
        println("  $(ex.pdb_id): $(ex.n_atoms) atoms")
    end
    println("fetch+tokenize: $(round(t_fetch; digits=1))s")
    total_atoms = sum(ex.n_atoms for ex in examples)
    println("total atoms across all structures: $total_atoms (mean $(round(total_atoms/length(examples); digits=1)))")

    cfg = ModelConfig(d_single=32, d_pair=16, n_heads=4, n_pairformer_layers=2, n_dit_layers=2, d_time=32)
    model = build_model(cfg)
    ps, st = DiffuseBioMol.Model.Network.Lux.setup(rng, model)
    opt_state = Optimisers.setup(Optimisers.Adam(lr), ps)

    println("\nTraining $n_epochs epochs, batch size $batch_size ($(cld(length(examples), batch_size)) batches/epoch)...")
    t_train = @elapsed for epoch in 1:n_epochs
        batches = make_batches(examples, batch_size, rng)
        epoch_losses = Float64[]
        t_epoch = @elapsed for batch in batches
            ps, opt_state, loss = train_on_batch(model, ps, st, opt_state, batch, rng)
            push!(epoch_losses, loss)
        end
        println("  epoch $epoch ($(round(t_epoch; digits=1))s): losses = ", round.(epoch_losses; digits=1))
    end
    println("total training time ($n_epochs epochs): $(round(t_train; digits=1))s")

    println("\nSampling + validity check on each structure (unbatched single-structure API):")
    for ex in examples
        x_sample, _ = sample_flow(model, ps, st, ex.feat, ex.relpos, ex.cond_features, rng; n_steps=20)
        rmsd = aligned_rmsd(x_sample, Float32.(ex.x1))
        bonds = backbone_bonds(ex.tokens)
        elements = [t.element for t in ex.tokens]
        clashes = clash_count(x_sample, elements, ex.feat.chain_idx, ex.feat.res_index)
        brmsd = bond_length_rmsd(x_sample, bonds)
        println("  $(ex.pdb_id) ($(ex.n_atoms) atoms): RMSD (superposed) = $(round(rmsd; digits=2)) A, " *
                "clashes = $clashes, bond-length RMSD = $(round(brmsd; digits=3)) A")
    end

    (model=model, ps=ps, st=st, examples=examples)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
