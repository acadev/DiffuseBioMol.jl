"""
Phase 1 hardening, batched: train the Pairformer-lite + DiT flow-matching
backbone on real PDB structures using `Model.Batching`'s cross-structure
batching, rather than `train_phase1_real_data.jl`'s one-structure-at-a-time
loop.

Still a small-scale smoke test, not production training — the point is to
exercise the *batched* real-data path end-to-end (load -> tokenize -> batch
-> train -> sample -> validity-check) before trusting it on a larger run,
since the batched API has so far only been exercised by unit tests against
tiny synthetic fixtures (see `test/batching_test.jl`) and the unbatched
path's real-data run (`train_phase1_real_data.jl`).

Two data sources, chosen by whether `data_dir` is given:
- **Local directory** (`data_dir` keyword, or the first command-line
  argument) — the expected mode for a real run against a pre-staged dataset
  (e.g. a local PDB/PDBBind mirror on a training node), rather than live RCSB
  fetches every run. Every structure file under `data_dir` (via
  `Data.list_structure_files`) is restricted to its *largest* chain (via
  `Data.largest_chain` — local dataset files are often multi-chain with no
  "the chain we want is A" convention, unlike the curated single-chain
  RCSB list below), then loaded the same way as the RCSB path.
- **Built-in RCSB candidate list** (`data_dir=nothing`, the default) — live
  `fetch_pdb` downloads of a small curated single-chain set, what the
  CPU smoke test recorded in `docs/PLAN.md` used.

Run with:
    julia --project=. scripts/train_phase1_batched_real_data.jl                 # RCSB
    julia --project=. scripts/train_phase1_batched_real_data.jl /path/to/pdbs   # local directory
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

"""
    tokenize_example(label, residues) -> NamedTuple

Shared tail of both loading paths: residues (already restricted to one
chain) -> tokens -> the feature/coordinate bundle `train_on_batch`/sampling
need. Raises on an empty token list (e.g. a chain that turned out to be pure
water/heteroatom) so callers can skip-and-continue.
"""
function tokenize_example(label::AbstractString, residues)
    tokens = tokenize_structure(residues)
    isempty(tokens) && error("no tokens after parsing/filtering")
    feat = featurize(tokens)
    relpos = relpos_buckets(feat)
    x1 = target_coordinates(tokens)
    cond_features = constraint_features(no_constraints(length(tokens)))
    (pdb_id=label, tokens=tokens, feat=feat, relpos=relpos, x1=x1,
        cond_features=cond_features, n_atoms=length(tokens))
end

function load_rcsb_example(pdb_id, chain_id)
    tokenize_example(pdb_id, restrict_to_chain(fetch_pdb(pdb_id), chain_id))
end

"""
    load_local_example(path) -> NamedTuple

Parse a structure file from disk and restrict it to its largest chain (see
the module docstring's "Local directory" note) before tokenizing.
"""
function load_local_example(path::AbstractString)
    residues = parse_structure(path)
    isempty(residues) && error("no residues parsed")
    chain = largest_chain(residues)
    label = splitext(basename(path))[1]
    tokenize_example(label, restrict_to_chain(residues, chain))
end

"""
    load_targets(; data_dir=nothing, n_targets=N_TARGETS, max_atoms=nothing, rng) -> Vector

Loads up to `n_targets` examples, skipping (not aborting on) any individual
structure that fails to fetch/parse/tokenize, or — if `max_atoms` is given —
that exceeds it (a local dataset directory can contain much larger
complexes than the curated RCSB list, and an unbatched-per-structure
`sample_flow`/validity-check pass over a huge structure later in this script
could dominate the whole run's wall-clock).
"""
function load_targets(; data_dir::Union{Nothing,AbstractString}=nothing, n_targets::Int=N_TARGETS,
    max_atoms::Union{Nothing,Int}=nothing, rng::AbstractRNG)
    candidates = if data_dir === nothing
        CANDIDATE_TARGETS
    else
        files = list_structure_files(data_dir)
        isempty(files) && throw(ArgumentError("no PDB/mmCIF files found under $data_dir"))
        shuffle(rng, files)
    end

    loader = data_dir === nothing ? (c -> load_rcsb_example(c...)) : load_local_example
    examples = []
    for c in candidates
        length(examples) >= n_targets && break
        try
            ex = loader(c)
            (max_atoms !== nothing && ex.n_atoms > max_atoms) && continue
            push!(examples, ex)
        catch e
            println("  skipping $c ($e)")
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

function main(; data_dir::Union{Nothing,AbstractString}=nothing, n_targets=N_TARGETS, max_atoms=nothing,
    n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, lr=1.0f-3, seed=0)
    rng = Random.Xoshiro(seed)

    source = data_dir === nothing ? "RCSB (live fetch)" : "local directory $data_dir"
    println("Loading and tokenizing real structures from $source (target: $n_targets)...")
    t_fetch = @elapsed examples = load_targets(; data_dir, n_targets, max_atoms, rng)
    println("got $(length(examples)) structures:")
    for ex in examples
        println("  $(ex.pdb_id): $(ex.n_atoms) atoms")
    end
    println("load+tokenize: $(round(t_fetch; digits=1))s")
    isempty(examples) && error("no structures loaded -- nothing to train on")
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
    data_dir = isempty(ARGS) ? nothing : ARGS[1]
    main(; data_dir)
end
