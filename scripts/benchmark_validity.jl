"""
Stage A validity benchmark — the Phase 1 verification gate from
`docs/PLAN.md`, finally operationalized as a recurring harness: does the
model produce physically sane geometry on held-out real structures, and is
it actually better than doing no learning at all?

This deliberately does *not* reach for literature-standard benchmarks
(PoseBusters, CASP15, DockQ) — those need either a meaningfully-trained model
or external tool bridging, both premature right now. This harness uses only
this codebase's own `Geometry` machinery (clash count, bond-length RMSD), so
it's runnable today with zero new dependencies and zero GPU, and gives a
number to track as training scales up.

Three conditions, per held-out structure:
- `prior`: a raw physics-informed prior sample (`Sampling.Prior`), zero
  learning — the floor any trained model should beat.
- `untrained`: a freshly-initialized model's `sample_flow` output — isolates
  how much of any improvement comes from the architecture/guidance alone
  vs. actual learning.
- `trained`: after a brief training pass on a *different* set of real
  structures (no overlap with the held-out set) — what we're actually
  trying to improve as training scales up.

Run with: `julia --project=. scripts/benchmark_validity.jl`
"""

using DiffuseBioMol
using Random, Zygote, Optimisers

const TRAIN_TARGETS = [("1CRN", "A"), ("1UBQ", "A"), ("5PTI", "A")]
const HELDOUT_TARGETS = [("1L2Y", "A"), ("1VII", "A"), ("2GB1", "A")]

function load_example(pdb_id, chain_id)
    residues = restrict_to_chain(fetch_pdb(pdb_id), chain_id)
    tokens = tokenize_structure(residues)
    feat = featurize(tokens)
    relpos = relpos_buckets(feat)
    x1 = target_coordinates(tokens)
    cond_features = constraint_features(no_constraints(length(tokens)))
    bonds = backbone_bonds(tokens)
    elements = [t.element for t in tokens]
    (pdb_id=pdb_id, feat=feat, relpos=relpos, x1=x1, cond_features=cond_features,
        bonds=bonds, elements=elements, n_atoms=length(tokens))
end

"""
    validity_report(label, coords, ex) -> NamedTuple

Reports clash count (raw and normalized per 100 atoms, so structures of
different sizes are comparable) and bond-length RMSD (Å) for one sampled
structure against one loaded example's bonded-pair/element/chain metadata.
"""
function validity_report(label, coords, ex)
    nclash = clash_count(Float32.(coords), ex.elements, ex.feat.chain_idx, ex.feat.res_index)
    brmsd = bond_length_rmsd(Float32.(coords), ex.bonds)
    (label=label, pdb_id=ex.pdb_id, n_atoms=ex.n_atoms,
        clashes=nclash, clashes_per_100_atoms=round(100 * nclash / ex.n_atoms; digits=2),
        bond_rmsd_angstrom=round(brmsd; digits=3))
end

function train!(model, ps, st, examples, rng; n_epochs=5, lr=1.0f-3)
    opt_state = Optimisers.setup(Optimisers.Adam(lr), ps)
    for _epoch in 1:n_epochs
        for ex in examples
            training_example = prepare_training_example(ex.feat, ex.x1, ex.cond_features, rng)
            _loss, back = Zygote.pullback(p -> cfm_loss(model, p, st, ex.feat, ex.relpos, training_example)[1], ps)
            grad = back(1.0f0)[1]
            opt_state, ps = Optimisers.update(opt_state, ps, grad)
        end
    end
    ps
end

function main(; n_epochs=5, n_steps=20, seed=0)
    rng = Random.Xoshiro(seed)

    println("Loading training structures...")
    train_examples = [load_example(id, chain) for (id, chain) in TRAIN_TARGETS]
    println("Loading held-out structures (no overlap with training)...")
    heldout_examples = [load_example(id, chain) for (id, chain) in HELDOUT_TARGETS]

    cfg = ModelConfig(d_single=32, d_pair=16, n_heads=4, n_pairformer_layers=2, n_dit_layers=2, d_time=32)
    model = build_model(cfg)
    ps_untrained, st = DiffuseBioMol.Model.Network.Lux.setup(rng, model)

    println("Training briefly ($n_epochs epochs on $(length(train_examples)) structures)...")
    ps_trained = train!(model, ps_untrained, st, train_examples, rng; n_epochs=n_epochs)

    reports = NamedTuple[]
    for ex in heldout_examples
        prior_sample = sample_prior(rng, ex.feat.chain_idx)
        push!(reports, validity_report("prior", prior_sample, ex))

        x_untrained, _ = sample_flow(model, ps_untrained, st, ex.feat, ex.relpos, ex.cond_features, rng; n_steps=n_steps)
        push!(reports, validity_report("untrained", x_untrained, ex))

        x_trained, _ = sample_flow(model, ps_trained, st, ex.feat, ex.relpos, ex.cond_features, rng; n_steps=n_steps)
        push!(reports, validity_report("trained", x_trained, ex))
    end

    println("\n", "="^88)
    println(rpad("label", 12), rpad("pdb_id", 10), rpad("n_atoms", 10), rpad("clashes", 10),
        rpad("clashes/100atoms", 18), "bond_rmsd (A)")
    println("-"^88)
    for r in reports
        println(
            rpad(r.label, 12), rpad(r.pdb_id, 10), rpad(string(r.n_atoms), 10),
            rpad(string(r.clashes), 10), rpad(string(r.clashes_per_100_atoms), 18),
            string(r.bond_rmsd_angstrom),
        )
    end
    println("="^88)
    println("\nExpected pattern as training scales up: trained < untrained < prior")
    println("on both metrics, for each held-out structure. At this toy scale/training")
    println("budget, don't expect strong separation yet -- this harness is the gate to")
    println("track as Phase 1/3 training is scaled up, not a finished evaluation.")
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
