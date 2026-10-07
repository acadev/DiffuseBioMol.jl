"""
Stage A validity benchmark — the Phase 1 verification gate from
`docs/PLAN.md`, finally operationalized as a recurring harness: does the
model produce physically sane geometry on held-out real structures, and is
it actually better than doing no learning at all?

Also doubles as the Phase 3 guidance gate: `sample_flow` is run both with and
without `validity_guidance_step` (clash + bond + chirality) so the guided
column can be compared directly against the unguided one, on top of the
existing prior/untrained/trained comparison.

This deliberately does *not* reach for literature-standard benchmarks
(PoseBusters, CASP15, DockQ) — those need either a meaningfully-trained model
or external tool bridging, both premature right now. This harness uses only
this codebase's own `Geometry` machinery (clash count, bond-length RMSD,
chirality count), so it's runnable today with zero new dependencies and zero
GPU, and gives a number to track as training scales up.

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
    centers = chiral_centers(tokens)
    elements = [t.element for t in tokens]
    (pdb_id=pdb_id, feat=feat, relpos=relpos, x1=x1, cond_features=cond_features,
        bonds=bonds, centers=centers, elements=elements, n_atoms=length(tokens))
end

"""
    validity_report(label, coords, ex) -> NamedTuple

Reports clash count (raw and normalized per 100 atoms, so structures of
different sizes are comparable), bond-length RMSD (Å), and chirality
violation count for one sampled structure against one loaded example's
bonded-pair/chiral-center/element/chain metadata.
"""
function validity_report(label, coords, ex)
    nclash = clash_count(Float32.(coords), ex.elements, ex.feat.chain_idx, ex.feat.res_index)
    brmsd = bond_length_rmsd(Float32.(coords), ex.bonds)
    nchiral = chirality_count(Float32.(coords), ex.centers)
    (label=label, pdb_id=ex.pdb_id, n_atoms=ex.n_atoms,
        clashes=nclash, clashes_per_100_atoms=round(100 * nclash / ex.n_atoms; digits=2),
        bond_rmsd_angstrom=round(brmsd; digits=3),
        chirality_violations=nchiral, n_chiral_centers=length(ex.centers))
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

        # Phase 3 gate: same trained model/seed, with vs. without in-loop
        # clash+bond+chirality guidance (Sampling.FlowMatching's post_step hook).
        post = validity_guidance_step(ex.elements, ex.feat.chain_idx, ex.feat.res_index, ex.bonds, ex.centers)
        x_guided, _ = sample_flow(model, ps_trained, st, ex.feat, ex.relpos, ex.cond_features, rng; n_steps=n_steps, post_step=post)
        push!(reports, validity_report("trained+guided", x_guided, ex))
    end

    println("\n", "="^108)
    println(rpad("label", 16), rpad("pdb_id", 10), rpad("n_atoms", 10), rpad("clashes", 10),
        rpad("clashes/100atoms", 18), rpad("bond_rmsd (A)", 15), rpad("chirality_bad", 15), "n_chiral_centers")
    println("-"^108)
    for r in reports
        println(
            rpad(r.label, 16), rpad(r.pdb_id, 10), rpad(string(r.n_atoms), 10),
            rpad(string(r.clashes), 10), rpad(string(r.clashes_per_100_atoms), 18),
            rpad(string(r.bond_rmsd_angstrom), 15), rpad(string(r.chirality_violations), 15),
            string(r.n_chiral_centers),
        )
    end
    println("="^108)
    println("\nExpected pattern as training scales up: trained < untrained < prior on")
    println("clashes/bond_rmsd, for each held-out structure. trained+guided is the Phase 3")
    println("gate: it should match or beat trained on all three metrics (same model/seed,")
    println("guidance added at sampling time only). At this toy scale/training budget,")
    println("don't expect strong separation yet -- this harness is the gate to track as")
    println("Phase 1/3 training is scaled up, not a finished evaluation.")
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
