"""
Reproducible single-device baseline trainer.

This is the first non-smoke-test training entry point for DiffuseBioMol. It
uses a local PDB/mmCIF corpus, records a deterministic train/validation split,
trains length-bucketed padded batches, writes a resumable checkpoint, and
evaluates both reconstruction and native geometry gates at epoch zero and at
each checkpoint interval.

Usage:
    julia --project=. scripts/train_baseline.jl configs/baseline.toml runs/baseline
    julia --project=. scripts/train_baseline.jl configs/baseline.toml runs/baseline --resume
    julia --project=. scripts/train_baseline.jl configs/baseline.toml runs/baseline --gpu

The default configuration deliberately has a placeholder `data_dir`; copy it
and point it at a local, immutable corpus before running a real experiment.
Metrics are appended to `metrics.csv`, the exact split is saved to
`manifest.toml`, and `checkpoint_latest.jls` is atomically replaced after each
checkpoint.  Checkpoint state includes model parameters, optimizer state, RNG,
and the next epoch, so interruption does not change the training trajectory.

Set `[wandb].enabled = true` in the TOML configuration to mirror run metadata,
epoch losses, validation aggregates, and final gate status to a Weights & Biases
dashboard. A W&B run is initialized only in that opt-in path; set
`WANDB_API_KEY` in the job environment rather than storing a credential here.

Geometry reporting is native to this package: clash count, backbone bond RMSD,
and CA chirality violations. Reconstruction is aligned coordinate RMSD from a
deterministic flow sample.  A run is eligible for later fast-sampler work only
when its final `gates.toml` says all gates passed; a gate failure is a result to
investigate, never silently treated as a successful baseline.
"""

using DiffuseBioMol
using Random, Zygote, Optimisers
using TOML, Serialization, Statistics, Printf, Dates
import Wandb

const Lux = DiffuseBioMol.Model.Network.Lux

"""One loaded structure plus all immutable metadata needed by training/eval."""
struct BaselineExample
    label::String
    source::String
    tokens::Vector{AtomToken}
    feat::TokenFeatures
    relpos::Matrix{Int}
    x1::Matrix{Float64}
    cond_features::Matrix{Float32}
    bonds::Vector{Tuple{Int,Int,Float64}}
    centers::Vector{NTuple{4,Int}}
    elements::Vector{Symbol}
end

n_atoms(ex::BaselineExample) = length(ex.tokens)

"""Opt-in W&B handle; `nothing` denotes local-only tracking."""
struct WandbTracker
    logger
    upload_checkpoints::Bool
end

function require_key(table, key::AbstractString)
    haskey(table, key) || throw(ArgumentError("missing required configuration key `$key`"))
    table[key]
end

function load_config(path::AbstractString)
    config = TOML.parsefile(path)
    data, model, training, evaluation = (require_key(config, k) for k in ("data", "model", "training", "evaluation"))
    data_dir = String(require_key(data, "data_dir"))
    isdir(data_dir) || throw(ArgumentError("data.data_dir is not a readable directory: $data_dir"))
    (data=data, model=model, training=training, evaluation=evaluation, raw=config, path=abspath(path))
end

function load_example(path::AbstractString)
    residues = parse_structure(path)
    isempty(residues) && error("no residues parsed")
    chain = largest_chain(residues)
    tokens = tokenize_structure(restrict_to_chain(residues, chain))
    isempty(tokens) && error("no tokens after selecting largest chain")
    feat = featurize(tokens)
    BaselineExample(
        splitext(basename(path))[1], abspath(path), tokens, feat, relpos_buckets(feat),
        target_coordinates(tokens), constraint_features(no_constraints(length(tokens))),
        backbone_bonds(tokens), chiral_centers(tokens), [t.element for t in tokens],
    )
end

"""Load the fixed corpus once; malformed/oversized records are reported and skipped."""
function load_corpus(data_dir::AbstractString; max_atoms::Int)
    files = list_structure_files(data_dir; recursive=true)
    isempty(files) && throw(ArgumentError("no PDB/mmCIF files found under $data_dir"))
    examples = BaselineExample[]
    skipped = String[]
    for path in files
        try
            ex = load_example(path)
            if n_atoms(ex) <= max_atoms
                push!(examples, ex)
            else
                push!(skipped, "$path ($(n_atoms(ex)) atoms > max_atoms=$max_atoms)")
            end
        catch err
            push!(skipped, "$path ($err)")
        end
    end
    isempty(examples) && error("no usable structures in $data_dir")
    examples, skipped
end

"""A deterministic structure-level split; never split atoms from one structure across sets."""
function split_examples(examples::Vector{BaselineExample}, seed::Int, validation_fraction::Real)
    0 < validation_fraction < 1 || throw(ArgumentError("validation_fraction must lie in (0, 1)"))
    length(examples) >= 3 || throw(ArgumentError("need at least three usable structures for train/validation"))
    ordered = sort(examples; by = ex -> ex.source)
    order = randperm(MersenneTwister(seed), length(ordered))
    n_val = clamp(round(Int, validation_fraction * length(ordered)), 1, length(ordered) - 1)
    validation = ordered[order[1:n_val]]
    training = ordered[order[n_val+1:end]]
    training, validation
end

function write_manifest(path::AbstractString, config, training, validation, skipped; sentinel=BaselineExample[])
    doc = Dict(
        "config_path" => config.path,
        "created_at" => string(now()),
        "training" => [Dict("label" => ex.label, "source" => ex.source, "n_atoms" => n_atoms(ex)) for ex in training],
        "validation" => [Dict("label" => ex.label, "source" => ex.source, "n_atoms" => n_atoms(ex)) for ex in validation],
        "sentinel_validation" => [Dict("label" => ex.label, "source" => ex.source, "n_atoms" => n_atoms(ex)) for ex in sentinel],
        "skipped" => skipped,
    )
    open(path, "w") do io
        TOML.print(io, doc)
    end
end

"""Convert parsed TOML tables to a plain W&B-compatible nested dictionary."""
wandb_config(config) = Dict(string(k) => v for (k, v) in config)

function start_wandb(config, run_dir::AbstractString, training, validation, sentinel)
    wandb_cfg = get(config.raw, "wandb", Dict{String,Any}())
    Bool(get(wandb_cfg, "enabled", false)) || return nothing
    haskey(ENV, "WANDB_API_KEY") || error("[wandb].enabled=true requires WANDB_API_KEY in the job environment")

    name = String(get(wandb_cfg, "name", ""))
    isempty(name) && (name = basename(abspath(run_dir)))
    kwargs = Dict{Symbol,Any}(
        :project => String(get(wandb_cfg, "project", "DiffuseBioMol")),
        :name => name,
        :config => wandb_config(config.raw),
    )
    entity = String(get(wandb_cfg, "entity", ""))
    isempty(entity) || (kwargs[:entity] = entity)
    logger = Wandb.WandbLogger(; kwargs...)
    tracker = WandbTracker(logger, Bool(get(wandb_cfg, "upload_checkpoints", true)))
    Wandb.log(logger, Dict(
        "dataset/usable_structures" => length(training) + length(validation),
        "dataset/training_structures" => length(training),
        "dataset/validation_structures" => length(validation),
        "dataset/sentinel_structures" => length(sentinel),
    ); step=0)
    Wandb.save(logger, joinpath(run_dir, "manifest.toml"))
    tracker
end

log_wandb_epoch!(::Nothing, epoch::Int, train_loss::Real, scope::AbstractString, reports) = nothing

function log_wandb_epoch!(tracker::WandbTracker, epoch::Int, train_loss::Real, scope::AbstractString, reports)
    metrics = Dict{String,Any}("epoch" => epoch)
    isfinite(train_loss) && (metrics["training/cfm_loss"] = train_loss)
    for condition in ("prior", "model", "model_guided")
        for (metric, value) in aggregate(reports, condition)
            metrics["validation/$scope/$condition/$metric"] = value
        end
    end
    Wandb.log(tracker.logger, metrics; step=epoch)
    nothing
end

log_wandb_gates!(::Nothing, gates) = nothing

function log_wandb_gates!(tracker::WandbTracker, gates)
    metrics = Dict{String,Any}("gates/$key" => value for (key, value) in gates if value isa Bool)
    Wandb.log(tracker.logger, metrics)
    nothing
end

save_wandb_checkpoint!(::Nothing, path::AbstractString) = nothing
function save_wandb_checkpoint!(tracker::WandbTracker, path::AbstractString)
    tracker.upload_checkpoints && Wandb.save(tracker.logger, path)
    nothing
end

close_wandb!(::Nothing) = nothing
function close_wandb!(tracker::WandbTracker)
    close(tracker.logger)
    nothing
end

"""Select a fixed, reproducible subset for frequent inexpensive validation."""
function validation_sentinel(validation::Vector{BaselineExample}, n::Int, seed::Int)
    n > 0 || throw(ArgumentError("evaluation.sentinel_size must be positive"))
    ordered = sort(validation; by = ex -> ex.source)
    ordered[randperm(MersenneTwister(seed), length(ordered))[1:min(n, length(ordered))]]
end

"""Shuffle batch order but keep similarly sized structures together to limit padding waste."""
function length_bucket_batches(examples, batch_size::Int, rng::AbstractRNG)
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    sorted = sort(examples; by=n_atoms)
    batches = [sorted[i:min(i + batch_size - 1, end)] for i in 1:batch_size:length(sorted)]
    shuffle!(rng, batches)
end

function train_batch(model, ps, st, opt_state, batch, rng, device)
    batched_feat = batch_features([ex.feat for ex in batch])
    relpos = batch_relpos([ex.relpos for ex in batch])
    cond = batch_cond_features([ex.cond_features for ex in batch])
    pad_bias = attention_pad_bias(batched_feat.pad_mask)
    coords = batch_coords([ex.x1 for ex in batch])
    example = prepare_training_example(batched_feat, coords, cond, rng)

    # Sampling the prior/SE(3) augmentation is CPU-side and outside AD; only
    # the differentiable batch is moved to the selected single device.
    batched_feat = to_device(batched_feat, device)
    relpos, pad_bias, example = device(relpos), device(pad_bias), to_device(example, device)
    loss, back = Zygote.pullback(p -> cfm_loss(model, p, st, batched_feat, relpos, pad_bias, example)[1], ps)
    grad = back(1.0f0)[1]
    opt_state, ps = Optimisers.update(opt_state, ps, grad)
    ps, opt_state, Float64(loss)
end

function geometry_metrics(coords, ex::BaselineExample)
    c = Float32.(coords)
    (
        rmsd=Float64(aligned_rmsd(c, Float32.(ex.x1))),
        clashes=Float64(clash_count(c, ex.elements, ex.feat.chain_idx, ex.feat.res_index)),
        bond_rmsd=Float64(bond_length_rmsd(c, ex.bonds)),
        chirality_bad=Float64(chirality_count(c, ex.centers)),
    )
end

mean_metric(rows, field) = mean(getfield(row, field) for row in rows)

"""Evaluate comparable prior/untrained/trained/trained+guided trajectories with fixed seeds."""
function evaluate(model, ps, st, examples, seed::Int, n_steps::Int)
    reports = NamedTuple[]
    for (i, ex) in enumerate(examples)
        sample_seed = seed + 10_000 * i
        prior = sample_prior(MersenneTwister(sample_seed), ex.feat.chain_idx)
        push!(reports, merge((condition="prior", label=ex.label, n_atoms=n_atoms(ex)), geometry_metrics(prior, ex)))

        raw_rng = MersenneTwister(sample_seed)
        x, _ = sample_flow(model, ps, st, ex.feat, ex.relpos, ex.cond_features, raw_rng; n_steps=n_steps)
        push!(reports, merge((condition="model", label=ex.label, n_atoms=n_atoms(ex)), geometry_metrics(x, ex)))

        guided_rng = MersenneTwister(sample_seed)
        post = validity_guidance_step(ex.elements, ex.feat.chain_idx, ex.feat.res_index, ex.bonds, ex.centers)
        x_guided, _ = sample_flow(model, ps, st, ex.feat, ex.relpos, ex.cond_features, guided_rng;
            n_steps=n_steps, post_step=post)
        push!(reports, merge((condition="model_guided", label=ex.label, n_atoms=n_atoms(ex)), geometry_metrics(x_guided, ex)))
    end
    reports
end

function aggregate(reports, condition::String)
    rows = filter(r -> r.condition == condition, reports)
    isempty(rows) && error("no evaluation rows for $condition")
    Dict(
        "mean_rmsd" => mean_metric(rows, :rmsd),
        "mean_clashes" => mean_metric(rows, :clashes),
        "mean_bond_rmsd" => mean_metric(rows, :bond_rmsd),
        "mean_chirality_bad" => mean_metric(rows, :chirality_bad),
    )
end

"""Write a compact, append-only metric stream suitable for plotting or CI ingestion."""
function append_metrics(path::AbstractString, epoch::Int, train_loss::Real, scope::AbstractString, reports)
    new_file = !isfile(path)
    open(path, "a") do io
        new_file && println(io, "epoch,scope,condition,label,n_atoms,train_loss,rmsd,clashes,bond_rmsd,chirality_bad")
        for r in reports
            @printf(io, "%d,%s,%s,%s,%d,%.8f,%.8f,%.8f,%.8f,%.8f\n", epoch, scope, r.condition, r.label,
                r.n_atoms, train_loss, r.rmsd, r.clashes, r.bond_rmsd, r.chirality_bad)
        end
    end
end

function gate_report(initial_reports, final_reports, min_rmsd_improvement::Real)
    untrained, prior = aggregate(initial_reports, "model"), aggregate(initial_reports, "prior")
    trained, guided = aggregate(final_reports, "model"), aggregate(final_reports, "model_guided")
    reconstruction = trained["mean_rmsd"] <= untrained["mean_rmsd"] - min_rmsd_improvement
    beats_prior = all(trained[k] <= prior[k] for k in ("mean_rmsd", "mean_clashes", "mean_bond_rmsd", "mean_chirality_bad"))
    guidance_safe = all(guided[k] <= trained[k] for k in ("mean_clashes", "mean_bond_rmsd", "mean_chirality_bad"))
    Dict(
        "initial_untrained" => untrained, "prior" => prior, "final_trained" => trained, "final_guided" => guided,
        "reconstruction_pass" => reconstruction,
        "trained_beats_prior_on_all_metrics" => beats_prior,
        "guidance_non_regression" => guidance_safe,
        "all_passed" => reconstruction && beats_prior && guidance_safe,
    )
end

function checkpoint(path::AbstractString, epoch::Int, ps, st, opt_state, rng, config_path, initial_reports)
    temporary = path * ".tmp"
    serialize(temporary, (epoch=epoch, ps=ps, st=st, opt_state=opt_state, rng=rng,
        config_path=config_path, initial_reports=initial_reports))
    mv(temporary, path; force=true)
end

function selected_device(use_gpu::Bool)
    use_gpu || return identity
    Base.find_package("CUDA") === nothing && error("--gpu requires CUDA.jl in the active Julia environment")
    # CUDA remains an optional trigger dependency; importing it here makes
    # Lux.gpu_device() select the first visible CUDA device without adding it
    # to DiffuseBioMol's package dependencies.
    @eval using CUDA
    Lux.gpu_device()
end

function main(config_path::AbstractString, run_dir::AbstractString; resume::Bool=false, device=identity)
    config = load_config(config_path)
    mkpath(run_dir)
    data, training_cfg, eval_cfg = config.data, config.training, config.evaluation
    seed = Int(require_key(training_cfg, "seed"))
    rng = MersenneTwister(seed)
    examples, skipped = load_corpus(String(require_key(data, "data_dir")); max_atoms=Int(require_key(data, "max_atoms")))
    max_structures = Int(get(data, "max_structures", 0))
    if max_structures > 0 && length(examples) > max_structures
        examples = sort(examples; by = ex -> ex.source)[randperm(MersenneTwister(seed), length(examples))[1:max_structures]]
    end
    min_structures = Int(get(data, "min_structures", 3))
    length(examples) >= min_structures || error("only $(length(examples)) usable structures; data.min_structures requires $min_structures")
    training, validation = split_examples(examples, seed, Float64(require_key(data, "validation_fraction")))
    sentinel = validation_sentinel(validation, Int(get(eval_cfg, "sentinel_size", length(validation))), seed + 1)
    write_manifest(joinpath(run_dir, "manifest.toml"), config, training, validation, skipped; sentinel)
    println("Loaded $(length(examples)) usable structures: $(length(training)) train, $(length(validation)) validation; $(length(skipped)) skipped.")

    m = config.model
    model = build_model(ModelConfig(
        d_single=Int(require_key(m, "d_single")), d_pair=Int(require_key(m, "d_pair")),
        n_heads=Int(require_key(m, "n_heads")), n_pairformer_layers=Int(require_key(m, "n_pairformer_layers")),
        n_dit_layers=Int(require_key(m, "n_dit_layers")), d_time=Int(require_key(m, "d_time")),
        d_hidden_mult=Int(require_key(m, "d_hidden_mult")),
    ))
    checkpoint_path = joinpath(run_dir, "checkpoint_latest.jls")
    # Evaluation is deliberately CPU-side today: the geometry functions are
    # scalar CPU code and the single-structure sampler constructs its prior on
    # CPU.  Training can still use one GPU by passing `Lux.gpu_device()` as
    # `device`; parameters are copied back only at checkpoint/evaluation time.
    host_device = Lux.cpu_device()
    tracker = start_wandb(config, run_dir, training, validation, sentinel)
    if resume
        isfile(checkpoint_path) || throw(ArgumentError("--resume requested but $checkpoint_path does not exist"))
        saved = deserialize(checkpoint_path)
        saved.config_path == abspath(config_path) || throw(ArgumentError("checkpoint was created with a different config file"))
        hasproperty(saved, :initial_reports) || error("checkpoint predates baseline-gate support; start a fresh run")
        ps, st, opt_state, rng, start_epoch = device(saved.ps), device(saved.st), device(saved.opt_state), saved.rng, saved.epoch + 1
        initial_reports = saved.initial_reports
        println("Resuming from epoch $(saved.epoch).")
    else
        ps, st = Lux.setup(rng, model)
        ps, st = device(ps), device(st)
        opt_state = Optimisers.setup(Optimisers.Adam(Float32(require_key(training_cfg, "learning_rate"))), ps)
        start_epoch = 1
        # A full untrained validation pass is done exactly once so the final
        # reconstruction gate compares against the actual initialization.
        initial_reports = evaluate(model, host_device(ps), host_device(st), validation, seed, Int(require_key(eval_cfg, "sample_steps")))
        append_metrics(joinpath(run_dir, "metrics.csv"), 0, NaN, "full_initial", initial_reports)
        log_wandb_epoch!(tracker, 0, NaN, "full_initial", initial_reports)
    end

    epochs, checkpoint_every = Int(require_key(training_cfg, "epochs")), Int(require_key(training_cfg, "checkpoint_every"))
    full_validation_every = Int(get(eval_cfg, "full_validation_every", epochs))
    full_validation_every > 0 || throw(ArgumentError("evaluation.full_validation_every must be positive"))
    final_reports = initial_reports
    for epoch in start_epoch:epochs
        losses = Float64[]
        for batch in length_bucket_batches(training, Int(require_key(training_cfg, "batch_size")), rng)
            ps, opt_state, loss = train_batch(model, ps, st, opt_state, batch, rng, device)
            push!(losses, loss)
        end
        train_loss = mean(losses)
        println("epoch $epoch/$epochs: train CFM loss = $(@sprintf("%.6f", train_loss))")
        if epoch % checkpoint_every == 0 || epoch == epochs
            full = epoch % full_validation_every == 0 || epoch == epochs
            eval_examples, scope = full ? (validation, "full") : (sentinel, "sentinel")
            final_reports = evaluate(model, host_device(ps), host_device(st), eval_examples, seed, Int(require_key(eval_cfg, "sample_steps")))
            append_metrics(joinpath(run_dir, "metrics.csv"), epoch, train_loss, scope, final_reports)
            log_wandb_epoch!(tracker, epoch, train_loss, scope, final_reports)
            checkpoint(checkpoint_path, epoch, host_device(ps), host_device(st), host_device(opt_state), rng,
                abspath(config_path), initial_reports)
            save_wandb_checkpoint!(tracker, checkpoint_path)
        end
    end
    gates = gate_report(initial_reports, final_reports, Float64(require_key(eval_cfg, "min_rmsd_improvement")))
    open(joinpath(run_dir, "gates.toml"), "w") do io
        TOML.print(io, gates)
    end
    log_wandb_gates!(tracker, gates)
    if tracker !== nothing
        Wandb.save(tracker.logger, joinpath(run_dir, "metrics.csv"))
        Wandb.save(tracker.logger, joinpath(run_dir, "gates.toml"))
    end
    close_wandb!(tracker)
    println("Gate result: all_passed = $(gates["all_passed"]) (see $(joinpath(run_dir, "gates.toml"))).")
    gates["all_passed"] || println("Baseline did not clear gates; do not start fast-sampler work from this checkpoint.")
    (model=model, ps=ps, st=st, gates=gates)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 2 || error("usage: julia --project=. scripts/train_baseline.jl CONFIG.toml RUN_DIR [--resume]")
    main(ARGS[1], ARGS[2]; resume="--resume" in ARGS[3:end], device=selected_device("--gpu" in ARGS[3:end]))
end
