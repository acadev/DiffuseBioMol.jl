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
    julia --project=. scripts/train_baseline.jl configs/baseline.toml runs/baseline --prepare-only

The default configuration deliberately has a placeholder `data_dir`; copy it
and point it at a local, immutable corpus before running a real experiment.
Metrics are appended to `metrics.csv`, the exact split is saved to
`manifest.toml`, and `checkpoint_latest.jls` is atomically replaced after each
checkpoint.  Checkpoint state includes model parameters, optimizer state, RNG,
and the next epoch, so interruption does not change the training trajectory.

The source tokens and coordinates are cached as `corpus_cache.jls` in the run
directory by default. Pairwise features are deliberately built only after a
bounded crop is selected, so the cache never contains full-chain O(N²)
tensors. Re-launching the same run, or pointing another run at a shared
`data.cache_path`, reuses that cache whenever the source file paths, sizes,
modification times, and `max_atoms` match. This avoids re-parsing mmCIF files
on every training attempt.

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
const CORPUS_CACHE_VERSION = 4

"""Lightweight cached source: tokens only, with no O(N²) pair features."""
struct BaselineSource
    label::String
    source::String
    tokens::Vector{AtomToken}
end

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

n_atoms(record) = length(record.tokens)

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

function make_example(label::AbstractString, source::AbstractString, tokens::Vector{AtomToken})
    isempty(tokens) && error("no tokens after selecting/cropping structure")
    feat = featurize(tokens)
    BaselineExample(
        String(label), abspath(source), tokens, feat, relpos_buckets(feat),
        target_coordinates(tokens), constraint_features(no_constraints(length(tokens))),
        backbone_bonds(tokens), chiral_centers(tokens), [t.element for t in tokens],
    )
end

make_source(label::AbstractString, source::AbstractString, tokens::Vector{AtomToken}) =
    BaselineSource(String(label), abspath(source), tokens)

function load_source(path::AbstractString)
    residues = parse_structure(path)
    isempty(residues) && error("no residues parsed")
    chain = largest_chain(residues)
    tokens = tokenize_structure(restrict_to_chain(residues, chain))
    isempty(tokens) && error("no tokens after selecting largest chain")
    make_source(splitext(basename(path))[1], path, tokens)
end

"""Token-index groups, one complete residue per group, in source order."""
function residue_groups(record)
    groups = Vector{Vector{Int}}()
    previous = nothing
    for (i, token) in enumerate(record.tokens)
        key = (token.chain_id, token.res_index)
        if previous != key
            push!(groups, Int[])
            previous = key
        end
        push!(groups[end], i)
    end
    groups
end

function residue_center(record, group)
    observed = [i for i in group if !record.tokens[i].is_virtual && record.tokens[i].coord !== nothing]
    isempty(observed) && return zeros(Float64, 3)
    coords = [record.tokens[i].coord for i in observed]
    [sum(c[j] for c in coords) / length(coords) for j in 1:3]
end

function sequence_crop_indices(record, groups, max_atoms::Int, rng::AbstractRNG)
    valid_starts = findall(group -> length(group) <= max_atoms, groups)
    isempty(valid_starts) && error("no residue fits within crop_max_atoms=$max_atoms")
    start = rand(rng, valid_starts)
    selected = Int[]
    used = 0
    for group in groups[start:end]
        used + length(group) <= max_atoms || break
        append!(selected, group)
        used += length(group)
    end
    selected
end

function spatial_crop_indices(record, groups, max_atoms::Int, rng::AbstractRNG)
    centers = [residue_center(record, group) for group in groups]
    valid_anchors = findall(group -> length(group) <= max_atoms, groups)
    isempty(valid_anchors) && error("no residue fits within crop_max_atoms=$max_atoms")
    anchor = rand(rng, valid_anchors)
    ranked = sortperm(1:length(groups); by=i -> (sum(abs2, centers[i] .- centers[anchor]), i))
    selected = Int[]
    used = 0
    for i in ranked
        used + length(groups[i]) <= max_atoms || continue
        append!(selected, groups[i])
        used += length(groups[i])
    end
    sort!(selected)
end

"""Draw a residue-complete bounded crop; source and residue identifiers are preserved."""
function crop_example(source::BaselineSource, max_atoms::Int, strategy::AbstractString, rng::AbstractRNG; crop_id::Int=0)
    n_atoms(source) <= max_atoms && return make_example(source.label, source.source, source.tokens)
    groups = residue_groups(source)
    selected_strategy = strategy == "mixed" ? (rand(rng, Bool) ? "sequence" : "spatial") : String(strategy)
    indices = if selected_strategy == "sequence"
        sequence_crop_indices(source, groups, max_atoms, rng)
    elseif selected_strategy == "spatial"
        spatial_crop_indices(source, groups, max_atoms, rng)
    else
        throw(ArgumentError("crop_strategy must be sequence, spatial, or mixed; got $strategy"))
    end
    make_example("$(source.label)#$(selected_strategy)-$(crop_id)", source.source, source.tokens[indices])
end

"""One deterministic, epoch-varying crop per source; all crops retain their original source path."""
function materialize_crops(sources::Vector{BaselineSource}, max_atoms::Int, strategy::AbstractString, seed::Int, epoch::Int)
    rng = MersenneTwister(seed + 1_000_003 * epoch)
    [crop_example(ex, max_atoms, strategy, rng; crop_id=epoch) for ex in sources]
end

"""Materialize one source batch without retaining pair features for an entire epoch."""
function materialize_crop_batch(sources::Vector{BaselineSource}, max_atoms::Int, strategy::AbstractString,
                                seed::Int, epoch::Int, batch_number::Int)
    rng = MersenneTwister(seed + 1_000_003 * epoch + 7_919 * batch_number)
    [crop_example(ex, max_atoms, strategy, rng; crop_id=epoch) for ex in sources]
end

"""A cheap, deterministic fingerprint for invalidating a preprocessed corpus cache."""
function corpus_signature(files)
    [(path=abspath(path), size=filesize(path), mtime=stat(path).mtime) for path in files]
end

function load_cached_corpus(cache_path::AbstractString, signature, max_atoms::Int,
                            max_candidate_files::Int, selection_seed::Int, oversize_policy::AbstractString)
    isfile(cache_path) || return nothing
    try
        cached = deserialize(cache_path)
        if cached.version == CORPUS_CACHE_VERSION && cached.max_atoms == max_atoms &&
           cached.max_candidate_files == max_candidate_files && cached.selection_seed == selection_seed &&
           cached.oversize_policy == oversize_policy &&
           cached.signature == signature
            println("Corpus cache: loaded $(length(cached.sources)) lightweight sources from $cache_path.")
            return cached.sources, cached.skipped
        end
        println("Corpus cache: source files or max_atoms changed; rebuilding $cache_path.")
    catch err
        @warn "Corpus cache at $cache_path could not be read; rebuilding it" exception=(err, catch_backtrace())
    end
    nothing
end

function write_corpus_cache(cache_path::AbstractString, signature, max_atoms::Int,
                            max_candidate_files::Int, selection_seed::Int, oversize_policy::AbstractString, sources, skipped)
    parent = dirname(cache_path)
    isempty(parent) || mkpath(parent)
    temporary = cache_path * ".tmp.$(getpid()).$(rand(UInt))"
    try
        serialize(temporary, (version=CORPUS_CACHE_VERSION, signature=signature,
            max_atoms=max_atoms, max_candidate_files=max_candidate_files,
            selection_seed=selection_seed, oversize_policy=String(oversize_policy), sources=sources, skipped=skipped))
        mv(temporary, cache_path; force=true)
    finally
        isfile(temporary) && rm(temporary; force=true)
    end
    println("Corpus cache: wrote $(length(sources)) lightweight sources to $cache_path.")
end

"""Load a fixed corpus with observable progress and optional persistent preprocessing cache."""
function load_corpus(data_dir::AbstractString; max_atoms::Int, cache_path::Union{Nothing,AbstractString}=nothing,
                     progress_every::Int=25, max_candidate_files::Int=0, selection_seed::Int=0,
                     oversize_policy::AbstractString="skip")
    files = list_structure_files(data_dir; recursive=true)
    isempty(files) && throw(ArgumentError("no PDB/mmCIF files found under $data_dir"))
    progress_every > 0 || throw(ArgumentError("progress_every must be positive"))
    max_candidate_files >= 0 || throw(ArgumentError("max_candidate_files must be nonnegative"))
    oversize_policy in ("skip", "crop") || throw(ArgumentError("oversize_policy must be skip or crop"))
    if max_candidate_files > 0 && length(files) > max_candidate_files
        selection = randperm(MersenneTwister(selection_seed), length(files))[1:max_candidate_files]
        files = sort(files[selection])
        println("Corpus loading: selected $max_candidate_files candidate files deterministically from the source corpus.")
    end
    signature = corpus_signature(files)
    if cache_path !== nothing
        cached = load_cached_corpus(cache_path, signature, max_atoms, max_candidate_files, selection_seed, oversize_policy)
        cached === nothing || return cached
    end

    println("Corpus loading: parsing $(length(files)) files from $data_dir (progress every $progress_every files).")
    sources = BaselineSource[]
    skipped = String[]
    for (i, path) in enumerate(files)
        if i == 1 || i % progress_every == 0 || i == length(files)
            println("Corpus loading: file $i/$(length(files)) ($(basename(path))); $(length(sources)) usable so far.")
        end
        try
            source = load_source(path)
            if n_atoms(source) <= max_atoms || oversize_policy == "crop"
                push!(sources, source)
            else
                push!(skipped, "$path ($(n_atoms(source)) atoms > max_atoms=$max_atoms)")
            end
        catch err
            push!(skipped, "$path ($err)")
        end
    end
    isempty(sources) && error("no usable structures in $data_dir")
    cache_path === nothing || write_corpus_cache(cache_path, signature, max_atoms,
        max_candidate_files, selection_seed, oversize_policy, sources, skipped)
    sources, skipped
end

"""A deterministic structure-level split; never split atoms from one structure across sets."""
function split_examples(examples::Vector{BaselineSource}, seed::Int, validation_fraction::Real)
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

log_wandb_infill!(::Nothing, epoch::Int, reports) = nothing
function log_wandb_infill!(tracker::WandbTracker, epoch::Int, reports)
    metrics = Dict("infill/$key" => value for (key, value) in aggregate_infill(reports))
    metrics["epoch"] = epoch
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
function validation_sentinel(validation, n::Int, seed::Int)
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

"""Select a contiguous, residue-complete observed motif for conditional infilling."""
function infill_fixed_mask(ex::BaselineExample, fraction::Real, rng::AbstractRNG)
    0 < fraction < 1 || throw(ArgumentError("infill_fixed_fraction must lie in (0, 1)"))
    groups = residue_groups(ex)
    length(groups) >= 2 || return falses(n_atoms(ex))
    n_groups = clamp(round(Int, fraction * length(groups)), 1, length(groups) - 1)
    start = rand(rng, 1:(length(groups) - n_groups + 1))
    fixed = falses(n_atoms(ex))
    for group in groups[start:start+n_groups-1]
        for i in group
            fixed[i] = !ex.tokens[i].is_virtual
        end
    end
    fixed
end

function infill_conditioning(ex::BaselineExample, fraction::Real, rng::AbstractRNG)
    fixed = infill_fixed_mask(ex, fraction, rng)
    constraints = AtomConstraints(fixed, falses(n_atoms(ex)), zeros(Float32, n_atoms(ex)), falses(n_atoms(ex)), Float32.(ex.x1))
    constraint_features(constraints), fixed, constraints.fixed_coord
end

function train_batch(model, ps, st, opt_state, batch, rng, device; infill_probability::Real=0.0,
                     infill_fixed_fraction::Real=0.2)
    0 <= infill_probability <= 1 || throw(ArgumentError("infill_probability must lie in [0, 1]"))
    batched_feat = batch_features([ex.feat for ex in batch])
    relpos = batch_relpos([ex.relpos for ex in batch])
    conditioned = [rand(rng) < infill_probability ? infill_conditioning(ex, infill_fixed_fraction, rng) :
        (ex.cond_features, falses(n_atoms(ex)), zeros(Float32, 3, n_atoms(ex))) for ex in batch]
    cond = batch_cond_features(first.(conditioned))
    pad_bias = attention_pad_bias(batched_feat.pad_mask)
    coords = batch_coords([ex.x1 for ex in batch])
    is_fixed = reduce(hcat, [vcat(mask, falses(size(coords, 2) - length(mask))) for (_, mask, _) in conditioned])
    fixed_coord = batch_coords(last.(conditioned))
    example = prepare_training_example(batched_feat, coords, cond, rng; is_fixed, fixed_coord)

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

"""Held-out motif-infill evaluation: fixed residues are clamped, all other real atoms are scored."""
function evaluate_infill(model, ps, st, examples, seed::Int, n_steps::Int, fixed_fraction::Real)
    reports = NamedTuple[]
    for (i, ex) in enumerate(examples)
        mask_rng = MersenneTwister(seed + 20_000 * i)
        cond, is_fixed, fixed_coord = infill_conditioning(ex, fixed_fraction, mask_rng)
        sample_rng = MersenneTwister(seed + 20_000 * i + 1)
        x, _ = sample_flow(model, ps, st, ex.feat, ex.relpos, cond, sample_rng;
            n_steps=n_steps, is_fixed, fixed_coord)
        generated = .!is_fixed .& .!ex.feat.is_virtual
        generated_rmsd = any(generated) ? sqrt(sum(abs2, Float32.(x[:, generated]) .- Float32.(ex.x1[:, generated])) / count(generated)) : NaN
        fixed_rmsd = any(is_fixed) ? sqrt(sum(abs2, Float32.(x[:, is_fixed]) .- Float32.(ex.x1[:, is_fixed])) / count(is_fixed)) : NaN
        push!(reports, (label=ex.label, n_atoms=n_atoms(ex), fixed_atoms=count(is_fixed),
            generated_atoms=count(generated), generated_rmsd=Float64(generated_rmsd), fixed_rmsd=Float64(fixed_rmsd)))
    end
    reports
end

function aggregate_infill(reports)
    isempty(reports) && error("no infill reports")
    Dict("mean_generated_rmsd" => mean(r.generated_rmsd for r in reports),
        "mean_fixed_rmsd" => mean(r.fixed_rmsd for r in reports),
        "mean_fixed_atoms" => mean(r.fixed_atoms for r in reports))
end

function append_infill_metrics(path::AbstractString, epoch::Int, reports)
    new_file = !isfile(path)
    open(path, "a") do io
        new_file && println(io, "epoch,label,n_atoms,fixed_atoms,generated_atoms,generated_rmsd,fixed_rmsd")
        for r in reports
            @printf(io, "%d,%s,%d,%d,%d,%.8f,%.8f\n", epoch, r.label, r.n_atoms, r.fixed_atoms,
                r.generated_atoms, r.generated_rmsd, r.fixed_rmsd)
        end
    end
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

"""Assess final held-out samples against the matched prior from that same evaluation."""
function gate_report(initial_reports, final_reports, min_rmsd_improvement::Real; initial_infill=nothing, final_infill=nothing)
    # Every evaluation includes a matched prior sample. Comparing final model
    # samples to that prior avoids an expensive CPU-only epoch-zero model pass
    # and keeps the reconstruction comparison on the same held-out set.
    prior = aggregate(final_reports, "prior")
    trained, guided = aggregate(final_reports, "model"), aggregate(final_reports, "model_guided")
    reconstruction = trained["mean_rmsd"] <= prior["mean_rmsd"] - min_rmsd_improvement
    beats_prior = all(trained[k] <= prior[k] for k in ("mean_rmsd", "mean_clashes", "mean_bond_rmsd", "mean_chirality_bad"))
    guidance_safe = all(guided[k] <= trained[k] for k in ("mean_clashes", "mean_bond_rmsd", "mean_chirality_bad"))
    infill_pass = if initial_infill === nothing || final_infill === nothing
        true
    else
        aggregate_infill(final_infill)["mean_generated_rmsd"] <=
            aggregate_infill(initial_infill)["mean_generated_rmsd"] - min_rmsd_improvement
    end
    Dict(
        "initial_untrained" => initial_reports === nothing ? Dict{String,Float64}() : aggregate(initial_reports, "model"),
        "prior" => prior, "final_trained" => trained, "final_guided" => guided,
        "reconstruction_pass" => reconstruction,
        "trained_beats_prior_on_all_metrics" => beats_prior,
        "guidance_non_regression" => guidance_safe,
        "infill_reconstruction_pass" => infill_pass,
        "all_passed" => reconstruction && beats_prior && guidance_safe && infill_pass,
    )
end

function checkpoint(path::AbstractString, epoch::Int, ps, st, opt_state, rng, config_path, initial_reports, initial_infill)
    temporary = path * ".tmp"
    serialize(temporary, (epoch=epoch, ps=ps, st=st, opt_state=opt_state, rng=rng,
        config_path=config_path, initial_reports=initial_reports, initial_infill=initial_infill))
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

"""Use a format-specific filename so an incompatible cache is never deserialized."""
function versioned_cache_path(path::AbstractString)
    base, extension = splitext(abspath(path))
    "$(base).v$(CORPUS_CACHE_VERSION)$(extension)"
end

function main(config_path::AbstractString, run_dir::AbstractString; resume::Bool=false, prepare_only::Bool=false, device=identity)
    config = load_config(config_path)
    mkpath(run_dir)
    data, training_cfg, eval_cfg = config.data, config.training, config.evaluation
    seed = Int(require_key(training_cfg, "seed"))
    rng = MersenneTwister(seed)
    configured_cache = String(get(data, "cache_path", joinpath(run_dir, "corpus_cache.jls")))
    cache_path = isempty(configured_cache) ? nothing : versioned_cache_path(configured_cache)
    oversize_policy = String(get(data, "oversize_policy", "skip"))
    crop_strategy = String(get(data, "crop_strategy", "mixed"))
    crop_strategy in ("sequence", "spatial", "mixed") || error("data.crop_strategy must be sequence, spatial, or mixed")
    Int(require_key(data, "max_atoms")) > 0 || error("data.max_atoms must be positive for baseline training")
    sources, skipped = load_corpus(String(require_key(data, "data_dir"));
        max_atoms=Int(require_key(data, "max_atoms")), cache_path,
        progress_every=Int(get(data, "load_progress_every", 25)),
        max_candidate_files=Int(get(data, "max_candidate_files", 0)), selection_seed=seed, oversize_policy)
    max_structures = Int(get(data, "max_structures", 0))
    if max_structures > 0 && length(sources) > max_structures
        sources = sort(sources; by = ex -> ex.source)[randperm(MersenneTwister(seed), length(sources))[1:max_structures]]
    end
    min_structures = Int(get(data, "min_structures", 3))
    length(sources) >= min_structures || error("only $(length(sources)) usable structures; data.min_structures requires $min_structures")
    training, validation = split_examples(sources, seed, Float64(require_key(data, "validation_fraction")))
    # Crops are drawn only after the source-level split, so no source chain can
    # leak a different crop into train and validation.
    max_atoms = Int(require_key(data, "max_atoms"))
    sentinel_sources = validation_sentinel(validation, Int(get(eval_cfg, "sentinel_size", length(validation))), seed + 1)
    sentinel = materialize_crops(sentinel_sources, max_atoms, crop_strategy, seed, 0)
    infill_enabled = Bool(get(eval_cfg, "infill_enabled", false))
    infill_fixed_fraction = Float64(get(eval_cfg, "infill_fixed_fraction", 0.2))
    infill_size = Int(get(eval_cfg, "infill_size", min(16, length(validation))))
    infill_sources = infill_enabled ? validation_sentinel(validation, infill_size, seed + 2) : BaselineSource[]
    infill_examples = infill_enabled ? materialize_crops(infill_sources, max_atoms, crop_strategy, seed, 0) : BaselineExample[]
    write_manifest(joinpath(run_dir, "manifest.toml"), config, training, validation, skipped; sentinel)
    println("Loaded $(length(sources)) usable structures: $(length(training)) train, $(length(validation)) validation; $(length(skipped)) skipped.")
    if prepare_only
        resume && throw(ArgumentError("--prepare-only cannot be combined with --resume"))
        println("Corpus preparation complete; cache is ready for a later training launch.")
        return (training=training, validation=validation, sentinel=sentinel, skipped=skipped)
    end

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
        ps, st, opt_state, rng, start_epoch = device(saved.ps), device(saved.st), device(saved.opt_state), saved.rng, saved.epoch + 1
        initial_reports = nothing
        initial_infill = nothing
        println("Resuming from epoch $(saved.epoch).")
    else
        ps, st = Lux.setup(rng, model)
        ps, st = device(ps), device(st)
        opt_state = Optimisers.setup(Optimisers.Adam(Float32(require_key(training_cfg, "learning_rate"))), ps)
        start_epoch = 1
        # Do not run epoch-zero validation. On a 10k corpus it would be a
        # large CPU-only sampling job before the first H100 kernel. Each later
        # validation already includes a matched prior for gate comparison.
        initial_reports = nothing
        initial_infill = nothing
    end

    epochs, checkpoint_every = Int(require_key(training_cfg, "epochs")), Int(require_key(training_cfg, "checkpoint_every"))
    full_validation_every = Int(get(eval_cfg, "full_validation_every", epochs))
    full_validation_every > 0 || throw(ArgumentError("evaluation.full_validation_every must be positive"))
    final_reports = initial_reports
    final_infill = initial_infill
    println("Starting training at epoch $start_epoch; GPU work begins with the first batch.")
    for epoch in start_epoch:epochs
        losses = Float64[]
        source_batches = length_bucket_batches(training, Int(require_key(training_cfg, "batch_size")), rng)
        println("epoch $epoch/$epochs: training $(length(source_batches)) crop batches lazily.")
        batch_progress_every = max(1, cld(length(source_batches), 10))
        for (batch_number, source_batch) in enumerate(source_batches)
            # Creating all crops up front would retain an O(N²) pair matrix for
            # every training source and can delay the first GPU batch for many
            # minutes. Keep only one materialized batch alive at a time.
            batch = materialize_crop_batch(source_batch, max_atoms, crop_strategy, seed, epoch, batch_number)
            if batch_number == 1
                println("epoch $epoch/$epochs: first crop batch ready; compiling/launching GPU training.")
            elseif batch_number % batch_progress_every == 0 || batch_number == length(source_batches)
                println("epoch $epoch/$epochs: batch $batch_number/$(length(source_batches)).")
            end
            ps, opt_state, loss = train_batch(model, ps, st, opt_state, batch, rng, device;
                infill_probability=Float64(get(training_cfg, "infill_probability", 0.0)), infill_fixed_fraction)
            push!(losses, loss)
        end
        train_loss = mean(losses)
        println("epoch $epoch/$epochs: train CFM loss = $(@sprintf("%.6f", train_loss))")
        if epoch % checkpoint_every == 0 || epoch == epochs
            full = epoch % full_validation_every == 0 || epoch == epochs
            # Full held-out crops are deliberately materialized only when a
            # full evaluation is due; startup needs only the small sentinel.
            eval_examples, scope = full ? (materialize_crops(validation, max_atoms, crop_strategy, seed, 0), "full") : (sentinel, "sentinel")
            final_reports = evaluate(model, host_device(ps), host_device(st), eval_examples, seed, Int(require_key(eval_cfg, "sample_steps")))
            append_metrics(joinpath(run_dir, "metrics.csv"), epoch, train_loss, scope, final_reports)
            log_wandb_epoch!(tracker, epoch, train_loss, scope, final_reports)
            final_infill = infill_enabled ? evaluate_infill(model, host_device(ps), host_device(st), infill_examples,
                seed, Int(require_key(eval_cfg, "sample_steps")), infill_fixed_fraction) : nothing
            if final_infill !== nothing
                append_infill_metrics(joinpath(run_dir, "infill_metrics.csv"), epoch, final_infill)
                log_wandb_infill!(tracker, epoch, final_infill)
            end
            checkpoint(checkpoint_path, epoch, host_device(ps), host_device(st), host_device(opt_state), rng,
                abspath(config_path), initial_reports, initial_infill)
            save_wandb_checkpoint!(tracker, checkpoint_path)
        end
    end
    gates = gate_report(initial_reports, final_reports, Float64(require_key(eval_cfg, "min_rmsd_improvement"));
        initial_infill, final_infill=infill_enabled ? final_infill : nothing)
    open(joinpath(run_dir, "gates.toml"), "w") do io
        TOML.print(io, gates)
    end
    log_wandb_gates!(tracker, gates)
    if tracker !== nothing
        Wandb.save(tracker.logger, joinpath(run_dir, "metrics.csv"))
        isfile(joinpath(run_dir, "infill_metrics.csv")) && Wandb.save(tracker.logger, joinpath(run_dir, "infill_metrics.csv"))
        Wandb.save(tracker.logger, joinpath(run_dir, "gates.toml"))
    end
    close_wandb!(tracker)
    println("Gate result: all_passed = $(gates["all_passed"]) (see $(joinpath(run_dir, "gates.toml"))).")
    gates["all_passed"] || println("Baseline did not clear gates; do not start fast-sampler work from this checkpoint.")
    (model=model, ps=ps, st=st, gates=gates)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 2 || error("usage: julia --project=. scripts/train_baseline.jl CONFIG.toml RUN_DIR [--resume] [--prepare-only] [--gpu]")
    flags = ARGS[3:end]
    prepare_only = "--prepare-only" in flags
    main(ARGS[1], ARGS[2]; resume="--resume" in flags, prepare_only,
        device=selected_device("--gpu" in flags && !prepare_only))
end
