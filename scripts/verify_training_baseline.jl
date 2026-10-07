#=
Real-data CPU training proof using the production baseline runner.

    julia --project=. scripts/verify_training_baseline.jl DATA_DIR RUN_DIR

Requires at least six local PDB/mmCIF sources; does not download during timing.
Uses 128-token residue-complete crops, a small model and two BLAS threads.
Checks a fixed-example learning signal, three training epochs, checkpoint
resume equivalence and finite sampling. Writes evidence.toml and per-step CSVs.
This is a pipeline proof, not a protein-quality or accelerator-scaling gate.
RUN_DIR must not already exist to avoid mixing evidence across executions.
=#
include(joinpath(@__DIR__, "train_baseline.jl"))
using LinearAlgebra

function verify_training_baseline(data_dir, run_dir)
    ispath(run_dir) && error("Use a new output directory: $run_dir")
    mkpath(run_dir)
    BLAS.set_num_threads(2)
    config = Dict(
        "data" => Dict("data_dir" => abspath(data_dir), "max_atoms" => 128,
            "oversize_policy" => "crop", "crop_strategy" => "mixed",
            "max_candidate_files" => 6, "max_structures" => 6, "min_structures" => 6,
            "validation_fraction" => 0.33, "sentinel_size" => 2,
            "cache_path" => abspath(joinpath(run_dir, "corpus_cache.jls"))),
        "model" => Dict("d_single" => 16, "d_pair" => 8, "n_heads" => 4,
            "n_pairformer_layers" => 1, "n_dit_layers" => 1, "d_time" => 16, "d_hidden_mult" => 2),
        "training" => Dict("seed" => 17, "epochs" => 3, "batch_size" => 2,
            "learning_rate" => 0.001, "checkpoint_every" => 1, "profile_steps" => true),
        "evaluation" => Dict("sample_steps" => 3, "sentinel_size" => 2,
            "full_validation_every" => 3, "min_rmsd_improvement" => 0.0, "infill_enabled" => false),
        "wandb" => Dict("enabled" => false),
    )
    config_path = joinpath(run_dir, "config.toml")
    write_config() = open(io -> TOML.print(io, config), config_path, "w")
    write_config()
    println("=== Uninterrupted real-data training (3 epochs) ===")
    full_dir = joinpath(run_dir, "uninterrupted")
    full_s = @elapsed main(config_path, full_dir)
    full = deserialize(joinpath(full_dir, "checkpoint_latest.jls"))

    println("=== Restart equivalence: 2 epochs, then resume to 3 ===")
    config["training"]["epochs"] = 2
    write_config()
    resumed_dir = joinpath(run_dir, "resumed")
    main(config_path, resumed_dir)
    config["training"]["epochs"] = 3
    write_config()
    main(config_path, resumed_dir; resume=true)
    resumed = deserialize(joinpath(resumed_dir, "checkpoint_latest.jls"))
    @assert full.epoch == resumed.epoch == 3
    @assert isequal(full.ps, resumed.ps) "Resumed parameters differ"
    @assert isequal(full.st, resumed.st) "Resumed layer state differs"
    # Optimiser leaves are mutable structs; compare their serialized values.
    encoded(x) = (io = IOBuffer(); serialize(io, x); take!(io))
    @assert encoded(full.opt_state) == encoded(resumed.opt_state) "Resumed optimizer differs"
    @assert rand(deepcopy(full.rng), 10) == rand(deepcopy(resumed.rng), 10) "Resumed RNG differs"

    println("=== Fixed real crop: demonstrate learnability ===")
    sources, _ = load_corpus(data_dir; max_atoms=128, max_candidate_files=6,
        selection_seed=17, oversize_policy="crop")
    ex = crop_example(first(sources), 128, "sequence", MersenneTwister(5))
    rng = MersenneTwister(19)
    model = build_model(ModelConfig(d_single=16, d_pair=8, n_heads=4,
        n_pairformer_layers=1, n_dit_layers=1, d_time=16, d_hidden_mult=2))
    ps, st = Lux.setup(rng, model)
    example = prepare_training_example(ex.feat, ex.x1, ex.cond_features, rng)
    loss(p) = cfm_loss(model, p, st, ex.feat, ex.relpos, example)[1]
    initial_loss = Float64(loss(ps))
    opt = Optimisers.setup(Optimisers.Adam(0.01f0), ps)
    fit_s = @elapsed for _ in 1:150
        value, back = Zygote.pullback(loss, ps)
        @assert isfinite(value)
        opt, ps = Optimisers.update(opt, ps, back(1.0f0)[1])
    end
    final_loss = Float64(loss(ps))
    @assert final_loss < initial_loss / 5 "Fixed-example loss did not decrease sufficiently"
    sample, _ = sample_flow(model, ps, st, ex.feat, ex.relpos, ex.cond_features,
        MersenneTwister(23); n_steps=3)
    @assert size(sample) == size(ex.x1) && all(isfinite, sample)
    evidence = Dict("objective" => "flow_matching", "device" => "CPU",
        "julia_version" => string(VERSION), "cpu" => Sys.CPU_NAME,
        "julia_threads" => Threads.nthreads(), "blas_threads" => BLAS.get_num_threads(),
        "sources" => 6, "training_sources" => 4, "held_out_sources" => 2,
        "crop_atom_cap" => 128, "epochs" => 3, "training_presentations" => 12,
        "resume_equivalent" => true, "finite_sample" => true,
        "fixed_example_initial_loss" => initial_loss, "fixed_example_final_loss" => final_loss,
        "fixed_example_updates" => 150, "fixed_example_seconds" => fit_s,
        "uninterrupted_seconds_including_compile_and_evaluation" => full_s,
        "protein_quality_established" => false, "diffusion_implemented" => false)
    open(io -> TOML.print(io, evidence), joinpath(run_dir, "evidence.toml"), "w")
    println("Training proof passed. Evidence: $(joinpath(run_dir, "evidence.toml"))")
    evidence
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("usage: verify_training_baseline.jl DATA_DIR NEW_RUN_DIR")
    verify_training_baseline(ARGS...)
end
