"""
GPU-accelerated training script for DiffuseBioMol.jl with Weights & Biases
experiment tracking.

Targets NVIDIA H100 (80 GB) or any CUDA device; falls back to CPU
transparently when LuxCUDA is unavailable or no GPU is detected.
WandB tracking is optional — omit --wandb-project to disable it.

One-time setup (run once on each machine, inside the project environment):
    julia --project=. -e 'using Pkg; Pkg.add("LuxCUDA")'     # GPU support
    julia --project=. -e 'using Pkg; Pkg.add("Wandb")'        # experiment tracking
    wandb login                                                 # authenticate

Usage:
    # H100 with a local PDB mirror (flat directory) + WandB tracking
    julia --project=. scripts/train_cuda.jl /path/to/pdbs \\
        --wandb-project diffusebio --wandb-name h100-run-01

    # Nested PDB mirror layout (e.g. standard rsync mirror: pdb/ab/1abc.ent)
    julia --project=. scripts/train_cuda.jl /path/to/pdbs --recursive

    # Override model / training knobs
    julia --project=. scripts/train_cuda.jl /path/to/pdbs \\
        --epochs 200 --batch-size 32 --lr 3e-4 --d-single 256 \\
        --checkpoint-dir ./ckpts --checkpoint-every 10

    # Resume from a checkpoint (WandB run id can be passed to resume the run)
    julia --project=. scripts/train_cuda.jl /path/to/pdbs \\
        --resume ./ckpts/checkpoint_0050.jls \\
        --wandb-project diffusebio --wandb-resume-id <run-id>

    # Cache preprocessed (parsed+tokenized+featurized) structures on first run,
    # skip re-preprocessing entirely on every subsequent launch against the
    # same data_dir/n-targets/max-atoms/recursive combination
    julia --project=. scripts/train_cuda.jl /path/to/pdbs --cache-file ./cache/examples.jls

    # Build a reusable library once (see preprocess_dataset.jl), then train
    # against whatever subset of it a given run wants — no re-parsing ever,
    # and --n-targets/--max-atoms can differ freely between runs since the
    # library isn't tied to either
    julia --project=. scripts/preprocess_dataset.jl /path/to/pdbs \\
        --library-dir ./library --limit 1000
    julia --project=. scripts/train_cuda.jl --library-dir ./library --n-targets 200

    # CPU smoke-test, no WandB
    julia --project=. scripts/train_cuda.jl \\
        --epochs 5 --d-single 32 --d-pair 16 --n-heads 4 \\
        --n-pairformer-layers 2 --n-dit-layers 2

    # Community-standard train/val/test split (defaults: 10% val, 10% test,
    # 80% train). Val loss is tracked every epoch alongside train loss; test
    # loss is computed once, after all training is done.
    julia --project=. scripts/train_cuda.jl /path/to/pdbs \\
        --val-fraction 0.1 --test-fraction 0.1

Checkpoints are Serialization.serialize'd NamedTuples (epoch, ps, st) written
to --checkpoint-dir and loadable again with --resume.

--cache-file similarly serializes the preprocessed examples (post parse/
tokenize/featurize, pre-batching) after the first run, keyed against
data_dir/n_targets/max_atoms/recursive — a mismatch on any of those triggers
a warning and a full re-preprocess rather than silently reusing stale data.
Preprocessing is CPU-bound and cheap per structure (parsing + tokenizing),
but at real-dataset scale (hundreds of thousands of structures) redoing it on
every launch adds up; this trades that recurring cost for a one-time cost.
Note --seed is deliberately *not* part of the cache key: it only affects
which subset of files get selected when candidates outnumber --n-targets,
and re-randomizing that on every launch would defeat the point of caching.

--library-dir is the other, more flexible way to skip re-preprocessing: point
it at a directory built by `preprocess_dataset.jl` (one small serialized file
per structure, independent of any single run's --n-targets/--max-atoms/
--recursive) and every run just reads whatever subset it wants straight off
disk. Prefer --cache-file for a single training configuration you'll rerun
as-is (simplest — one flag, no separate preprocessing step); prefer
--library-dir once you're exploring multiple --n-targets/--max-atoms
combinations against the same larger pool of structures, since --cache-file
would force a full re-preprocess on every combination change. When both a
source directory and --library-dir are given, --library-dir wins.

--val-fraction/--test-fraction carve the *loaded* pool (i.e. out of
--n-targets, before any split — a run with --n-targets 1000 --val-fraction
0.1 --test-fraction 0.1 trains on 800, not 1000) into the standard
three-way split:
  - train (the rest, after val/test are carved off): the only set that ever
    produces a gradient / touches `ps`.
  - val (--val-fraction, default 0.1): evaluated every epoch with a
    forward-only pass (`eval_on_batch` — no gradient, never influences
    training) and logged as `val/loss` alongside `train/loss`. This is what
    tells you the model is generalizing rather than memorizing — train loss
    falling while val loss stalls or rises is the classic overfitting
    signature. Since it's checked every epoch, it's implicitly used to
    *watch* training even though nothing here currently acts on it
    automatically (e.g. no early stopping / best-checkpoint selection yet).
  - test (--test-fraction, default 0.1): held out completely and evaluated
    exactly once, after all training/checkpointing is finished, logged as a
    single `test/loss` value. Never seen — not even for monitoring — until
    that final check, which is what makes it a trustworthy final number
    rather than something that was implicitly tuned against.
Either fraction can be set to 0 to disable that split (falls back to the
next-most-held-out set for the end-of-run structural validity check: test,
then val, then train).
"""

using Printf
import Serialization

# ── Optional CUDA — graceful CPU fallback ────────────────────────────────────
# LuxCUDA is not a hard project dependency (avoids forcing CUDA toolkit
# downloads on CPU-only dev machines). Loading it registers the CUDA backend
# with Lux so Lux.gpu_device() returns a real CUDA device.
if !isnothing(Base.find_package("LuxCUDA"))
    @eval using LuxCUDA
end
const HAS_GPU = @isdefined(CUDA) && CUDA.functional()

# ── Optional WandB — graceful no-op when not installed / not logged in ────────
# Wandb.jl wraps the Python wandb library via CondaPkg.jl. Install with:
#   julia --project=. -e 'using Pkg; Pkg.add("Wandb")'
if !isnothing(Base.find_package("Wandb"))
    @eval using Wandb
end
const HAS_WANDB = @isdefined(WandbLogger)

using DiffuseBioMol
using Lux
using Random, Zygote, Optimisers

# ── CLI argument parsing ──────────────────────────────────────────────────────

function parse_args(args)
    # Defaults are tuned for H100 80 GB. Scale down for smaller GPUs / CPU.
    kw = Dict{Symbol,Any}(
        :data_dir              => nothing,
        :n_targets             => 500,
        :max_atoms             => 2000,
        :val_fraction          => 0.1,      # --val-fraction: held out for per-epoch val/loss (0 disables)
        :test_fraction         => 0.1,      # --test-fraction: held out for a one-time final test/loss (0 disables)
        :n_epochs              => 100,
        :batch_size            => 16,
        :lr                    => 1.0f-4,
        :seed                  => 42,
        :checkpoint_dir        => nothing,
        :checkpoint_every      => 10,
        :resume                => nothing,
        :recursive             => false,    # --recursive: walk subdirs for PDB mirrors
        :cache_file            => nothing,  # --cache-file: cache preprocessed examples across runs
        :library_dir           => nothing,  # --library-dir: read from a preprocess_dataset.jl library instead
        # Model config — H100-scale defaults
        :d_single              => 128,
        :d_pair                => 64,
        :n_heads               => 8,
        :n_pairformer_layers   => 4,
        :n_dit_layers          => 4,
        :d_time                => 64,
        # WandB
        :wandb_project         => nothing,  # --wandb-project <name>
        :wandb_name            => nothing,  # --wandb-name <run-name>
        :wandb_entity          => nothing,  # --wandb-entity <team>
        :wandb_resume_id       => nothing,  # --wandb-resume-id <run-id>  (for resumed runs)
    )
    i = 1
    while i <= length(args)
        a = args[i]
        nxt = i < length(args) ? args[i+1] : ""
        if     a == "--epochs"               ; kw[:n_epochs]            = parse(Int,     nxt); i += 2
        elseif a == "--batch-size"           ; kw[:batch_size]          = parse(Int,     nxt); i += 2
        elseif a == "--lr"                   ; kw[:lr]                  = parse(Float32, nxt); i += 2
        elseif a == "--n-targets"            ; kw[:n_targets]           = parse(Int,     nxt); i += 2
        elseif a == "--max-atoms"            ; kw[:max_atoms]           = parse(Int,     nxt); i += 2
        elseif a == "--val-fraction"         ; kw[:val_fraction]        = parse(Float64, nxt); i += 2
        elseif a == "--test-fraction"        ; kw[:test_fraction]       = parse(Float64, nxt); i += 2
        elseif a == "--seed"                 ; kw[:seed]                = parse(Int,     nxt); i += 2
        elseif a == "--checkpoint-dir"       ; kw[:checkpoint_dir]      = nxt;                 i += 2
        elseif a == "--checkpoint-every"     ; kw[:checkpoint_every]    = parse(Int,     nxt); i += 2
        elseif a == "--resume"               ; kw[:resume]              = nxt;                 i += 2
        elseif a == "--recursive"            ; kw[:recursive]           = true;                i += 1
        elseif a == "--cache-file"           ; kw[:cache_file]          = nxt;                 i += 2
        elseif a == "--library-dir"          ; kw[:library_dir]         = nxt;                 i += 2
        elseif a == "--d-single"             ; kw[:d_single]            = parse(Int,     nxt); i += 2
        elseif a == "--d-pair"               ; kw[:d_pair]              = parse(Int,     nxt); i += 2
        elseif a == "--n-heads"              ; kw[:n_heads]             = parse(Int,     nxt); i += 2
        elseif a == "--n-pairformer-layers"  ; kw[:n_pairformer_layers] = parse(Int,     nxt); i += 2
        elseif a == "--n-dit-layers"         ; kw[:n_dit_layers]        = parse(Int,     nxt); i += 2
        elseif a == "--wandb-project"        ; kw[:wandb_project]       = nxt;                 i += 2
        elseif a == "--wandb-name"           ; kw[:wandb_name]          = nxt;                 i += 2
        elseif a == "--wandb-entity"         ; kw[:wandb_entity]        = nxt;                 i += 2
        elseif a == "--wandb-resume-id"      ; kw[:wandb_resume_id]     = nxt;                 i += 2
        elseif startswith(a, "--")
            @warn "Unknown flag: $a (ignored)"
            i += 2
        else
            kw[:data_dir] = a; i += 1
        end
    end
    kw
end

# ── Data loading ──────────────────────────────────────────────────────────────

include(joinpath(@__DIR__, "dataset_utils.jl"))

"""
    load_library_example(path) -> NamedTuple

Deserializes one already-preprocessed structure written by
`preprocess_dataset.jl` (same shape as `tokenize_example`'s return value) —
the whole point of `--library-dir` is that this is just a disk read, no
parsing/tokenizing/featurizing.
"""
load_library_example(path) = Serialization.deserialize(path)

"""
    cache_key(; data_dir, library_dir, n_targets, max_atoms, recursive) -> NamedTuple

The subset of `load_targets`' arguments that determine *which structures* end
up in `examples` (deliberately excludes `rng`/`--seed` — see this file's
module docstring for why). Stored alongside the cached examples so a stale
cache built with different arguments is detected and rejected rather than
silently reused.
"""
cache_key(; data_dir, library_dir, n_targets, max_atoms, recursive) =
    (data_dir=data_dir, library_dir=library_dir, n_targets=n_targets, max_atoms=max_atoms, recursive=recursive)

function load_targets(; data_dir, library_dir=nothing, n_targets, max_atoms, recursive, rng, cache_file=nothing)
    key = cache_key(; data_dir, library_dir, n_targets, max_atoms, recursive)

    if cache_file !== nothing && isfile(cache_file)
        cached = Serialization.deserialize(cache_file)
        if cached.key == key
            println("Cache   : loaded $(length(cached.examples)) preprocessed structures from $cache_file (skipped re-parsing)")
            return cached.examples
        end
        @warn "Cache at $cache_file was built with different arguments than this run; re-preprocessing and overwriting it.\n" *
              "  cached : $(cached.key)\n  current: $key"
    end

    if library_dir !== nothing
        # Already preprocessed by preprocess_dataset.jl — one small file per
        # structure, decoupled from any run's n_targets/max_atoms, so this is
        # a plain disk read (deserialize), not a re-parse.
        candidates = shuffle(rng, filter(f -> endswith(f, ".jls"), readdir(library_dir; join=true)))
        isempty(candidates) && throw(ArgumentError(
            "no preprocessed structures (*.jls) found in $library_dir — " *
            "run scripts/preprocess_dataset.jl against a source directory first"))
        loader = load_library_example
    elseif data_dir === nothing
        candidates = CANDIDATE_TARGETS
        loader = load_rcsb_example
    else
        candidates = shuffle(rng, list_structure_files(data_dir; recursive))
        isempty(candidates) && throw(ArgumentError(
            "no PDB/mmCIF files found under $data_dir (try --recursive for nested layouts)"))
        loader = load_local_example
    end

    examples = []
    for c in candidates
        length(examples) >= n_targets && break
        try
            ex = loader(c)
            max_atoms !== nothing && ex.n_atoms > max_atoms && continue
            push!(examples, ex)
        catch e
            label = c isa Tuple ? c[1] : basename(string(c))
            println("  skip $label: $(sprint(showerror, e))")
        end
    end

    if cache_file !== nothing
        cache_dir = dirname(cache_file)
        isempty(cache_dir) || mkpath(cache_dir)
        Serialization.serialize(cache_file, (key=key, examples=examples))
        println("Cache   : wrote $(length(examples)) preprocessed structures to $cache_file")
    end

    examples
end

# ── Training step (GPU-aware) ─────────────────────────────────────────────────

function train_on_batch(model, ps, st, opt_state, batch, rng, dev)
    # Build padded batch on CPU — cheap; no GPU needed for padding logic
    bf     = batch_features([ex.feat for ex in batch])
    rp     = batch_relpos([ex.relpos for ex in batch])
    cf     = batch_cond_features([ex.cond_features for ex in batch])
    pb     = attention_pad_bias(bf.pad_mask)
    x1     = batch_coords([Float32.(ex.x1) for ex in batch])
    # prepare_training_example uses RNG — must stay outside the AD path
    ex_cpu = prepare_training_example(bf, x1, cf, rng)

    # Move batch to compute device
    bf_d = to_device(bf, dev)
    rp_d = dev(rp)
    pb_d = dev(pb)
    ex_d = to_device(ex_cpu, dev)

    loss, back = Zygote.pullback(
        p -> cfm_loss(model, p, st, bf_d, rp_d, pb_d, ex_d)[1],
        ps,
    )
    grad = back(one(Float32))[1]
    opt_state, ps = Optimisers.update(opt_state, ps, grad)
    # Float32(loss) pulls a GPU scalar to host; no-op on CPU
    (ps, opt_state, Float32(loss))
end

"""
    eval_on_batch(model, ps, st, batch, rng, dev) -> Float32

Same forward pass as `train_on_batch`, minus the `Zygote.pullback`/gradient/
optimizer-update — a read-only loss estimate that never touches `ps`. Used
for both the per-epoch val loss and the one-time final test loss, so
evaluating either can never leak into the trained weights.
"""
function eval_on_batch(model, ps, st, batch, rng, dev)
    bf     = batch_features([ex.feat for ex in batch])
    rp     = batch_relpos([ex.relpos for ex in batch])
    cf     = batch_cond_features([ex.cond_features for ex in batch])
    pb     = attention_pad_bias(bf.pad_mask)
    x1     = batch_coords([Float32.(ex.x1) for ex in batch])
    ex_cpu = prepare_training_example(bf, x1, cf, rng)

    bf_d = to_device(bf, dev)
    rp_d = dev(rp)
    pb_d = dev(pb)
    ex_d = to_device(ex_cpu, dev)

    loss, _ = cfm_loss(model, ps, st, bf_d, rp_d, pb_d, ex_d)
    Float32(loss)
end

# ── Checkpointing ─────────────────────────────────────────────────────────────

function save_checkpoint(dir, epoch, ps, st, cpu_dev)
    mkpath(dir)
    path = joinpath(dir, @sprintf("checkpoint_%04d.jls", epoch))
    Serialization.serialize(path, (epoch=epoch, ps=cpu_dev(ps), st=cpu_dev(st)))
    println("  [ckpt] $path")
end

function load_checkpoint(path, dev)
    println("Resuming: $path")
    ckpt = Serialization.deserialize(path)
    println("  resumed at epoch $(ckpt.epoch)")
    (dev(ckpt.ps), dev(ckpt.st), ckpt.epoch)
end

# ── WandB helpers ─────────────────────────────────────────────────────────────

"""
    init_wandb(kw, cfg, n_params, n_train, n_val, n_test, device_str) -> WandbLogger or nothing

Initialises a WandB run if --wandb-project was given and Wandb.jl is installed;
returns nothing otherwise so all call sites can use `isnothing(lg)` to skip.
"""
function init_wandb(kw, cfg, n_params, n_train, n_val, n_test, device_str)
    kw[:wandb_project] === nothing && return nothing
    if !HAS_WANDB
        @warn "--wandb-project given but Wandb.jl is not installed; tracking disabled.\n" *
              "Install with: julia --project=. -e 'using Pkg; Pkg.add(\"Wandb\")'"
        return nothing
    end
    config = Dict(
        "device"            => device_str,
        "n_structures"      => n_train + n_val + n_test,
        "n_train"           => n_train,
        "n_val"             => n_val,
        "n_test"            => n_test,
        "val_fraction"      => kw[:val_fraction],
        "test_fraction"     => kw[:test_fraction],
        "max_atoms"         => kw[:max_atoms],
        "batch_size"        => kw[:batch_size],
        "lr"                => kw[:lr],
        "n_epochs"          => kw[:n_epochs],
        "seed"              => kw[:seed],
        "d_single"          => cfg.d_single,
        "d_pair"            => cfg.d_pair,
        "n_heads"           => cfg.n_heads,
        "n_pairformer_layers" => cfg.n_pairformer_layers,
        "n_dit_layers"      => cfg.n_dit_layers,
        "d_time"            => cfg.d_time,
        "n_params"          => n_params,
        "resumed_from"      => kw[:resume],
    )
    # wandb resume: pass the existing run id to continue the same run's charts
    extra = kw[:wandb_resume_id] !== nothing ?
        Dict(:id => kw[:wandb_resume_id], :resume => "must") : Dict()
    lg = WandbLogger(;
        project = kw[:wandb_project],
        name    = kw[:wandb_name],
        entity  = kw[:wandb_entity],
        config,
        extra...,
    )
    println("WandB   : $(kw[:wandb_project]) / $(something(kw[:wandb_name], "(auto)"))")
    lg
end

"""
    wlog(lg, d; step) — log `d` to WandB if `lg` is not nothing (no-op otherwise).
"""
wlog(::Nothing, ::Any; kwargs...) = nothing
wlog(lg, d; step=nothing) = Wandb.log(lg, d; step)

# ── Recursive parameter count (works on Lux's nested NamedTuple ps) ──────────

count_params(ps::AbstractArray) = length(ps)
count_params(ps::NamedTuple)    = sum(count_params(v) for v in values(ps); init=0)
count_params(ps::Tuple)         = sum(count_params(v) for v in ps;         init=0)
count_params(::Any)             = 0

# ── Main ──────────────────────────────────────────────────────────────────────

function main(args=ARGS)
    kw = parse_args(args)

    # ── device
    dev     = HAS_GPU ? Lux.gpu_device() : Lux.cpu_device()
    cpu_dev = Lux.cpu_device()
    if HAS_GPU
        device_str = @sprintf("GPU — %s (%.1f GiB)", CUDA.name(CUDA.device()),
                              CUDA.total_memory() / 2^30)
        @printf("Device  : %s\n", device_str)
    else
        device_str = "CPU"
        println("Device  : CPU  (install LuxCUDA for GPU: " *
                "`julia --project=. -e 'using Pkg; Pkg.add(\"LuxCUDA\")'`)")
    end

    # ── data
    rng = Random.Xoshiro(kw[:seed])
    if kw[:library_dir] !== nothing
        kw[:data_dir] !== nothing && println("Note    : both a source dir and --library-dir were given; --library-dir takes precedence.")
        src = "library: $(kw[:library_dir])"
    else
        src = kw[:data_dir] === nothing ? "RCSB (live fetch)" : kw[:data_dir]
    end
    @printf("Data    : %s  (target=%d, max_atoms=%s)\n",
        src, kw[:n_targets], kw[:max_atoms] === nothing ? "none" : string(kw[:max_atoms]))
    t_load = @elapsed examples = load_targets(;
        data_dir    = kw[:data_dir],
        library_dir = kw[:library_dir],
        n_targets   = kw[:n_targets],
        max_atoms   = kw[:max_atoms],
        recursive   = kw[:recursive],
        cache_file  = kw[:cache_file],
        rng)
    isempty(examples) && error("no structures loaded — check data_dir or network")
    natoms = [ex.n_atoms for ex in examples]
    @printf("Loaded  : %d structures  %.1fs  (atoms min=%d mean=%.0f max=%d)\n",
        length(examples), t_load,
        minimum(natoms), sum(natoms) / length(natoms), maximum(natoms))

    # ── train/val/test split — the standard three-way split (see module
    # docstring): val is checked every epoch to watch generalization during
    # training, test is held out completely and checked exactly once at the
    # end. Neither ever produces a gradient (see eval_on_batch).
    n_val  = kw[:val_fraction]  > 0 ? max(1, round(Int, kw[:val_fraction]  * length(examples))) : 0
    n_test = kw[:test_fraction] > 0 ? max(1, round(Int, kw[:test_fraction] * length(examples))) : 0
    n_val + n_test < length(examples) || error(
        "--val-fraction ($(kw[:val_fraction])) + --test-fraction ($(kw[:test_fraction])) " *
        "leaves no structures for training (n=$(length(examples)), val=$n_val, test=$n_test)")
    split_order    = shuffle(rng, 1:length(examples))
    val_examples   = examples[split_order[1:n_val]]
    test_examples  = examples[split_order[n_val+1:n_val+n_test]]
    train_examples = examples[split_order[n_val+n_test+1:end]]
    @printf("Split   : %d train / %d val / %d test  (val_fraction=%g, test_fraction=%g)\n",
        length(train_examples), length(val_examples), length(test_examples),
        kw[:val_fraction], kw[:test_fraction])

    # ── model
    cfg   = ModelConfig(
        d_single           = kw[:d_single],
        d_pair             = kw[:d_pair],
        n_heads            = kw[:n_heads],
        n_pairformer_layers = kw[:n_pairformer_layers],
        n_dit_layers       = kw[:n_dit_layers],
        d_time             = kw[:d_time],
    )
    model = build_model(cfg)

    # ── params / state (resume or fresh init)
    start_epoch = 1
    if kw[:resume] !== nothing
        ps, st, start_epoch = load_checkpoint(kw[:resume], dev)
        start_epoch += 1
    else
        ps_cpu, st_cpu = Lux.setup(rng, model)
        ps, st = dev(ps_cpu), dev(st_cpu)
    end

    n_params = count_params(ps)
    @printf("Model   : d_single=%d  d_pair=%d  n_heads=%d  pairformer=%d  dit=%d  ~%.2fM params\n",
        cfg.d_single, cfg.d_pair, cfg.n_heads,
        cfg.n_pairformer_layers, cfg.n_dit_layers, n_params / 1e6)

    # Optimizer set up AFTER moving ps to device — Optimisers mirrors the
    # parameter tree's device, so opt_state must live on the same device as ps.
    opt_state = Optimisers.setup(Optimisers.Adam(kw[:lr]), ps)

    # ── WandB init (after model is built so we can log n_params in config)
    wandb_lg = init_wandb(kw, cfg, n_params, length(train_examples), length(val_examples), length(test_examples), device_str)

    # ── training loop
    n_epochs   = kw[:n_epochs]
    batch_size = kw[:batch_size]
    n_batches  = cld(length(train_examples), batch_size)
    @printf("\nTraining: epochs %d–%d  batch_size=%d  (~%d train batches/epoch, %d val batches/epoch)  lr=%g\n\n",
        start_epoch, n_epochs, batch_size, n_batches, cld(length(val_examples), batch_size), kw[:lr])

    # Val batches are fixed once — no reshuffling needed, since eval_on_batch
    # never updates ps and batch composition doesn't affect a forward-only pass.
    val_batches = [val_examples[i:min(i + batch_size - 1, end)]
                    for i in 1:batch_size:length(val_examples)]

    ep_pad = ndigits(n_epochs)
    for epoch in start_epoch:n_epochs
        order   = shuffle(rng, 1:length(train_examples))
        batches = [train_examples[order[i:min(i + batch_size - 1, end)]]
                   for i in 1:batch_size:length(order)]

        epoch_losses = Float32[]
        t_epoch = @elapsed for batch in batches
            ps, opt_state, loss = train_on_batch(model, ps, st, opt_state, batch, rng, dev)
            push!(epoch_losses, loss)
        end
        mean_loss = sum(epoch_losses) / length(epoch_losses)

        # Val loss: forward-only, no gradient update — checked every epoch to
        # watch generalization during training (see module docstring).
        val_loss = if isempty(val_batches)
            nothing
        else
            val_losses = [eval_on_batch(model, ps, st, vb, rng, dev) for vb in val_batches]
            sum(val_losses) / length(val_losses)
        end

        @printf("epoch %*d/%d  %5.1fs  train_loss=%.4f%s  per-batch=%s\n",
            ep_pad, epoch, n_epochs, t_epoch, mean_loss,
            val_loss === nothing ? "" : @sprintf("  val_loss=%.4f", val_loss),
            string(round.(epoch_losses; digits=3)))

        # Log epoch metrics to WandB
        epoch_log = Dict(
            "train/loss"         => mean_loss,
            "train/loss_min"     => minimum(epoch_losses),
            "train/loss_max"     => maximum(epoch_losses),
            "train/epoch_time_s" => t_epoch,
            "train/lr"          => kw[:lr],
        )
        val_loss !== nothing && (epoch_log["val/loss"] = val_loss)
        wlog(wandb_lg, epoch_log; step=epoch)

        if kw[:checkpoint_dir] !== nothing && epoch % kw[:checkpoint_every] == 0
            save_checkpoint(kw[:checkpoint_dir], epoch, ps, st, cpu_dev)
        end
    end

    # always write a final checkpoint if a dir was given
    if kw[:checkpoint_dir] !== nothing
        save_checkpoint(kw[:checkpoint_dir], n_epochs, ps, st, cpu_dev)
    end

    # ── final test loss — computed exactly once, after all training and
    # checkpointing decisions are already made, so it's never implicitly
    # tuned against (unlike val, which was watched every epoch above).
    if !isempty(test_examples)
        test_batches = [test_examples[i:min(i + batch_size - 1, end)]
                         for i in 1:batch_size:length(test_examples)]
        test_losses  = [eval_on_batch(model, ps, st, tb, rng, dev) for tb in test_batches]
        final_test_loss = sum(test_losses) / length(test_losses)
        @printf("\nFinal test loss (%d held-out structures, never seen during training): %.4f\n",
            length(test_examples), final_test_loss)
        wlog(wandb_lg, Dict("test/loss" => final_test_loss))
    end

    # ── validity check (single-structure CPU API — keeps sample_flow unchanged)
    # Prefer the held-out test set (same rationale as the final test loss
    # above); fall back to val, then train_examples, if a set was disabled.
    check_pool, check_label = if !isempty(test_examples)
        (test_examples, "held-out test")
    elseif !isempty(val_examples)
        (val_examples, "held-out val")
    else
        (train_examples, "training")
    end
    n_check = min(5, length(check_pool))
    ps_cpu  = cpu_dev(ps)
    st_cpu  = cpu_dev(st)
    println("\nValidity check on $n_check $check_label structures (single-structure, CPU):")
    for ex in check_pool[1:n_check]
        x_hat, _ = sample_flow(model, ps_cpu, st_cpu, ex.feat, ex.relpos,
                                ex.cond_features, rng; n_steps=20)
        rmsd    = aligned_rmsd(x_hat, Float32.(ex.x1))
        bonds   = backbone_bonds(ex.tokens)
        brmsd   = bond_length_rmsd(x_hat, bonds)
        elems   = [t.element for t in ex.tokens]
        clashes = clash_count(x_hat, elems, ex.feat.chain_idx, ex.feat.res_index)
        @printf("  %-12s %4d atoms  RMSD=%6.2f Å  clashes=%3d  bond-RMSD=%.3f Å\n",
            ex.pdb_id, ex.n_atoms, rmsd, clashes, brmsd)

        # Log per-structure validity metrics to WandB
        wlog(wandb_lg, Dict(
            "test/rmsd_$(ex.pdb_id)"      => rmsd,
            "test/clashes_$(ex.pdb_id)"   => clashes,
            "test/bond_rmsd_$(ex.pdb_id)" => brmsd,
        ))
    end

    # Log summary validity stats across all checked structures
    if n_check > 0
        all_rmsds = Float64[]
        all_clashes = Int[]
        all_brmsds = Float64[]
        for ex in check_pool[1:n_check]
            x_hat, _ = sample_flow(model, ps_cpu, st_cpu, ex.feat, ex.relpos,
                                    ex.cond_features, rng; n_steps=20)
            push!(all_rmsds,   aligned_rmsd(x_hat, Float32.(ex.x1)))
            push!(all_clashes, clash_count(x_hat, [t.element for t in ex.tokens],
                                           ex.feat.chain_idx, ex.feat.res_index))
            push!(all_brmsds,  bond_length_rmsd(x_hat, backbone_bonds(ex.tokens)))
        end
        wlog(wandb_lg, Dict(
            "test/mean_rmsd"      => sum(all_rmsds)   / n_check,
            "test/mean_clashes"   => sum(all_clashes) / n_check,
            "test/mean_bond_rmsd" => sum(all_brmsds)  / n_check,
        ))
    end

    # Close the WandB run — uploads remaining data and marks the run finished
    wandb_lg !== nothing && close(wandb_lg)

    (model=model, ps=ps_cpu, st=st_cpu, examples=examples,
        train_examples=train_examples, val_examples=val_examples, test_examples=test_examples)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
