"""
GPU-accelerated training script for DiffuseBioMol.jl.

Targets NVIDIA H100 (80 GB) or any CUDA device; falls back to CPU
transparently when LuxCUDA is unavailable or no GPU is detected.

One-time setup (run once on each machine, inside the project environment):
    julia --project=. -e 'using Pkg; Pkg.add("LuxCUDA")'

Usage:
    # H100 with a local PDB mirror (flat directory)
    julia --project=. scripts/train_cuda.jl /path/to/pdbs

    # Nested mirror layout (e.g. standard PDB rsync: pdb/ab/1abc.ent)
    julia --project=. scripts/train_cuda.jl /path/to/pdbs --recursive

    # Override model / training knobs
    julia --project=. scripts/train_cuda.jl /path/to/pdbs \\
        --epochs 200 --batch-size 32 --lr 3e-4 --d-single 256 \\
        --checkpoint-dir ./ckpts --checkpoint-every 10

    # Resume from a checkpoint
    julia --project=. scripts/train_cuda.jl /path/to/pdbs \\
        --resume ./ckpts/checkpoint_0050.jls

    # CPU smoke-test (RCSB live fetch, tiny model)
    julia --project=. scripts/train_cuda.jl \\
        --epochs 5 --d-single 32 --d-pair 16 --n-heads 4 \\
        --n-pairformer-layers 2 --n-dit-layers 2

Checkpoints are Serialization.serialize'd NamedTuples (epoch, ps, st) written
to --checkpoint-dir and loadable again with --resume.
"""

using Printf
import Serialization

# ── Optional CUDA — graceful CPU fallback ────────────────────────────────────
# LuxCUDA is not a hard project dependency (avoids forcing CUDA toolkit
# downloads on CPU-only dev machines). If it is installed in the active
# environment, loading it registers the CUDA backend with Lux so that
# Lux.gpu_device() returns a real CUDA device instead of falling back to CPU.
if !isnothing(Base.find_package("LuxCUDA"))
    @eval using LuxCUDA
end
const HAS_GPU = @isdefined(CUDA) && CUDA.functional()

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
        :n_epochs              => 100,
        :batch_size            => 16,
        :lr                    => 1.0f-4,
        :seed                  => 42,
        :checkpoint_dir        => nothing,
        :checkpoint_every      => 10,
        :resume                => nothing,
        :recursive             => false,   # pass --recursive for nested PDB mirrors
        # Model config — H100-scale defaults
        :d_single              => 128,
        :d_pair                => 64,
        :n_heads               => 8,
        :n_pairformer_layers   => 4,
        :n_dit_layers          => 4,
        :d_time                => 64,
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
        elseif a == "--seed"                 ; kw[:seed]                = parse(Int,     nxt); i += 2
        elseif a == "--checkpoint-dir"       ; kw[:checkpoint_dir]      = nxt;                 i += 2
        elseif a == "--checkpoint-every"     ; kw[:checkpoint_every]    = parse(Int,     nxt); i += 2
        elseif a == "--resume"               ; kw[:resume]              = nxt;                 i += 2
        elseif a == "--d-single"             ; kw[:d_single]            = parse(Int,     nxt); i += 2
        elseif a == "--d-pair"               ; kw[:d_pair]              = parse(Int,     nxt); i += 2
        elseif a == "--n-heads"             ; kw[:n_heads]              = parse(Int,     nxt); i += 2
        elseif a == "--n-pairformer-layers"  ; kw[:n_pairformer_layers] = parse(Int,     nxt); i += 2
        elseif a == "--n-dit-layers"         ; kw[:n_dit_layers]        = parse(Int,     nxt); i += 2
        elseif a == "--recursive"            ; kw[:recursive]           = true;                i += 1
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

const CANDIDATE_TARGETS = [
    ("1CRN","A"),("1UBQ","A"),("5PTI","A"),("1L2Y","A"),("1VII","A"),("2GB1","A"),
    ("1ENH","A"),("1PGB","A"),("1SHG","A"),("2CI2","A"),("1IGD","A"),("1TEN","A"),
    ("2TRX","A"),("1AHO","A"),("1BTA","A"),("1CSP","A"),("1FNA","A"),("1MJC","A"),
    ("1PIN","A"),("1HHP","A"),("2PTL","A"),("1UBI","A"),("1COA","A"),("1FXD","A"),
]

function tokenize_example(label, residues)
    tokens = tokenize_structure(residues)
    isempty(tokens) && error("no tokens after parsing/filtering")
    feat = featurize(tokens)
    (pdb_id       = label,
     tokens       = tokens,
     feat         = feat,
     relpos       = relpos_buckets(feat),
     x1           = target_coordinates(tokens),
     cond_features = constraint_features(no_constraints(length(tokens))),
     n_atoms      = length(tokens))
end

function load_rcsb_example((pdb_id, chain_id))
    tokenize_example(pdb_id, restrict_to_chain(fetch_pdb(pdb_id), chain_id))
end

function load_local_example(path)
    res = parse_structure(path)
    isempty(res) && error("no residues parsed")
    ch = largest_chain(res)
    tokenize_example(splitext(basename(path))[1], restrict_to_chain(res, ch))
end

function load_targets(; data_dir, n_targets, max_atoms, rng)
    if data_dir === nothing
        candidates = CANDIDATE_TARGETS
        loader = load_rcsb_example
    else
        candidates = shuffle(rng, list_structure_files(data_dir; recursive=kw[:recursive]))
        isempty(candidates) && throw(ArgumentError("no PDB/mmCIF files found under $data_dir (try --recursive for nested directory layouts)"))
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
            label = data_dir === nothing ? c[1] : basename(string(c))
            println("  skip $label: $(sprint(showerror, e))")
        end
    end
    examples
end

# ── Training step (GPU-aware) ─────────────────────────────────────────────────

function train_on_batch(model, ps, st, opt_state, batch, rng, dev)
    # Build padded batch on CPU — cheap; GPU not needed for the padding logic
    bf = batch_features([ex.feat for ex in batch])
    rp = batch_relpos([ex.relpos for ex in batch])
    cf = batch_cond_features([ex.cond_features for ex in batch])
    pb = attention_pad_bias(bf.pad_mask)
    x1 = batch_coords([Float32.(ex.x1) for ex in batch])
    # prepare_training_example runs on CPU (uses RNG; must stay outside AD path)
    ex_cpu = prepare_training_example(bf, x1, cf, rng)

    # Move batch to compute device
    bf_d  = to_device(bf, dev)
    rp_d  = dev(rp)
    pb_d  = dev(pb)
    ex_d  = to_device(ex_cpu, dev)

    loss, back = Zygote.pullback(
        p -> cfm_loss(model, p, st, bf_d, rp_d, pb_d, ex_d)[1],
        ps,
    )
    grad = back(one(Float32))[1]
    opt_state, ps = Optimisers.update(opt_state, ps, grad)
    # Float32(loss) pulls a GPU scalar to host; is a no-op on CPU
    (ps, opt_state, Float32(loss))
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

# ── Recursive parameter count (works on Lux's nested NamedTuple ps) ───────────

count_params(ps::AbstractArray) = length(ps)
count_params(ps::NamedTuple)    = sum(count_params(v) for v in values(ps); init=0)
count_params(ps::Tuple)         = sum(count_params(v) for v in ps;         init=0)
count_params(::Any)             = 0

# ── Main ──────────────────────────────────────────────────────────────────────

function main(args=ARGS)
    kw = parse_args(args)

    # ── device setup
    dev     = HAS_GPU ? Lux.gpu_device() : Lux.cpu_device()
    cpu_dev = Lux.cpu_device()
    if HAS_GPU
        @printf("Device  : GPU — %s  (%.1f GiB VRAM)\n",
            CUDA.name(CUDA.device()), CUDA.total_memory() / 2^30)
    else
        println("Device  : CPU  (install LuxCUDA for GPU: `julia --project=. -e 'using Pkg; Pkg.add(\"LuxCUDA\")'`)")
    end

    # ── data
    rng = Random.Xoshiro(kw[:seed])
    src = kw[:data_dir] === nothing ? "RCSB (live fetch)" : kw[:data_dir]
    @printf("Data    : %s  (target=%d, max_atoms=%s)\n",
        src, kw[:n_targets], kw[:max_atoms] === nothing ? "none" : string(kw[:max_atoms]))
    t_load = @elapsed examples = load_targets(;
        data_dir=kw[:data_dir], n_targets=kw[:n_targets],
        max_atoms=kw[:max_atoms], rng)
    isempty(examples) && error("no structures loaded — check data_dir or network")
    natoms = [ex.n_atoms for ex in examples]
    @printf("Loaded  : %d structures  %.1fs  (atoms min=%d mean=%.0f max=%d)\n",
        length(examples), t_load, minimum(natoms),
        sum(natoms) / length(natoms), maximum(natoms))

    # ── model
    cfg = ModelConfig(
        d_single=kw[:d_single], d_pair=kw[:d_pair], n_heads=kw[:n_heads],
        n_pairformer_layers=kw[:n_pairformer_layers], n_dit_layers=kw[:n_dit_layers],
        d_time=kw[:d_time],
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
        cfg.d_single, cfg.d_pair, cfg.n_heads, cfg.n_pairformer_layers, cfg.n_dit_layers,
        n_params / 1e6)

    # Optimizer set up AFTER moving ps to device — Optimisers mirrors the
    # parameter tree's device, so opt_state must live on the same device as ps.
    opt_state = Optimisers.setup(Optimisers.Adam(kw[:lr]), ps)

    # ── training loop
    n_epochs   = kw[:n_epochs]
    batch_size = kw[:batch_size]
    n_batches  = cld(length(examples), batch_size)
    @printf("\nTraining: epochs %d–%d  batch_size=%d  (~%d batches/epoch)  lr=%g\n\n",
        start_epoch, n_epochs, batch_size, n_batches, kw[:lr])

    ep_pad = ndigits(n_epochs)
    for epoch in start_epoch:n_epochs
        order   = shuffle(rng, 1:length(examples))
        batches = [examples[order[i:min(i + batch_size - 1, end)]]
                   for i in 1:batch_size:length(order)]

        epoch_losses = Float32[]
        t_epoch = @elapsed for batch in batches
            ps, opt_state, loss = train_on_batch(model, ps, st, opt_state, batch, rng, dev)
            push!(epoch_losses, loss)
        end

        mean_loss = sum(epoch_losses) / length(epoch_losses)
        @printf("epoch %*d/%d  %5.1fs  mean_loss=%.4f  per-batch=%s\n",
            ep_pad, epoch, n_epochs, t_epoch, mean_loss,
            string(round.(epoch_losses; digits=3)))

        if kw[:checkpoint_dir] !== nothing && epoch % kw[:checkpoint_every] == 0
            save_checkpoint(kw[:checkpoint_dir], epoch, ps, st, cpu_dev)
        end
    end

    # always save a final checkpoint if a dir was given
    if kw[:checkpoint_dir] !== nothing
        save_checkpoint(kw[:checkpoint_dir], n_epochs, ps, st, cpu_dev)
    end

    # ── quick validity check (single-structure CPU API — keeps sample_flow simple)
    n_check = min(5, length(examples))
    ps_cpu  = cpu_dev(ps)
    st_cpu  = cpu_dev(st)
    println("\nValidity check on $n_check structures (single-structure, CPU):")
    for ex in examples[1:n_check]
        x_hat, _ = sample_flow(model, ps_cpu, st_cpu, ex.feat, ex.relpos,
                                ex.cond_features, rng; n_steps=20)
        rmsd    = aligned_rmsd(x_hat, Float32.(ex.x1))
        bonds   = backbone_bonds(ex.tokens)
        brmsd   = bond_length_rmsd(x_hat, bonds)
        elems   = [t.element for t in ex.tokens]
        clashes = clash_count(x_hat, elems, ex.feat.chain_idx, ex.feat.res_index)
        @printf("  %-12s %4d atoms  RMSD=%6.2f Å  clashes=%3d  bond-RMSD=%.3f Å\n",
            ex.pdb_id, ex.n_atoms, rmsd, clashes, brmsd)
    end

    (model=model, ps=ps_cpu, st=st_cpu, examples=examples)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
