"""
Build a reusable, on-disk library of preprocessed structures — one small
`Serialization.serialize`'d file per structure (parsed, tokenized,
featurized; the same shape `train_cuda.jl` trains on) — decoupled from any
single training run's `--n-targets`/`--max-atoms`. Run this once against a
local PDB/mmCIF mirror; `train_cuda.jl --library-dir <dir>` then reads
whatever subset a given run wants straight from the library, with no
re-parsing, and different runs can freely use different `--n-targets`/
`--max-atoms` without forcing a rebuild (unlike `train_cuda.jl`'s
`--cache-file`, which is keyed to one exact argument combination).

Idempotent / incremental: a structure already present in `--library-dir` is
skipped, so re-running this after new files land in the source mirror only
processes what's new. `--limit` caps how many *new* files this run writes,
not the library's total size — safe to call repeatedly to grow the library
over time (e.g. `--limit 1000` today, `--limit 1000` again next week once
more source files have arrived). When combined with sharding (below),
`--limit` applies *per shard*, not to the run as a whole.

Usage:
    julia --project=. scripts/preprocess_dataset.jl /path/to/pdbs \\
        --library-dir ./library --limit 1000

    # Nested PDB mirror layout (e.g. standard rsync mirror: pdb/ab/1abc.ent)
    julia --project=. scripts/preprocess_dataset.jl /path/to/pdbs \\
        --library-dir ./library --recursive

    # Then train against the library (see train_cuda.jl's module docstring):
    julia --project=. scripts/train_cuda.jl --library-dir ./library --n-targets 200

Parallelizing across many structures (see --num-shards/--shard-id below):
this script's per-structure work (parse -> tokenize -> featurize -> write)
is fully independent across structures — no shared state, no cross-file
dependencies, and it's pure CPU work (no GPU involved at all) — so it's a
textbook case for splitting across processes rather than one long serial
run. Each shard computes the exact same candidate file list from `data_dir`
and takes every `num_shards`-th file starting at `shard_id` (0-indexed), so
running shards 0..num_shards-1 covers every candidate exactly once with zero
coordination between them; writes never collide since every shard's output
filenames are disjoint. Combined with this script already being idempotent
(skips files already in --library-dir), a partially-failed or interrupted
parallel run is always safe to just re-launch.

    # Plain background processes on one multi-core machine (N = core count)
    N=32
    for i in \$(seq 0 \$((N-1))); do
        julia --project=. scripts/preprocess_dataset.jl /path/to/pinder \\
            --library-dir ./library --recursive --num-shards \$N --shard-id \$i &
    done
    wait

    # SLURM array job (one task per shard, across one or many CPU nodes —
    # prefer a CPU partition/queue over a GPU allocation for this: it's
    # pure CPU work and GPU-node time is the expensive resource to save)
    #SBATCH --array=0-31
    julia --project=. scripts/preprocess_dataset.jl /path/to/pinder \\
        --library-dir ./library --recursive \\
        --num-shards 32 --shard-id \$SLURM_ARRAY_TASK_ID
"""

using Printf
import Serialization

using DiffuseBioMol

include(joinpath(@__DIR__, "dataset_utils.jl"))

function parse_args(args)
    kw = Dict{Symbol,Any}(
        :data_dir    => nothing,
        :library_dir => nothing,
        :limit       => nothing,
        :recursive   => false,
        :num_shards  => 1,      # --num-shards: total number of parallel workers
        :shard_id    => 0,      # --shard-id: this worker's index, 0-based, in [0, num_shards)
    )
    i = 1
    while i <= length(args)
        a = args[i]
        nxt = i < length(args) ? args[i+1] : ""
        if     a == "--library-dir" ; kw[:library_dir] = nxt;             i += 2
        elseif a == "--limit"       ; kw[:limit]       = parse(Int, nxt); i += 2
        elseif a == "--recursive"   ; kw[:recursive]   = true;            i += 1
        elseif a == "--num-shards"  ; kw[:num_shards]  = parse(Int, nxt); i += 2
        elseif a == "--shard-id"    ; kw[:shard_id]    = parse(Int, nxt); i += 2
        elseif startswith(a, "--")
            @warn "Unknown flag: $a (ignored)"
            i += 2
        else
            kw[:data_dir] = a; i += 1
        end
    end
    kw
end

function main(args=ARGS)
    kw = parse_args(args)
    kw[:data_dir] === nothing && error("a source directory of PDB/mmCIF files is required (positional arg)")
    kw[:library_dir] === nothing && error("--library-dir is required")
    kw[:num_shards] >= 1 || error("--num-shards must be >= 1 (got $(kw[:num_shards]))")
    0 <= kw[:shard_id] < kw[:num_shards] || error(
        "--shard-id must be in [0, num_shards) — got shard_id=$(kw[:shard_id]) with num_shards=$(kw[:num_shards])")

    all_files = list_structure_files(kw[:data_dir]; recursive=kw[:recursive])
    isempty(all_files) && error("no PDB/mmCIF files found under $(kw[:data_dir]) (try --recursive for nested layouts)")
    mkpath(kw[:library_dir])

    # Every shard computes this same full list and deterministically takes
    # every num_shards-th entry — no coordination needed between shards, and
    # shards 0..num_shards-1 together cover every file exactly once.
    files = kw[:num_shards] == 1 ? all_files :
        [f for (idx, f) in enumerate(all_files) if (idx - 1) % kw[:num_shards] == kw[:shard_id]]

    @printf("Source  : %s  (%d candidate files found)\n", kw[:data_dir], length(all_files))
    kw[:num_shards] > 1 && @printf("Shard   : %d/%d  (%d files assigned to this shard)\n",
        kw[:shard_id], kw[:num_shards], length(files))
    @printf("Library : %s\n", kw[:library_dir])
    println("Limit   : ", kw[:limit] === nothing ? "none (process every candidate)" : "$(kw[:limit]) new structures")

    n_written, n_present, n_failed = 0, 0, 0
    t = @elapsed for path in files
        kw[:limit] !== nothing && n_written >= kw[:limit] && break

        label = splitext(basename(path))[1]
        out_path = joinpath(kw[:library_dir], "$label.jls")
        if isfile(out_path)
            n_present += 1
            continue
        end

        try
            ex = load_local_example(path)
            Serialization.serialize(out_path, ex)
            n_written += 1
        catch e
            n_failed += 1
            println("  skip $label: $(sprint(showerror, e))")
        end
    end

    @printf("\nDone: %d written, %d already present (skipped), %d failed, in %.1fs\n",
        n_written, n_present, n_failed, t)
    @printf("Library now holds %d preprocessed structures.\n",
        count(f -> endswith(f, ".jls"), readdir(kw[:library_dir])))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
