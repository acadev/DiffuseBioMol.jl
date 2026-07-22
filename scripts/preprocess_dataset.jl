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
more source files have arrived).

Usage:
    julia --project=. scripts/preprocess_dataset.jl /path/to/pdbs \\
        --library-dir ./library --limit 1000

    # Nested PDB mirror layout (e.g. standard rsync mirror: pdb/ab/1abc.ent)
    julia --project=. scripts/preprocess_dataset.jl /path/to/pdbs \\
        --library-dir ./library --recursive

    # Then train against the library (see train_cuda.jl's module docstring):
    julia --project=. scripts/train_cuda.jl --library-dir ./library --n-targets 200
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
    )
    i = 1
    while i <= length(args)
        a = args[i]
        nxt = i < length(args) ? args[i+1] : ""
        if     a == "--library-dir" ; kw[:library_dir] = nxt;             i += 2
        elseif a == "--limit"       ; kw[:limit]       = parse(Int, nxt); i += 2
        elseif a == "--recursive"   ; kw[:recursive]   = true;            i += 1
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

    files = list_structure_files(kw[:data_dir]; recursive=kw[:recursive])
    isempty(files) && error("no PDB/mmCIF files found under $(kw[:data_dir]) (try --recursive for nested layouts)")
    mkpath(kw[:library_dir])

    @printf("Source  : %s  (%d candidate files found)\n", kw[:data_dir], length(files))
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
