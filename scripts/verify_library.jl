"""
Verify every structure file in a `--library-dir` (as built by
`preprocess_dataset.jl`) actually deserializes, and delete any that don't.

Why this is needed: a truncated/corrupt file — most commonly from a
preprocessing shard getting killed mid-write (OOM, a job time limit, a
manual interrupt, or a network-filesystem hiccup at scale) — is invisible to
`preprocess_dataset.jl`'s idempotency check, which only tests file
*existence* (`isfile(out_path)`), not validity. Once that gap opens, the
corrupt file is silently skipped by every future preprocessing re-run
(it "exists", so it's treated as done) and will keep failing with an
EOFError/similar every time something tries to load it — forever, unless
actively found and removed. Deleting it here makes it look "missing" again,
so the next `preprocess_dataset.jl` run against the same `--library-dir`
naturally regenerates it from the original source file.

`preprocess_dataset.jl` itself now writes atomically (temp file + rename)
specifically so new corruption of this kind shouldn't occur going forward —
this script is for cleaning up anything already written before that fix, or
for periodic sanity-checking a library that's been through a rough
preprocessing run.

Usage:
    julia --project=. scripts/verify_library.jl ./library

    # Report only, don't delete anything
    julia --project=. scripts/verify_library.jl ./library --dry-run
"""

import Serialization
using DiffuseBioMol  # required: deserializing needs the defining types loaded
using Printf

function main(args=ARGS)
    dry_run = "--dry-run" in args
    dirs = filter(a -> a != "--dry-run", args)
    length(dirs) == 1 || error("usage: verify_library.jl <library-dir> [--dry-run]")
    dir = only(dirs)
    isdir(dir) || error("'$dir' is not a directory")

    files = filter(f -> endswith(f, ".jls"), readdir(dir; join=true))
    println("Checking $(length(files)) files in $dir", dry_run ? " (dry run -- nothing will be deleted)" : "", "...")

    n_ok, n_bad = 0, 0
    t = @elapsed for f in files
        try
            Serialization.deserialize(f)
            n_ok += 1
        catch e
            n_bad += 1
            println("  corrupt: $(basename(f)): $(sprint(showerror, e))")
            dry_run || rm(f)
        end
    end

    @printf("\nDone in %.1fs: %d OK, %d corrupt%s\n",
        t, n_ok, n_bad, dry_run ? " (left in place -- dry run)" : " (deleted)")
    if n_bad > 0 && !dry_run
        println("Re-run scripts/preprocess_dataset.jl against the same source directory")
        println("and --library-dir to regenerate the $n_bad removed file(s).")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
