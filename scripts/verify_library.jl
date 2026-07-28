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

A malformed (not just cleanly truncated — e.g. two writers' output
interleaved by a race, from before the atomic-write fix) file can crash
Julia's deserializer outright (segfault) rather than raise a catchable
error — `Serialization.jl` isn't hardened against corrupted input the way
it's hardened against a clean EOF. To survive that: every file's check is
logged to `<library-dir>/.verify_progress.log` *before* it's attempted, so
a crash leaves a record of exactly which file was being checked when it
died. Re-running this script picks up where it left off (already-resolved
files aren't re-checked) and automatically skips — without retrying, to
avoid an infinite crash loop — any file that was mid-check when a previous
run ended, printing it prominently so you can inspect/delete it by hand.

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

    # Resume support (see module docstring): replay the progress log, keep
    # only each file's *last* recorded status, and treat a dangling
    # "CHECKING" (no later OK/BAD for that file) as a suspected crash cause
    # -- skip it rather than risk immediately re-crashing on it.
    progress_path = joinpath(dir, ".verify_progress.log")
    checked = Set{String}()
    if isfile(progress_path)
        last_status = Dict{String,String}()
        for line in eachline(progress_path)
            parts = split(line, '\t'; limit=2)
            length(parts) == 2 || continue
            status, name = parts
            last_status[name] = status
        end
        suspects = String[]
        for (name, status) in last_status
            push!(checked, name)
            status in ("OK", "BAD") || push!(suspects, name)
        end
        if !isempty(suspects)
            println("Skipping $(length(suspects)) file(s) mid-check when a previous run ended")
            println("(most likely cause of a crash -- inspect/delete manually if you want them handled):")
            for s in sort(suspects)
                println("  ", s)
            end
        end
        println("Resuming: $(length(checked)) file(s) already resolved or flagged from a previous run")
    end

    remaining = filter(f -> basename(f) ∉ checked, files)
    println("Checking $(length(remaining)) of $(length(files)) files in $dir",
        dry_run ? " (dry run -- nothing will be deleted)" : "", "...")

    progress_io = open(progress_path, "a")
    n_ok, n_bad = 0, 0
    t = @elapsed for (i, f) in enumerate(remaining)
        # Written and flushed *before* attempting deserialize -- if this
        # exact call segfaults, this line survives as the crash record.
        println(progress_io, "CHECKING\t$(basename(f))")
        flush(progress_io)
        try
            Serialization.deserialize(f)
            n_ok += 1
            println(progress_io, "OK\t$(basename(f))")
        catch e
            n_bad += 1
            println("  corrupt: $(basename(f)): $(sprint(showerror, e))")
            println(progress_io, "BAD\t$(basename(f))")
            dry_run || rm(f)
        end
        flush(progress_io)
        i % 5000 == 0 && @printf("  ...%d/%d checked (%d OK, %d corrupt so far)\n", i, length(remaining), n_ok, n_bad)
    end
    close(progress_io)

    @printf("\nDone in %.1fs: %d OK, %d corrupt%s\n",
        t, n_ok, n_bad, dry_run ? " (left in place -- dry run)" : " (deleted)")
    if n_bad > 0 && !dry_run
        println("Re-run scripts/preprocess_dataset.jl against the same source directory")
        println("and --library-dir to regenerate the $n_bad removed file(s).")
    end
    if n_ok + n_bad == length(remaining) && isfile(progress_path)
        rm(progress_path)
        println("(all remaining files resolved cleanly -- removed $progress_path)")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
