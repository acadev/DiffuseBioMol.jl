"""
Build a reproducible protein-chain corpus from a locally staged PDB/mmCIF tree.

This script is deliberately download-free: stage candidate coordinate files on
the training filesystem first, then run, for example:

    JULIA_NUM_THREADS=16 julia --project=. scripts/curate_protein_dataset.jl /data/pdb-raw /data/pdb10k \
        --n-structures=12000 --max-atoms=1200 --seed=20260909 --concurrency=16

The output directory must not already contain files. Every selected source file
is copied with a deterministic name; `manifest.toml` records its source and
quality fields, while `sequences.fasta` is ready for external sequence
clustering (for example, with MMseqs2 at 30% identity). After selecting one
representative per cluster, rerun this script on that representative-only tree
to construct the final 10k corpus.
"""

module CurateProteinDataset

using DiffuseBioMol
using Random, TOML, Printf, Dates

export inspect_candidate, curate, main

const AA1 = Dict(
    "ALA" => 'A', "ARG" => 'R', "ASN" => 'N', "ASP" => 'D', "CYS" => 'C',
    "GLN" => 'Q', "GLU" => 'E', "GLY" => 'G', "HIS" => 'H', "ILE" => 'I',
    "LEU" => 'L', "LYS" => 'K', "MET" => 'M', "PHE" => 'F', "PRO" => 'P',
    "SER" => 'S', "THR" => 'T', "TRP" => 'W', "TYR" => 'Y', "VAL" => 'V',
)

struct CuratedCandidate
    source::String
    label::String
    chain_id::String
    n_residues::Int
    n_atoms::Int
    backbone_coverage::Float64
    sequence::String
end

"""Detect obvious NMR ensembles before BioStructures flattens their models."""
function is_nmr(path::AbstractString)
    content = lowercase(read(path, String))
    occursin("solution nmr", content) || occursin("solid-state nmr", content) || occursin("nmr ensemble", content)
end

function standard_protein_chain(residues)
    chain = largest_chain(residues)
    selected = restrict_to_chain(residues, chain)
    isempty(selected) && return nothing
    all(r -> r.modality == PROTEIN && haskey(AA1, r.res_name), selected) || return nothing
    selected
end

"""
    inspect_candidate(path; max_atoms, min_residues, min_backbone_coverage,
                      skip_nmr=true) -> CuratedCandidate | nothing

Applies the initial-baseline policy: one largest all-standard-protein chain,
adequate observed N/CA/C/O backbone coverage, and a token count below the
model's padded-atom budget. `nothing` means the file is deliberately excluded.
"""
function inspect_candidate(path::AbstractString; max_atoms::Int=1200, min_residues::Int=40,
    min_backbone_coverage::Real=0.95, skip_nmr::Bool=true)
    skip_nmr && is_nmr(path) && return nothing
    residues = parse_structure(path)
    selected = standard_protein_chain(residues)
    selected === nothing && return nothing
    n_residues = length(selected)
    n_residues >= min_residues || return nothing

    complete = count(r -> all(atom -> haskey(r.present_atoms, atom), ("N", "CA", "C", "O")), selected)
    coverage = complete / n_residues
    coverage >= min_backbone_coverage || return nothing

    tokens = tokenize_structure(selected)
    length(tokens) <= max_atoms || return nothing
    sequence = join(AA1[r.res_name] for r in selected)
    CuratedCandidate(abspath(path), splitext(basename(path))[1], first(selected).chain_id,
        n_residues, length(tokens), coverage, sequence)
end

function curate_write_manifest(path::AbstractString, candidates, skipped, options)
    doc = Dict(
        "created_at" => string(now()),
        "options" => options,
        "selected" => [Dict(
            "source" => c.source, "label" => c.label, "chain_id" => c.chain_id,
            "n_residues" => c.n_residues, "n_atoms" => c.n_atoms,
            "backbone_coverage" => c.backbone_coverage,
        ) for c in candidates],
        "skipped" => skipped,
    )
    open(path, "w") do io
        TOML.print(io, doc)
    end
end

function write_fasta(path::AbstractString, candidates)
    open(path, "w") do io
        for (i, candidate) in enumerate(candidates)
            println(io, ">CURATED_", lpad(i, 5, '0'), "|", candidate.label, "|chain=", candidate.chain_id)
            println(io, candidate.sequence)
        end
    end
end

"""
    curate(input_dir, output_dir; n_structures=12000, seed=20260909,
           max_atoms=1200, min_residues=40, min_backbone_coverage=0.95,
           skip_nmr=true, concurrency=1) -> Vector{CuratedCandidate}

Inspect every supported coordinate file recursively, then deterministically
sample up to `n_structures` accepted chains and copy the source coordinate
files to an otherwise empty output directory. The manifest is the provenance
record to retain alongside the dataset.
"""
function representative_prefixes(path::AbstractString)
    prefixes = Set{String}()
    for line in eachline(path)
        startswith(line, '>') || continue
        header = first(split(line[2:end], '|'; limit=2))
        startswith(header, "CURATED_") || error("representative FASTA header must begin with CURATED_: $line")
        push!(prefixes, replace(header, "CURATED_" => "") * "_")
    end
    isempty(prefixes) && error("no CURATED_ headers found in representative FASTA: $path")
    prefixes
end

"""Inspect independent coordinate files concurrently, retaining source-list order in results."""
function inspect_files(files; max_atoms::Int, min_residues::Int, min_backbone_coverage::Real,
                       skip_nmr::Bool, concurrency::Int)
    concurrency > 0 || throw(ArgumentError("concurrency must be positive"))
    n_workers = min(concurrency, length(files))
    n_workers > Threads.nthreads() && @warn "curation concurrency exceeds JULIA_NUM_THREADS; workers will not run in parallel" concurrency threads=Threads.nthreads()
    println("Inspecting $(length(files)) coordinate files with $n_workers worker$(n_workers == 1 ? "" : "s").")
    jobs = Channel{Tuple{Int,String}}(length(files))
    results = Channel{Any}(length(files))
    for job in enumerate(files)
        put!(jobs, job)
    end
    close(jobs)
    workers = [Threads.@spawn begin
        for (i, path) in jobs
            try
                candidate = inspect_candidate(path; max_atoms, min_residues, min_backbone_coverage, skip_nmr)
                put!(results, (i=i, candidate=candidate, skipped=nothing))
            catch err
                put!(results, (i=i, candidate=nothing, skipped="$path ($(sprint(showerror, err)))"))
            end
        end
    end for _ in 1:n_workers]

    outcomes = Vector{Any}(undef, length(files))
    for completed in 1:length(files)
        outcome = take!(results)
        outcomes[outcome.i] = outcome
        (completed % 1_000 == 0 || completed == length(files)) &&
            println("Curation inspection: $completed / $(length(files)) files complete.")
    end
    foreach(fetch, workers)
    accepted = CuratedCandidate[]
    skipped = String[]
    for (path, outcome) in zip(files, outcomes)
        if outcome.candidate !== nothing
            push!(accepted, outcome.candidate)
        elseif outcome.skipped !== nothing
            push!(skipped, outcome.skipped)
        else
            push!(skipped, "$path (did not meet curation criteria)")
        end
    end
    accepted, skipped
end

function curate(input_dir::AbstractString, output_dir::AbstractString; n_structures::Int=12_000,
    seed::Int=20_260_909, max_atoms::Int=1200, min_residues::Int=40,
    min_backbone_coverage::Real=0.95, skip_nmr::Bool=true,
    representatives_fasta::Union{Nothing,AbstractString}=nothing, concurrency::Int=1)
    isdir(input_dir) || throw(ArgumentError("input directory does not exist: $input_dir"))
    abspath(input_dir) == abspath(output_dir) && throw(ArgumentError("input and output directories must differ"))
    isdir(output_dir) && !isempty(readdir(output_dir)) && throw(ArgumentError("output directory must be empty: $output_dir"))
    n_structures > 0 || throw(ArgumentError("n_structures must be positive"))
    0 < min_backbone_coverage <= 1 || throw(ArgumentError("min_backbone_coverage must lie in (0, 1]"))
    mkpath(output_dir)

    files = list_structure_files(input_dir; recursive=true)
    if representatives_fasta !== nothing
        prefixes = representative_prefixes(representatives_fasta)
        files = filter(path -> any(prefix -> startswith(basename(path), prefix), prefixes), files)
        isempty(files) && error("no coordinate files in $input_dir matched $representatives_fasta")
    end
    accepted, skipped = inspect_files(files; max_atoms, min_residues, min_backbone_coverage, skip_nmr, concurrency)
    isempty(accepted) && error("no candidates passed curation from $input_dir")
    ordered = sort(accepted; by = c -> c.source)
    selected = ordered[randperm(MersenneTwister(seed), length(ordered))[1:min(n_structures, length(ordered))]]
    for (i, candidate) in enumerate(selected)
        extension = lowercase(splitext(candidate.source)[2])
        destination = joinpath(output_dir, @sprintf("%05d_%s%s", i, candidate.label, extension))
        cp(candidate.source, destination)
    end
    options = Dict(
        "input_dir" => abspath(input_dir), "n_structures_requested" => n_structures,
        "n_structures_selected" => length(selected), "seed" => seed, "max_atoms" => max_atoms,
        "min_residues" => min_residues, "min_backbone_coverage" => Float64(min_backbone_coverage),
        "skip_nmr" => skip_nmr, "concurrency" => concurrency,
        "representatives_fasta" => something(representatives_fasta, ""),
    )
    curate_write_manifest(joinpath(output_dir, "manifest.toml"), selected, skipped, options)
    write_fasta(joinpath(output_dir, "sequences.fasta"), selected)
    println("Selected $(length(selected)) / $(length(accepted)) accepted candidates; $(length(skipped)) skipped.")
    println("Wrote coordinate files, manifest.toml, and sequences.fasta to $output_dir")
    selected
end

function main(args=ARGS)
    length(args) >= 2 || error("usage: julia --project=. scripts/curate_protein_dataset.jl INPUT_DIR OUTPUT_DIR [--n-structures=N] [--max-atoms=N] [--min-residues=N] [--seed=N] [--concurrency=N]")
    input_dir, output_dir = args[1], args[2]
    options = Dict{String,Int}("n-structures" => 12_000, "max-atoms" => 1200, "min-residues" => 40, "seed" => 20_260_909, "concurrency" => 1)
    representatives_fasta = nothing
    for arg in args[3:end]
        startswith(arg, "--") && occursin('=', arg) || error("invalid option: $arg")
        key, value = split(arg[3:end], '='; limit=2)
        if key == "representatives-fasta"
            representatives_fasta = value
        else
            haskey(options, key) || error("unsupported option: --$key")
            options[key] = parse(Int, value)
        end
    end
    curate(input_dir, output_dir; n_structures=options["n-structures"], max_atoms=options["max-atoms"],
        min_residues=options["min-residues"], seed=options["seed"], representatives_fasta,
        concurrency=options["concurrency"])
end

end # module CurateProteinDataset

if abspath(PROGRAM_FILE) == @__FILE__
    CurateProteinDataset.main()
end
