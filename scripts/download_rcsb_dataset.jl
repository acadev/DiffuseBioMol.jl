"""
Download and cache a seeded random sample of experimental protein structures
from RCSB PDB in mmCIF format.

Usage:
    JULIA_NUM_THREADS=8 julia --project=. scripts/download_rcsb_dataset.jl /data/rcsb-raw \
        --n-structures=15000 --seed=20260909 --concurrency=4

The first run queries RCSB's Search API for experimental entries containing at
least one protein polymer entity, samples IDs with the supplied seed, and saves
the exact selection in `sampled_ids.txt`. Subsequent invocations reuse that
file, skip valid cached files, and retry only missing/failed IDs. Feed the
resulting directory to `curate_protein_dataset.jl`; request more than 10k here
because structural-quality and homology filters will remove candidates later.
"""

module RCSBDownloader

using Downloads, JSON, Random, TOML, Dates

export experimental_protein_query, extract_ids, download_dataset, main

const SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
const FILE_URL_PREFIX = "https://files.rcsb.org/download/"

"""Search request for released experimental entries with at least one protein entity."""
function experimental_protein_query(; start::Int=0, rows::Int=0, return_counts::Bool=false)
    options = Dict{String,Any}(
        "results_content_type" => ["experimental"],
        "results_verbosity" => "compact",
        "return_counts" => return_counts,
    )
    return_counts || (options["paginate"] = Dict("start" => start, "rows" => rows))
    Dict(
        "query" => Dict(
            "type" => "terminal", "service" => "text",
            "parameters" => Dict(
                "attribute" => "rcsb_entry_info.polymer_entity_count_protein",
                "operator" => "greater", "value" => 0,
            ),
        ),
        "return_type" => "entry",
        "request_options" => options,
    )
end

function post_json(url::AbstractString, payload)
    body = JSON.json(payload)
    output = IOBuffer()
    response = Downloads.request(url; method="POST", input=IOBuffer(body), output,
        headers=["Content-Type" => "application/json", "Accept" => "application/json"])
    response_body = String(take!(output))
    200 <= response.status < 300 || error("RCSB Search API returned HTTP $(response.status): $response_body")
    JSON.parse(response_body)
end

"""Handle compact string hits and regular object hits from the Search API."""
function extract_ids(response)
    result_set = get(response, "result_set", Any[])
    [uppercase(item isa AbstractString ? item : String(item["identifier"])) for item in result_set]
end

function search_count()
    response = post_json(SEARCH_URL, experimental_protein_query(return_counts=true))
    Int(response["total_count"])
end

function fetch_page(start::Int, rows::Int)
    extract_ids(post_json(SEARCH_URL, experimental_protein_query(start=start, rows=rows)))
end

"""Sample uniform result positions, fetching only the Search API pages that contain them."""
function sampled_rcsb_ids(n_structures::Int, rng::AbstractRNG; page_size::Int=10_000)
    total = search_count()
    total >= n_structures || error("RCSB query has only $total matching entries; requested $n_structures")
    positions = sort(randperm(rng, total)[1:n_structures]) .- 1  # Search API pagination is zero-based.
    pages = Dict{Int,Vector{String}}()
    sampled = String[]
    for position in positions
        page_start = (position ÷ page_size) * page_size
        ids = get!(pages, page_start) do
            fetch_page(page_start, min(page_size, total - page_start))
        end
        offset = position - page_start + 1
        offset <= length(ids) || error("RCSB result page $page_start was shorter than expected")
        push!(sampled, ids[offset])
    end
    sampled
end

function valid_cif(path::AbstractString)
    isfile(path) && filesize(path) > 0 || return false
    open(path, "r") do io
        startswith(String(read(io, min(filesize(path), 64))), "data_")
    end
end

function read_ids(path::AbstractString)
    [strip(line) for line in eachline(path) if !isempty(strip(line))]
end

function write_ids(path::AbstractString, ids)
    open(path, "w") do io
        for id in ids
            println(io, id)
        end
    end
end

function download_one(id::AbstractString, cache_dir::AbstractString; retries::Int=3)
    destination = joinpath(cache_dir, lowercase(id) * ".cif")
    valid_cif(destination) && return :cached
    url = FILE_URL_PREFIX * uppercase(id) * ".cif"
    last_error = nothing
    for attempt in 1:retries
        temporary = tempname(cache_dir)
        try
            Downloads.download(url, temporary)
            valid_cif(temporary) || error("download was not a valid mmCIF file")
            mv(temporary, destination; force=true)
            return :downloaded
        catch err
            last_error = err
            isfile(temporary) && rm(temporary; force=true)
            attempt < retries && sleep(0.5 * attempt)
        end
    end
    throw(last_error)
end

function write_download_manifest(path::AbstractString, ids, downloaded, cached, failures, seed, concurrency)
    doc = Dict(
        "created_at" => string(now()), "seed" => seed, "requested_ids" => ids,
        "downloaded" => downloaded, "cached" => cached, "failures" => failures,
        "concurrency" => concurrency,
    )
    open(path, "w") do io
        TOML.print(io, doc)
    end
end

"""
    download_dataset(cache_dir; n_structures=15000, seed=20260909,
                     retries=3, concurrency=1) -> NamedTuple

Creates/reuses a persistent `sampled_ids.txt` selection. Downloads are
performed by a bounded number of worker tasks. Each complete result is
collected before the manifest is written; manifest lists retain the sampled-ID
order rather than nondeterministic completion order. A rerun skips valid files
and retries only missing/failed IDs, including after an interruption.
"""
function download_dataset(cache_dir::AbstractString; n_structures::Int=15_000,
    seed::Int=20_260_909, retries::Int=3, concurrency::Int=1, download_fn=download_one)
    n_structures > 0 || throw(ArgumentError("n_structures must be positive"))
    retries > 0 || throw(ArgumentError("retries must be positive"))
    concurrency > 0 || throw(ArgumentError("concurrency must be positive"))
    mkpath(cache_dir)
    sampled_path = joinpath(cache_dir, "sampled_ids.txt")
    ids = if isfile(sampled_path)
        prior = read_ids(sampled_path)
        length(prior) == n_structures || error("$sampled_path contains $(length(prior)) IDs, but n_structures=$n_structures; use a new cache directory")
        prior
    else
        selection = sampled_rcsb_ids(n_structures, MersenneTwister(seed))
        write_ids(sampled_path, selection)
        selection
    end

    n_workers = min(concurrency, length(ids))
    println("Downloading $(length(ids)) structures with $n_workers concurrent worker$(n_workers == 1 ? "" : "s").")
    jobs = Channel{Tuple{Int,String}}(length(ids))
    results = Channel{Any}(length(ids))
    for job in enumerate(ids)
        put!(jobs, job)
    end
    close(jobs)
    workers = [Threads.@spawn begin
        for (i, id) in jobs
            try
                put!(results, (i=i, id=id, status=download_fn(id, cache_dir; retries), failure=nothing))
            catch err
                put!(results, (i=i, id=id, status=:failed, failure=sprint(showerror, err)))
            end
        end
    end for _ in 1:n_workers]

    statuses = Vector{Symbol}(undef, length(ids))
    errors = Vector{Union{Nothing,String}}(undef, length(ids))
    completed = downloaded_count = cached_count = failed_count = 0
    for _ in ids
        result = take!(results)
        statuses[result.i] = result.status
        errors[result.i] = result.failure
        completed += 1
        result.status == :downloaded && (downloaded_count += 1)
        result.status == :cached && (cached_count += 1)
        result.status == :failed && (failed_count += 1)
        (completed % 100 == 0 || completed == length(ids)) && println("$completed / $(length(ids)): $downloaded_count downloaded, $cached_count cached, $failed_count failed")
    end
    foreach(fetch, workers)
    downloaded = [id for (id, status) in zip(ids, statuses) if status == :downloaded]
    cached = [id for (id, status) in zip(ids, statuses) if status == :cached]
    failures = Dict(id => something(errors[i], "unknown download failure") for (i, id) in enumerate(ids) if statuses[i] == :failed)
    write_download_manifest(joinpath(cache_dir, "download_manifest.toml"), ids, downloaded, cached, failures, seed, concurrency)
    isempty(failures) || println("$(length(failures)) downloads failed; rerun the same command to retry them.")
    (ids=ids, downloaded=downloaded, cached=cached, failures=failures, concurrency=n_workers)
end

function main(args=ARGS)
    isempty(args) && error("usage: julia --project=. scripts/download_rcsb_dataset.jl CACHE_DIR [--n-structures=N] [--seed=N] [--retries=N] [--concurrency=1]")
    cache_dir = args[1]
    options = Dict{String,Int}("n-structures" => 15_000, "seed" => 20_260_909, "retries" => 3, "concurrency" => 1)
    for arg in args[2:end]
        startswith(arg, "--") && occursin('=', arg) || error("invalid option: $arg")
        key, value = split(arg[3:end], '='; limit=2)
        haskey(options, key) || error("unsupported option: --$key")
        options[key] = parse(Int, value)
    end
    download_dataset(cache_dir; n_structures=options["n-structures"], seed=options["seed"], retries=options["retries"], concurrency=options["concurrency"])
end

end # module RCSBDownloader

if abspath(PROGRAM_FILE) == @__FILE__
    RCSBDownloader.main()
end
