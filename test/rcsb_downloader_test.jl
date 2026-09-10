include(joinpath(@__DIR__, "..", "scripts", "download_rcsb_dataset.jl"))
using .RCSBDownloader
using TOML

@testset "RCSB downloader: query encoding and cache helpers" begin
    query = experimental_protein_query(start=25, rows=10)
    @test query["return_type"] == "entry"
    @test query["request_options"]["paginate"] == Dict("start" => 25, "rows" => 10)
    @test query["request_options"]["results_content_type"] == ["experimental"]
    @test !haskey(experimental_protein_query(return_counts=true)["request_options"], "paginate")
    @test extract_ids(Dict("result_set" => ["1abc", Dict("identifier" => "2def")])) == ["1ABC", "2DEF"]

    mktempdir() do dir
        cif = joinpath(dir, "demo.cif")
        write(cif, "data_demo\n#\n")
        @test RCSBDownloader.valid_cif(cif)
        @test !RCSBDownloader.valid_cif(joinpath(dir, "missing.cif"))
        ids_path = joinpath(dir, "ids.txt")
        RCSBDownloader.write_ids(ids_path, ["1ABC", "2DEF"])
        @test RCSBDownloader.read_ids(ids_path) == ["1ABC", "2DEF"]
    end

    mktempdir() do dir
        ids = ["3AAA", "1BBB", "2BAD", "4CCC"]
        RCSBDownloader.write_ids(joinpath(dir, "sampled_ids.txt"), ids)
        active, peak = Ref(0), Ref(0)
        function mock_download(id, _cache_dir; retries)
            active[] += 1
            peak[] = max(peak[], active[])
            sleep(0.01) # yields, so the async worker bound is observable without network I/O
            active[] -= 1
            id == "2BAD" && error("fixture failure")
            id == "1BBB" ? :cached : :downloaded
        end
        result = download_dataset(dir; n_structures=length(ids), retries=1, concurrency=2,
            download_fn=mock_download)
        @test result.concurrency == 2
        @test peak[] == 2
        @test result.downloaded == ["3AAA", "4CCC"]
        @test result.cached == ["1BBB"]
        @test result.failures == Dict("2BAD" => "fixture failure")
        manifest = TOML.parsefile(joinpath(dir, "download_manifest.toml"))
        @test manifest["concurrency"] == 2
        @test manifest["downloaded"] == result.downloaded
    end
end
