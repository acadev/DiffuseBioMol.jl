include(joinpath(@__DIR__, "..", "scripts", "download_rcsb_dataset.jl"))
using .RCSBDownloader

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
end
