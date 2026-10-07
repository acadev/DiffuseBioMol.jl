using TOML

include(joinpath(@__DIR__, "..", "scripts", "curate_protein_dataset.jl"))
using .CurateProteinDataset

@testset "protein dataset curation: accepts complete standard-protein largest chains" begin
    mktempdir() do root
        input_dir, output_dir = joinpath(root, "input"), joinpath(root, "output")
        mkpath(input_dir)
        for i in 1:3
            write(joinpath(input_dir, "candidate_$i.pdb"), SAMPLE_PDB)
        end
        candidate = inspect_candidate(joinpath(input_dir, "candidate_1.pdb"); max_atoms=100, min_residues=1)
        @test candidate !== nothing
        @test candidate.chain_id == "A"
        @test candidate.n_residues == 2
        @test candidate.backbone_coverage == 1.0
        @test inspect_candidate(joinpath(input_dir, "candidate_1.pdb"); max_atoms=1, min_residues=1) === nothing
        @test inspect_candidate(joinpath(input_dir, "candidate_1.pdb"); max_atoms=0, min_residues=1) !== nothing

        selected = curate(input_dir, output_dir; n_structures=2, seed=1, max_atoms=100, min_residues=1, concurrency=2)
        @test length(selected) == 2
        @test length(filter(f -> endswith(f, ".pdb"), readdir(output_dir))) == 2
        @test isfile(joinpath(output_dir, "manifest.toml"))
        @test isfile(joinpath(output_dir, "sequences.fasta"))
        manifest = TOML.parsefile(joinpath(output_dir, "manifest.toml"))
        @test length(manifest["selected"]) == 2
        @test manifest["options"]["concurrency"] == 2

        representatives = joinpath(root, "representatives.fasta")
        write(representatives, ">CURATED_00001|$(first(selected).label)|chain=A\nAG\n")
        clustered_output = joinpath(root, "clustered-output")
        clustered = curate(output_dir, clustered_output; n_structures=10, max_atoms=100, min_residues=1,
            representatives_fasta=representatives)
        @test length(clustered) == 1
        @test only(clustered).label == "00001_$(first(selected).label)"
    end
end
