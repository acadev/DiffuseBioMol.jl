using Test
using DiffuseBioMol

@testset "Data.list_structure_files: matches by extension, ignores non-structure files" begin
    mktempdir() do dir
        write(joinpath(dir, "a.pdb"), SAMPLE_PDB)
        write(joinpath(dir, "b.CIF"), "")  # case-insensitive match, even though it's empty
        write(joinpath(dir, "notes.txt"), "not a structure file")
        mkpath(joinpath(dir, "subdir"))
        write(joinpath(dir, "subdir", "c.pdb"), SAMPLE_PDB)

        files = list_structure_files(dir)
        @test length(files) == 2
        @test all(f -> lowercase(splitext(f)[2]) in (".pdb", ".cif"), files)
        @test issorted(files)

        files_recursive = list_structure_files(dir; recursive=true)
        @test length(files_recursive) == 3
    end

    @test_throws ArgumentError list_structure_files("/no/such/directory/at/all")
end

@testset "Data.largest_chain: picks the chain with the most residues" begin
    residues = parse_structure_string(SAMPLE_PDB)
    # Chain A: ALA + GLY (2 residues); chain B: ZN (1); chain C: HEM (1); water dropped.
    @test largest_chain(residues) == "A"

    @test_throws ArgumentError largest_chain(typeof(residues)())
end

@testset "Data.list_structure_files + parse_structure: end-to-end local-directory loading" begin
    mktempdir() do dir
        write(joinpath(dir, "sample.pdb"), SAMPLE_PDB)
        files = list_structure_files(dir)
        residues = parse_structure(only(files))
        chain = largest_chain(residues)
        tokens = tokenize_structure(restrict_to_chain(residues, chain))
        @test !isempty(tokens)
        @test all(t -> t.chain_id == "A", tokens)
    end
end
