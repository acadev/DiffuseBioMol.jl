using Test
using DiffuseBioMol

include("pdb_fixtures.jl")
include("data_local_dataset_test.jl")
include("augmentation_test.jl")
include("model_smoke_test.jl")
include("conditioning_test.jl")
include("geometry_test.jl")
include("verification_test.jl")
include("regression_test.jl")
include("atomworks_adapter_test.jl")
include("batching_test.jl")

@testset "AtomVocab" begin
    @test AtomVocab.residue_modality("ALA") == PROTEIN
    @test AtomVocab.residue_modality("A") == RNA
    @test AtomVocab.residue_modality("DA") == DNA
    @test_throws ArgumentError AtomVocab.residue_modality("HEM")

    @test atom_slots("GLY") == ["N", "CA", "C", "O"]
    @test "CB" in atom_slots("ALA")
    @test length(atom_slots("TRP")) == AtomVocab.MAX_ATOMS_PER_RESIDUE ||
          length(atom_slots("TRP")) <= AtomVocab.MAX_ATOMS_PER_RESIDUE
end

@testset "Tokenizer: protein residue with missing sidechain atoms" begin
    res = ParsedResidue(
        "A", 1, "SER", PROTEIN,
        Dict("N" => (0.0, 0.0, 0.0), "CA" => (1.0, 0.0, 0.0), "C" => (2.0, 0.0, 0.0), "O" => (3.0, 0.0, 0.0)),
    )
    tokens = tokenize_residue(res)
    @test length(tokens) == length(AtomVocab.atom_slots("SER"))

    backbone = filter(t -> t.atom_name in ("N", "CA", "C", "O"), tokens)
    @test all(t -> !t.is_virtual, backbone)
    @test all(t -> t.coord !== nothing, backbone)

    sidechain = filter(t -> t.atom_name in ("CB", "OG"), tokens)
    @test all(t -> t.is_virtual, sidechain)
    @test all(t -> t.coord === nothing, sidechain)

    @test all(t -> t.modality == PROTEIN, tokens)
    @test all(t -> t.chain_id == "A" && t.res_index == 1 && t.res_name == "SER", tokens)
end

@testset "Tokenizer: glycine has no sidechain slots" begin
    res = ParsedResidue("A", 2, "GLY", PROTEIN, Dict{String,NTuple{3,Float64}}())
    tokens = tokenize_residue(res)
    @test length(tokens) == 4
    @test all(t -> t.is_virtual, tokens)
end

@testset "Tokenizer: RNA residue" begin
    res = ParsedResidue(
        "B", 1, "G", RNA,
        Dict("P" => (0.0, 0.0, 0.0), "N9" => (1.0, 1.0, 1.0)),
    )
    tokens = tokenize_residue(res)
    @test length(tokens) == length(AtomVocab.atom_slots("G"))
    @test all(t -> t.modality == RNA, tokens)
    p_token = only(filter(t -> t.atom_name == "P", tokens))
    @test !p_token.is_virtual
    @test p_token.element == :P
end

@testset "Tokenizer: ligand and ion have no padding" begin
    ligand = ParsedResidue(
        "L", 1, "HEM", LIGAND,
        Dict("FE" => (0.0, 0.0, 0.0), "NA1" => (1.0, 0.0, 0.0)),
        Dict("FE" => :Fe),
    )
    tokens = tokenize_residue(ligand)
    @test length(tokens) == 2
    @test all(t -> !t.is_virtual, tokens)
    fe_token = only(filter(t -> t.atom_name == "FE", tokens))
    @test fe_token.element == :Fe
    @test fe_token.modality == LIGAND

    ion = ParsedResidue("I", 1, "ZN", ION, Dict("ZN" => (5.0, 5.0, 5.0)), Dict("ZN" => :Zn))
    ion_tokens = tokenize_residue(ion)
    @test length(ion_tokens) == 1
    @test ion_tokens[1].modality == ION
    @test ion_tokens[1].element == :Zn
end

@testset "tokenize_structure: multi-modality complex" begin
    residues = [
        ParsedResidue("A", 1, "ALA", PROTEIN, Dict("N" => (0.0, 0.0, 0.0), "CA" => (1.0, 0.0, 0.0), "C" => (2.0, 0.0, 0.0), "O" => (3.0, 0.0, 0.0), "CB" => (1.0, 1.0, 0.0))),
        ParsedResidue("B", 1, "DA", DNA, Dict{String,NTuple{3,Float64}}()),
        ParsedResidue("L", 1, "ATP", LIGAND, Dict("PA" => (10.0, 10.0, 10.0)), Dict("PA" => :P)),
        ParsedResidue("I", 1, "MG", ION, Dict("MG" => (0.0, 0.0, 0.0)), Dict("MG" => :Mg)),
    ]
    tokens = tokenize_structure(residues)
    modalities = Set(t.modality for t in tokens)
    @test modalities == Set([PROTEIN, DNA, LIGAND, ION])
    @test length(tokens) ==
          length(AtomVocab.atom_slots("ALA")) + length(AtomVocab.atom_slots("DA")) + 1 + 1
end

@testset "Data.parse_structure_string: end-to-end PDB -> tokens" begin
    residues = parse_structure_string(SAMPLE_PDB)

    # water (HOH) must be dropped
    @test !any(r -> r.res_name == "HOH", residues)
    @test length(residues) == 4  # ALA, GLY, ZN, HEM

    ala = only(filter(r -> r.res_name == "ALA", residues))
    @test ala.modality == PROTEIN
    @test ala.chain_id == "A"
    @test ala.res_index == 1
    @test Set(keys(ala.present_atoms)) == Set(["N", "CA", "C", "O", "CB"])

    gly = only(filter(r -> r.res_name == "GLY", residues))
    @test gly.modality == PROTEIN
    @test length(gly.present_atoms) == 4

    zn = only(filter(r -> r.res_name == "ZN", residues))
    @test zn.modality == ION
    @test zn.elements["ZN"] == :Zn

    hem = only(filter(r -> r.res_name == "HEM", residues))
    @test hem.modality == LIGAND  # multi-atom hetero, not in PTM list
    @test hem.elements["FE"] == :Fe

    tokens = tokenize_structure(residues)
    @test all(t -> t.res_name != "HOH", tokens)
    modalities = Set(t.modality for t in tokens)
    @test modalities == Set([PROTEIN, ION, LIGAND])

    ala_tokens = filter(t -> t.res_name == "ALA", tokens)
    @test length(ala_tokens) == length(AtomVocab.atom_slots("ALA"))
    missing_atoms = filter(t -> t.is_virtual, ala_tokens)
    @test Set(t.atom_name for t in missing_atoms) ==
          Set(setdiff(AtomVocab.atom_slots("ALA"), ["N", "CA", "C", "O", "CB"]))
end
