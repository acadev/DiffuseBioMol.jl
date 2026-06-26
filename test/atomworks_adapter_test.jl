using Test
using DiffuseBioMol

# These tests build synthetic data shaped exactly like what `pyconvert`-ing a
# real Biotite `AtomArray`'s annotation arrays would produce (see
# `from_atom_array`'s docstring for the real PythonCall.jl recipe) — they
# verify the conversion logic itself without requiring Python/biotite/
# atomworks to be installed, which this sandbox does not guarantee.

@testset "from_atom_array: single protein residue (ALA)" begin
    chain_id = ["A", "A", "A", "A", "A"]
    res_id = [1, 1, 1, 1, 1]
    res_name = ["ALA", "ALA", "ALA", "ALA", "ALA"]
    atom_name = ["N", "CA", "C", "O", "CB"]
    element = ["N", "C", "C", "O", "C"]
    coord = [0.0 0.0 0.0; 1.5 0.0 0.0; 2.9 0.4 0.0; 3.4 1.5 0.0; 1.9 -0.7 1.2]  # N x 3
    hetero = [false, false, false, false, false]

    residues = from_atom_array(chain_id, res_id, res_name, atom_name, element, coord, hetero)
    @test length(residues) == 1
    r = only(residues)
    @test r.chain_id == "A"
    @test r.res_index == 1
    @test r.res_name == "ALA"
    @test r.modality == PROTEIN
    @test r.present_atoms["CA"] == (1.5, 0.0, 0.0)
    @test r.elements["CA"] == :C
end

@testset "from_atom_array: ligand (multi-atom hetero) and ion (single-atom hetero)" begin
    chain_id = ["L", "L", "I"]
    res_id = [1, 1, 1]
    res_name = ["HEM", "HEM", "ZN"]
    atom_name = ["FE", "N1", "ZN"]
    element = ["FE", "N", "ZN"]
    coord = [0.0 0.0 0.0; 1.0 0.0 0.0; 10.0 10.0 10.0]
    hetero = [true, true, true]

    residues = from_atom_array(chain_id, res_id, res_name, atom_name, element, coord, hetero)
    @test length(residues) == 2

    hem = only(filter(r -> r.res_name == "HEM", residues))
    @test hem.modality == LIGAND
    @test hem.elements["FE"] == :Fe

    zn = only(filter(r -> r.res_name == "ZN", residues))
    @test zn.modality == ION
    @test zn.elements["ZN"] == :Zn
end

@testset "from_atom_array: water is dropped" begin
    chain_id = ["W", "W"]
    res_id = [1, 2]
    res_name = ["HOH", "HOH"]
    atom_name = ["O", "O"]
    element = ["O", "O"]
    coord = [0.0 0.0 0.0; 5.0 5.0 5.0]
    hetero = [true, true]

    residues = from_atom_array(chain_id, res_id, res_name, atom_name, element, coord, hetero)
    @test isempty(residues)
end

@testset "from_atom_array: multi-residue contiguous grouping within a chain" begin
    chain_id = ["A", "A", "A", "A", "A", "A", "A", "A"]
    res_id = [1, 1, 1, 1, 2, 2, 2, 2]
    res_name = ["ALA", "ALA", "ALA", "ALA", "GLY", "GLY", "GLY", "GLY"]
    atom_name = ["N", "CA", "C", "O", "N", "CA", "C", "O"]
    element = ["N", "C", "C", "O", "N", "C", "C", "O"]
    coord = zeros(8, 3)
    coord[:, 1] .= 0:7  # distinct x-coordinates so we can check ordering
    hetero = falses(8)

    residues = from_atom_array(chain_id, res_id, res_name, atom_name, element, coord, hetero)
    @test length(residues) == 2
    @test residues[1].res_name == "ALA" && residues[1].res_index == 1
    @test residues[2].res_name == "GLY" && residues[2].res_index == 2
    @test residues[1].present_atoms["N"] == (0.0, 0.0, 0.0)
    @test residues[2].present_atoms["N"] == (4.0, 0.0, 0.0)
end

@testset "from_atom_array: rejects a 3 x N coord matrix (must be N x 3)" begin
    chain_id, res_id, res_name, atom_name, element, hetero = ["A"], [1], ["ALA"], ["CA"], ["C"], [false]
    bad_coord = zeros(3, 1)  # wrong orientation
    @test_throws ArgumentError from_atom_array(chain_id, res_id, res_name, atom_name, element, bad_coord, hetero)
end

@testset "from_atom_array: end-to-end through tokenize_structure" begin
    chain_id = fill("A", 4)
    res_id = fill(1, 4)
    res_name = fill("GLY", 4)
    atom_name = ["N", "CA", "C", "O"]
    element = ["N", "C", "C", "O"]
    coord = [0.0 0.0 0.0; 1.0 0.0 0.0; 2.0 0.0 0.0; 3.0 0.0 0.0]
    hetero = falses(4)

    residues = from_atom_array(chain_id, res_id, res_name, atom_name, element, coord, hetero)
    tokens = tokenize_structure(residues)
    @test length(tokens) == 4  # GLY has no sidechain slots, no padding
    @test all(t -> !t.is_virtual, tokens)
    @test all(t -> t.modality == PROTEIN, tokens)
end
