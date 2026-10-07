# Regression tests: hardcoded golden values for deterministic, dependency-light
# components, distinct from geometry_test.jl/verification_test.jl's unit
# tests (which check logical correctness against hand-derived expectations for
# varied scenarios). These pin down *current* exact behavior so an accidental
# change to a constant table, a vocabulary, or an index-bucketing formula is
# caught even if no individual logical property is violated.
#
# Deliberately scoped to pure, order-independent functions (AtomVocab
# constants, Features.relpos_buckets, tokenization counts) rather than
# neural-network outputs: pinning exact floating-point weights/activations from
# a Lux model is brittle against routine dependency upgrades that don't change
# the model's *logic* (see model_smoke_test.jl/verification_test.jl for the
# "trains down by the expected ratio" style of regression check that's
# appropriate for those components instead).
using Test
using DiffuseBioMol

@testset "Regression: AtomVocab atom14-style layouts are exactly as defined" begin
    @test atom_slots("ALA") == ["N", "CA", "C", "O", "CB"]
    @test atom_slots("GLY") == ["N", "CA", "C", "O"]
    @test atom_slots("TRP") == ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"]
    @test atom_slots("A") == ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "O2'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"]
    @test atom_slots("DA") == ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"]
    @test DiffuseBioMol.AtomVocab.MAX_ATOMS_PER_RESIDUE == 23
    @test length(DiffuseBioMol.AtomVocab.PROTEIN_RESIDUES) == 20
    @test length(DiffuseBioMol.AtomVocab.RNA_RESIDUES) == 4
    @test length(DiffuseBioMol.AtomVocab.DNA_RESIDUES) == 4
end

@testset "Regression: Features vocabulary sizes" begin
    @test N_ELEMENTS == 17
    @test N_MODALITIES == 6
    @test N_POLYMER_ATOM_TYPES == 335
    @test N_RELPOS_BUCKETS == 66
    @test N_COND_FEATURES == 4
end

@testset "Regression: relpos_buckets exact bucketing formula" begin
    res1 = ParsedResidue("A", 1, "ALA", PROTEIN, Dict("N" => (0.0, 0.0, 0.0)))
    res2 = ParsedResidue("A", 3, "ALA", PROTEIN, Dict("N" => (0.0, 0.0, 0.0)))
    res3 = ParsedResidue("B", 1, "ALA", PROTEIN, Dict("N" => (0.0, 0.0, 0.0)))
    tokens = filter(t -> t.atom_name == "N", tokenize_structure([res1, res2, res3]))
    feat = featurize(tokens)
    @test feat.chain_idx == [1, 1, 2]
    @test feat.res_index == [1, 3, 1]

    buckets = relpos_buckets(feat)
    @test buckets == [33 31 66; 35 33 66; 66 66 33]

    # Clamping: a same-chain pair more than RELPOS_CLAMP (32) residues apart
    # saturates at the same bucket as exactly-32-apart.
    res_far = ParsedResidue("A", 1000, "ALA", PROTEIN, Dict("N" => (0.0, 0.0, 0.0)))
    res_origin = ParsedResidue("A", 1, "ALA", PROTEIN, Dict("N" => (0.0, 0.0, 0.0)))
    tokens2 = filter(t -> t.atom_name == "N", tokenize_structure([res_origin, res_far]))
    feat2 = featurize(tokens2)
    buckets2 = relpos_buckets(feat2)
    @test buckets2[1, 2] == 1   # clamp(1 - 1000, -32, 32) + 33 = -32 + 33 = 1
    @test buckets2[2, 1] == 65  # clamp(1000 - 1, -32, 32) + 33 = 32 + 33 = 65
end

@testset "Regression: tokenize_structure atom counts and virtual-atom pattern" begin
    # A residue with crystallographically missing sidechain density (SER:
    # only backbone observed) mixed with a full ligand and a bare ion.
    ser = ParsedResidue("A", 1, "SER", PROTEIN, Dict("N" => (0.0, 0.0, 0.0), "CA" => (1.0, 0.0, 0.0), "C" => (2.0, 0.0, 0.0), "O" => (3.0, 0.0, 0.0)))
    hem = ParsedResidue("L", 1, "HEM", LIGAND, Dict("FE" => (0.0, 0.0, 0.0), "N1" => (1.0, 0.0, 0.0)), Dict("FE" => :Fe))
    zn = ParsedResidue("I", 1, "ZN", ION, Dict("ZN" => (0.0, 0.0, 0.0)), Dict("ZN" => :Zn))

    tokens = tokenize_structure([ser, hem, zn])
    @test length(tokens) == 6 + 2 + 1  # SER atom14 layout (N,CA,C,O,CB,OG = 6) + HEM (2) + ZN (1)

    ser_tokens = filter(t -> t.res_name == "SER", tokens)
    @test length(ser_tokens) == 6
    @test count(t -> t.is_virtual, ser_tokens) == 2  # CB and OG (the sidechain) are missing
    @test Set(t.atom_name for t in filter(t -> t.is_virtual, ser_tokens)) == Set(["CB", "OG"])

    @test all(t -> !t.is_virtual, filter(t -> t.res_name in ("HEM", "ZN"), tokens))  # non-polymer atoms are never "virtual"

    feat = featurize(tokens)
    @test count(feat.is_virtual) == 2
end

@testset "Regression: Geometry golden combined scenario" begin
    # A fixed 3-atom configuration exercising clash + bond + lddt together,
    # numbers computed once (and cross-checked by hand) and frozen.
    coords = Float64[0.0 1.458 4.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
    elements = [:N, :C, :O]
    chain_idx = [1, 1, 2]
    res_index = [1, 1, 1]
    bonds = [(1, 2, 1.458)]  # N-CA-like bond, exactly at the ideal length

    @test bond_energy(coords, bonds) ≈ 0.0
    # clash: pair (1,3) and (2,3) cross chains, not excluded; pair (1,2) is
    # same-chain adjacent-residue (res diff 0) -> excluded.
    # (1,3): dist=4.0, threshold=vdw(N)+vdw(O)-0.4=1.55+1.52-0.4=2.67 -> no violation
    # (2,3): dist=2.542, threshold=vdw(C)+vdw(O)-0.4=1.70+1.52-0.4=2.82 -> violation=0.278 -> 0.077284
    @test clash_energy(coords, elements, chain_idx, res_index) ≈ 0.077284 atol = 1e-6
    @test validity_energy(coords, elements, chain_idx, res_index, bonds) ≈ 0.077284 atol = 1e-6

    ref = coords
    model_shifted = coords .+ [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 3.0]  # move atom 3 by 3 A in z
    scores = lddt(model_shifted, ref)
    @test scores ≈ [0.875, 0.75, 0.625]
end
