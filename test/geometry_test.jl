using Test
using DiffuseBioMol
using Zygote
using Random, LinearAlgebra

@testset "Geometry.clash_energy: hand-computed values" begin
    # Two carbons 2.0 A apart, different chains -> not excluded.
    # threshold = vdw(C) + vdw(C) - tol = 1.70 + 1.70 - 0.4 = 3.0
    # violation = 3.0 - 2.0 = 1.0 -> energy = 1.0
    coords = Float64[0.0 2.0; 0.0 0.0; 0.0 0.0]
    e = clash_energy(coords, [:C, :C], [1, 2], [1, 1]; tol=0.4)
    @test e ≈ 1.0

    # Same pair, but far enough apart that there's no violation.
    coords_far = Float64[0.0 5.0; 0.0 0.0; 0.0 0.0]
    @test clash_energy(coords_far, [:C, :C], [1, 2], [1, 1]; tol=0.4) ≈ 0.0

    # Same coordinates, but same chain & adjacent residues -> excluded entirely.
    @test clash_energy(coords, [:C, :C], [1, 1], [1, 2]; tol=0.4) ≈ 0.0

    # Same chain, residues 2 apart -> not excluded, same violation as the first case.
    e2 = clash_energy(coords, [:C, :C], [1, 1], [1, 3]; tol=0.4)
    @test e2 ≈ 1.0

    # Three atoms: only the close pair should contribute.
    coords3 = Float64[0.0 2.0 100.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
    e3 = clash_energy(coords3, [:C, :C, :C], [1, 2, 3], [1, 1, 1]; tol=0.4)
    @test e3 ≈ 1.0  # the atom at 100 A is far from both others, contributes 0
end

@testset "Geometry.bond_energy: hand-computed values" begin
    coords = Float64[0.0 1.0; 0.0 0.0; 0.0 0.0]
    # (1.0 - 1.5)^2 = 0.25
    @test bond_energy(coords, [(1, 2, 1.5)]) ≈ 0.25
    @test bond_energy(coords, [(1, 2, 1.0)]) ≈ 0.0

    coords3 = Float64[0.0 1.0 3.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
    # bond (1,2): dist=1, ideal=1.5 -> 0.25 ; bond (2,3): dist=2, ideal=1.5 -> 0.25
    @test bond_energy(coords3, [(1, 2, 1.5), (2, 3, 1.5)]) ≈ 0.5
end

@testset "Geometry.clash_count: hand-computed values" begin
    coords = Float64[0.0 2.0 100.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
    @test clash_count(coords, [:C, :C, :C], [1, 2, 3], [1, 1, 1]) == 1  # only the (1,2) pair violates
    @test clash_count(coords, [:C, :C, :C], [1, 1, 1], [1, 2, 3]) == 0  # all same-chain, all adjacent-or-self -> excluded
end

@testset "Geometry.bond_length_rmsd: hand-computed values" begin
    coords3 = Float64[0.0 1.0 3.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
    # bond_energy = 0.5 (see above), 2 bonds -> rmsd = sqrt(0.5/2) = 0.5
    @test bond_length_rmsd(coords3, [(1, 2, 1.5), (2, 3, 1.5)]) ≈ 0.5
    @test bond_length_rmsd(coords3, Tuple{Int,Int,Float64}[]) == 0.0
end

@testset "Geometry.validity_energy: combines clash + bond with weights" begin
    coords = Float64[0.0 2.0; 0.0 0.0; 0.0 0.0]
    elements = [:C, :C]
    chain_idx = [1, 2]
    res_index = [1, 1]
    bonds = Tuple{Int,Int,Float64}[]  # no bonds -> pure clash term
    @test validity_energy(coords, elements, chain_idx, res_index, bonds) ≈ 1.0
    @test validity_energy(coords, elements, chain_idx, res_index, bonds; clash_weight=2.0) ≈ 2.0
    @test validity_energy(coords, elements, chain_idx, res_index, bonds; clash_weight=0.0) ≈ 0.0
end

@testset "Geometry.validity_energy is differentiable and gradient points the right way" begin
    coords = Float64[0.0 2.0; 0.0 0.0; 0.0 0.0]
    elements = [:C, :C]
    chain_idx = [1, 2]
    res_index = [1, 1]
    bonds = Tuple{Int,Int,Float64}[]
    grad = only(Zygote.gradient(c -> validity_energy(c, elements, chain_idx, res_index, bonds), coords))
    @test all(isfinite, grad)
    # Pushing atom 1 in -x and atom 2 in +x increases their separation and
    # should decrease the clash energy -> gradient w.r.t. atom 1's x should be
    # positive (energy increases as atom 1 moves toward atom 2, i.e. +x).
    @test grad[1, 1] > 0
    @test grad[1, 2] < 0
end

@testset "Geometry.lddt: hand-computed values" begin
    # Identical structures: perfect score everywhere.
    x = Float64[0.0 1.0 2.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
    @test lddt(x, x) ≈ [1.0, 1.0, 1.0]

    # Two atoms, reference distance 1.0; model distance 1.0 + delta.
    # delta=0.3 -> within thresholds {0.5,1,2,4} all four preserved -> score 1.0 each
    ref = Float64[0.0 1.0; 0.0 0.0; 0.0 0.0]
    model_small_shift = Float64[0.0 1.3; 0.0 0.0; 0.0 0.0]
    @test lddt(model_small_shift, ref) ≈ [1.0, 1.0]

    # delta = 0.7 -> misses the 0.5 threshold only -> preserved 3/4 = 0.75 per atom
    model_mid_shift = Float64[0.0 1.7; 0.0 0.0; 0.0 0.0]
    @test lddt(model_mid_shift, ref) ≈ [0.75, 0.75]

    # delta = 10 -> misses all four thresholds -> score 0.0 per atom (each has 1 neighbor)
    model_big_shift = Float64[0.0 11.0; 0.0 0.0; 0.0 0.0]
    @test lddt(model_big_shift, ref) ≈ [0.0, 0.0]

    # An isolated atom (no neighbors within cutoff) scores 1.0 vacuously.
    ref_isolated = Float64[0.0 1.0 1000.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
    model_isolated = Float64[0.0 1.0 1000.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
    scores = lddt(model_isolated, ref_isolated; cutoff=15.0)
    @test scores[3] == 1.0
end

@testset "Geometry.aligned_rmsd: zero for a rotated+translated copy of the same shape" begin
    rng = Random.Xoshiro(99)
    reference = Float64[0.0 1.5 2.9 5.0; 0.0 0.0 0.4 1.0; 0.0 0.0 0.0 -0.3]

    R, centroid = random_se3_transform(Float32.(reference), trues(4), rng)
    rotated = apply_se3_transform(Float32.(reference), R, centroid) .+ Float32[100.0, -50.0, 7.0]  # extra arbitrary shift

    @test aligned_rmsd(rotated, reference) < 1e-3
    # A raw, un-superposed difference would be large for this same pair.
    @test sqrt(sum(abs2, rotated .- reference) / size(reference, 2)) > 1.0
end

@testset "Geometry.aligned_rmsd: nonzero for genuinely different shapes" begin
    reference = Float64[0.0 1.5 2.9 5.0; 0.0 0.0 0.4 1.0; 0.0 0.0 0.0 -0.3]
    perturbed = copy(reference)
    perturbed[:, 2] .+= [2.0, 0.0, 0.0]  # move one atom -> genuine shape change

    @test aligned_rmsd(perturbed, reference) > 0.5
end
