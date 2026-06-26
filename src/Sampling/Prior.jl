"""
Physics-informed prior for flow matching, following NeuralPLexer3's choice of a
globular-polymer/Langevin prior over pure Gaussian noise: consecutive atoms
within the same chain are harmonically connected (a 3D random walk with a
spring restoring the expected bond length), while atoms with no polymer
neighbor (ligands, ions, chain-terminal atoms) fall back to independent
Gaussian placement. This gives the flow's `t=0` endpoint roughly chain-like
geometry instead of an unstructured point cloud, which is the documented
reason NeuralPLexer3 needs far fewer integration steps than plain Gaussian
diffusion.
"""
module Prior

using Random

export sample_prior, PriorConfig

"""
    PriorConfig(; bond_length=3.8, bond_std=1.0, free_std=10.0)

- `bond_length`: target distance (Å) between harmonically-connected consecutive
  atoms (≈ Cα-Cα spacing; a coarse default, not atom-specific).
- `bond_std`: standard deviation of the harmonic bond-length fluctuation.
- `free_std`: standard deviation used to place the first atom of each chain and
  any atom with no polymer predecessor (ligands, ions), centered at the origin.
"""
struct PriorConfig
    bond_length::Float64
    bond_std::Float64
    free_std::Float64
end

PriorConfig(; bond_length=3.8, bond_std=1.0, free_std=10.0) = PriorConfig(bond_length, bond_std, free_std)

"""
    sample_prior(rng, chain_ids; config=PriorConfig()) -> Matrix{Float64}  # 3 x N

Draw one prior sample for `N = length(chain_ids)` atoms, where `chain_ids[i]`
identifies which polymer chain atom `i` belongs to (atoms are assumed given in
chain order; use any comparable/hashable id, e.g. a `String` or `Int` — atoms
with a unique singleton id, such as separate ligand atoms tokenized
individually, behave like a chain of length 1 and are placed freely).
"""
function sample_prior(rng::AbstractRNG, chain_ids::AbstractVector; config::PriorConfig=PriorConfig())
    n = length(chain_ids)
    coords = Matrix{Float64}(undef, 3, n)
    for i in 1:n
        same_chain_as_prev = i > 1 && chain_ids[i] == chain_ids[i-1]
        if same_chain_as_prev
            direction = randn(rng, 3)
            direction ./= norm3(direction)
            r = config.bond_length + config.bond_std * randn(rng)
            coords[:, i] = coords[:, i-1] .+ r .* direction
        else
            coords[:, i] = config.free_std .* randn(rng, 3)
        end
    end
    coords
end

"""
    sample_prior(rng, chain_idx::AbstractMatrix; config=PriorConfig()) -> Array{Float64,3}  # 3 x N x B

Batched counterpart: `chain_idx` is `N x B` (e.g. `Model.Batching.BatchedFeatures.chain_idx`),
one prior sample per column, stacked along a trailing batch dimension. A
plain Julia loop over `b` is fine here — this function is never called
inside a `Zygote.pullback` closure (`x0` doesn't depend on the model's
parameters; see `FlowMatching.prepare_training_example`), so there's no AD
performance concern, unlike the model-forward code where a loop over a
batch dimension would be the wrong pattern.
"""
function sample_prior(rng::AbstractRNG, chain_idx::AbstractMatrix; config::PriorConfig=PriorConfig())
    n, b_size = size(chain_idx)
    out = Array{Float64,3}(undef, 3, n, b_size)
    for b in 1:b_size
        out[:, :, b] .= sample_prior(rng, view(chain_idx, :, b); config=config)
    end
    out
end

norm3(v) = sqrt(v[1]^2 + v[2]^2 + v[3]^2)

end # module
