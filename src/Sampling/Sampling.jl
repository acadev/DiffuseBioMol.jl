"""
Phase 1, 3, 4: the physics-informed prior (`Prior`) and the conditional
flow-matching loss/sampler (`FlowMatching`) built on top of
`Model.Network.build_model`. Differentiable in-loop physical-validity guidance
(clash/bond-geometry/chirality terms added to the velocity field, Boltz-2
steering-potential style) and the consistency-distilled fast sampler are not
yet implemented — see `docs/PLAN.md` Phases 3-4.
"""
module Sampling

include("Prior.jl")
include("FlowMatching.jl")

using .Prior
using .FlowMatching

export Prior, FlowMatching
export PriorConfig, sample_prior
export TrainingExample, prepare_training_example, cfm_loss, sample_flow
export BatchedTrainingExample

end # module
