"""
Phase 1, 3, 4: the physics-informed prior (`Prior`) and the conditional
flow-matching loss/sampler (`FlowMatching`) built on top of
`Model.Network.build_model`. Differentiable in-loop physical-validity guidance
(clash/bond-geometry/chirality terms added to the velocity field, Boltz-2
steering-potential style) and the consistency-distilled fast sampler are not
yet implemented — see `docs/PLAN.md` Phases 3-4.

`Augmentation` is the random SE(3) (centering + random rotation) training
augmentation that compensates for `Model.Network` having no built-in
rotation/translation equivariance — see its module docstring, and
`FlowMatching.prepare_training_example`'s use of it, for why this isn't
optional.
"""
module Sampling

include("Augmentation.jl")
include("Prior.jl")
include("FlowMatching.jl")

using .Augmentation
using .Prior
using .FlowMatching

export Augmentation, Prior, FlowMatching
export random_rotation, random_se3_transform, apply_se3_transform, random_se3_augment
export PriorConfig, sample_prior
export TrainingExample, prepare_training_example, cfm_loss, sample_flow
export BatchedTrainingExample

end # module
