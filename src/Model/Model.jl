"""
Phase 1-3: the model package — token featurization (`Features`), the
Pairformer-lite + DiT-style backbone network (`Network`), and (later) the
generalized constraint-token conditioning system and jointly-trained
confidence/verifier head. Conditioning and verifier heads are not yet
implemented; see `docs/PLAN.md` Phases 2-3.
"""
module Model

using ..AtomVocab
using ..Tokenizer

include("Features.jl")
include("Conditioning.jl")
include("Batching.jl")
include("Network.jl")

using .Features
using .Conditioning
using .Batching
using .Network

export Features, Conditioning, Batching, Network
export TokenFeatures, featurize, relpos_buckets, target_coordinates
export N_ELEMENTS, N_MODALITIES, N_POLYMER_ATOM_TYPES, N_RELPOS_BUCKETS
export AtomConstraints, no_constraints, constraint_features, N_COND_FEATURES
export ChainCoMConstraint, apply_com_guidance!
export BatchedFeatures, batch_features, batch_relpos, batch_coords, batch_cond_features, attention_pad_bias, to_device
export ModelConfig, build_model

end # module
