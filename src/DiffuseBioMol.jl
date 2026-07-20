"""
DiffuseBioMol.jl

A Julia-native, all-atom biomolecular diffusion/flow model combining
RFdiffusion3's conditioning/design flexibility with NeuralPLexer3's any-to-any
co-folding generality, with physical correctness as an in-loop signal and a
closed-loop generate->verify->curate->retrain self-improvement controller.
See the project plan for the full design and phased roadmap.
"""
module DiffuseBioMol

include("Tokenization/AtomVocab.jl")
include("Tokenization/Tokenizer.jl")
include("Data/Data.jl")
include("Model/Model.jl")
include("Sampling/Sampling.jl")
include("Verification/Verification.jl")
include("AgenticLoop/AgenticLoop.jl")
include("Distributed/Distributed.jl")

using .AtomVocab
using .Tokenizer
using .Data
using .Model
using .Sampling
using .Verification

export AtomVocab, Tokenizer, Data, Model, Sampling, Verification, AgenticLoop, Distributed
export Modality, PROTEIN, RNA, DNA, LIGAND, ION, PTM, atom_slots
export AtomToken, ParsedResidue, tokenize_residue, tokenize_structure
export parse_structure, parse_structure_string, fetch_pdb, restrict_to_chain, from_atom_array
export list_structure_files, largest_chain
export TokenFeatures, featurize, relpos_buckets, target_coordinates
export N_ELEMENTS, N_MODALITIES, N_POLYMER_ATOM_TYPES, N_RELPOS_BUCKETS
export AtomConstraints, no_constraints, constraint_features, N_COND_FEATURES
export ChainCoMConstraint, apply_com_guidance!
export BatchedFeatures, batch_features, batch_relpos, batch_coords, batch_cond_features, attention_pad_bias, to_device
export ModelConfig, build_model
export random_rotation, random_se3_transform, apply_se3_transform, random_se3_augment
export PriorConfig, sample_prior
export TrainingExample, prepare_training_example, cfm_loss, sample_flow
export BatchedTrainingExample
export VDW_RADII, BACKBONE_BOND_LENGTHS, clash_energy, bond_energy, validity_energy, lddt, clash_count, bond_length_rmsd
export chiral_volume, normalized_chiral_volume, chirality_energy, chirality_count
export kabsch_align, aligned_rmsd
export VerifierConfig, build_verifier
export backbone_bonds, chiral_centers, validity_guidance_step, verifier_loss

end # module DiffuseBioMol
