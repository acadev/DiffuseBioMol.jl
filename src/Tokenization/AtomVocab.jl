"""
Atom vocabulary shared across all modalities (protein, RNA, DNA, ligand, ion, PTM).

This is the Phase 0 foundation described in the project plan: rather than RFdiffusion3's
split between a residue-level "atom14" scheme for proteins and a separate model for
nucleic acids, every modality is described with the same `AtomToken` type (see
`Tokenizer.jl`) and differs only in which `RESIDUE_ATOMS` entry (if any) it draws its
fixed atom-slot layout from.
"""
module AtomVocab

export Modality, PROTEIN, RNA, DNA, LIGAND, ION, PTM
export RESIDUE_ATOMS, PROTEIN_RESIDUES, RNA_RESIDUES, DNA_RESIDUES
export MAX_ATOMS_PER_RESIDUE, residue_modality, atom_slots

@enum Modality PROTEIN RNA DNA LIGAND ION PTM

# Polymer residues get a fixed, padded slot layout (cf. RFdiffusion3's atom14): a
# token always occupies the same number of slots regardless of how many atoms are
# actually present, with missing/inapplicable slots marked virtual at tokenization
# time. Ligands, ions, and PTMs have no canonical slot layout and are tokenized
# atom-by-atom instead (see `Tokenizer.jl`).

const BACKBONE_ATOMS = ("N", "CA", "C", "O")

const PROTEIN_RESIDUES = Dict{String,Vector{String}}(
    "ALA" => ["CB"],
    "ARG" => ["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN" => ["CB", "CG", "OD1", "ND2"],
    "ASP" => ["CB", "CG", "OD1", "OD2"],
    "CYS" => ["CB", "SG"],
    "GLN" => ["CB", "CG", "CD", "OE1", "NE2"],
    "GLU" => ["CB", "CG", "CD", "OE1", "OE2"],
    "GLY" => String[],
    "HIS" => ["CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE" => ["CB", "CG1", "CG2", "CD1"],
    "LEU" => ["CB", "CG", "CD1", "CD2"],
    "LYS" => ["CB", "CG", "CD", "CE", "NZ"],
    "MET" => ["CB", "CG", "SD", "CE"],
    "PHE" => ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO" => ["CB", "CG", "CD"],
    "SER" => ["CB", "OG"],
    "THR" => ["CB", "OG1", "CG2"],
    "TRP" => ["CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR" => ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL" => ["CB", "CG1", "CG2"],
)

const NUCLEOTIDE_BACKBONE = ("P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'")

const RNA_RESIDUES = Dict{String,Vector{String}}(
    "A" => ["O2'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
    "C" => ["O2'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
    "G" => ["O2'", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
    "U" => ["O2'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"],
)

const DNA_RESIDUES = Dict{String,Vector{String}}(
    "DA" => ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
    "DC" => ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
    "DG" => ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
    "DT" => ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C7", "C6"],
)

"""
    RESIDUE_ATOMS

Mapping from canonical residue name -> ordered slot layout (backbone-equivalent
followed by sidechain/base atoms), unified across the three polymer modalities.
Looking this up plus the modality is enough to build a fixed-size, padded token
sequence for any polymer residue.
"""
const RESIDUE_ATOMS = Dict{String,Vector{String}}(
    (name => vcat(collect(BACKBONE_ATOMS), atoms) for (name, atoms) in PROTEIN_RESIDUES)...,
    (name => vcat(collect(NUCLEOTIDE_BACKBONE), atoms) for (name, atoms) in RNA_RESIDUES)...,
    (name => vcat(collect(NUCLEOTIDE_BACKBONE), atoms) for (name, atoms) in DNA_RESIDUES)...,
)

const MAX_ATOMS_PER_RESIDUE = maximum(length(v) for v in values(RESIDUE_ATOMS))

"""
    residue_modality(res_name) -> Modality

Look up which polymer modality a canonical residue name belongs to. Throws for
non-polymer residue names (ligands/ions/PTMs are not in this table by design —
callers must tag those explicitly with their `Modality` since there is no
canonical name table for them).
"""
function residue_modality(res_name::AbstractString)
    haskey(PROTEIN_RESIDUES, res_name) && return PROTEIN
    haskey(RNA_RESIDUES, res_name) && return RNA
    haskey(DNA_RESIDUES, res_name) && return DNA
    throw(ArgumentError("Unknown polymer residue name: $res_name (ligand/ion/PTM residues must be tagged explicitly)"))
end

"""
    atom_slots(res_name) -> Vector{String}

Ordered, fixed atom-name slot layout for a canonical polymer residue name.
"""
atom_slots(res_name::AbstractString) = RESIDUE_ATOMS[res_name]

end # module
