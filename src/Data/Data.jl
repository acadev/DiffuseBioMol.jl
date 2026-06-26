"""
Structure-file ingestion: parse PDB/mmCIF files (via BioStructures.jl, which
already handles the format-level details well — no reason to hand-roll a PDB
parser) into `Tokenizer.ParsedResidue` records, classifying each residue into
the `AtomVocab.Modality` the rest of the pipeline expects (protein/RNA/DNA from
standard polymer residue names, ion from single-atom heteroatom residues, PTM
from a known modified-residue vocabulary, ligand as the catch-all for
everything else). Water is dropped.

Also provides `from_atom_array`, an adapter for AtomWorks
(github.com/RosettaCommons/atomworks, the data-loading/featurization layer
RFdiffusion3/foundry trains against) — see that function's docstring for the
real-interop recipe and exactly what compatibility means here, since
AtomWorks is a Python package and this is a Julia codebase.

Dataset curation (PDB + AlphaFold distillation + DFT-level ligand conformers,
mirroring RFD3/NeuralPLexer3's training curricula) and multi-node-aware data
loading for distributed training are not yet implemented.
"""
module Data

using BioStructures
using ..AtomVocab
using ..Tokenizer: ParsedResidue

export parse_structure, parse_structure_string, fetch_pdb, restrict_to_chain, from_atom_array

# Modified-residue (PTM) 3-letter codes commonly seen in the PDB as HETATM
# records that are still part of a polymer chain (phosphorylation, methylation,
# acetylation, selenomethionine, etc.). Not exhaustive — anything hetero,
# multi-atom, and not in this list falls back to LIGAND.
const PTM_RESIDUE_NAMES = Set([
    "SEP", "TPO", "PTR",   # phosphoserine/threonine/tyrosine
    "MSE",                 # selenomethionine
    "MLY", "M3L", "MLZ",   # methyllysine variants
    "ALY",                 # acetyllysine
    "CSO", "CSD",          # oxidized cysteine variants
    "HYP",                 # hydroxyproline
    "PCA",                 # pyroglutamate
])

"""
    classify_residue(resname, ishet, natoms) -> Modality

Decide which `Modality` a residue belongs to from its name, hetero flag, and
atom count, in the absence of a full Chemical Component Dictionary lookup.
"""
function classify_residue(resname::AbstractString, ishet::Bool, natoms::Int)::Modality
    haskey(AtomVocab.PROTEIN_RESIDUES, resname) && return PROTEIN
    haskey(AtomVocab.RNA_RESIDUES, resname) && return RNA
    haskey(AtomVocab.DNA_RESIDUES, resname) && return DNA
    !ishet && return PROTEIN  # non-hetero, non-standard polymer residue: treat as protein
    natoms == 1 && return ION
    resname in PTM_RESIDUE_NAMES && return PTM
    return LIGAND
end

function normalize_element(s::AbstractString)::Symbol
    isempty(s) && return :X
    length(s) == 1 && return Symbol(uppercase(s))
    Symbol(uppercase(s[1:1]) * lowercase(s[2:end]))
end

"""
    parse_structure(struc) -> Vector{ParsedResidue}

Convert an already-loaded `BioStructures.MolecularStructure` (or `Model`/
`Chain`) into `ParsedResidue` records, skipping water. Atom coordinates and
elements are taken directly from the parsed file (so e.g. crystallographically
unobserved sidechain atoms are simply absent from `present_atoms`, exactly the
input the tokenizer expects to pad).
"""
function parse_structure(struc)::Vector{ParsedResidue}
    residues = ParsedResidue[]
    for res in collectresidues(struc)
        rname = resname(res)
        rname in waterresnames && continue

        atoms_in_res = collectatoms(res)
        isempty(atoms_in_res) && continue

        present_atoms = Dict{String,NTuple{3,Float64}}()
        elements = Dict{String,Symbol}()
        for at in atoms_in_res
            name = atomname(at; strip=true)
            c = coords(at)
            present_atoms[name] = (Float64(c[1]), Float64(c[2]), Float64(c[3]))
            elements[name] = normalize_element(element(at; strip=true))
        end

        modality = classify_residue(rname, ishetero(res), length(atoms_in_res))
        push!(residues, ParsedResidue(chainid(res), resnumber(res), rname, modality, present_atoms, elements))
    end
    residues
end

"""
    parse_structure(path::AbstractString) -> Vector{ParsedResidue}

Read a structure file from disk. Format is inferred from the extension
(`.pdb`/`.ent` -> `PDBFormat`, `.cif`/`.mmcif` -> `MMCIFFormat`).
"""
function parse_structure(path::AbstractString)::Vector{ParsedResidue}
    fmt = lowercase(path)
    format = endswith(fmt, ".cif") || endswith(fmt, ".mmcif") ? MMCIFFormat : PDBFormat
    parse_structure(read(path, format))
end

"""
    parse_structure_string(content::AbstractString; format=PDBFormat) -> Vector{ParsedResidue}

Parse structure text held in memory (mainly for tests/fixtures) rather than
read from a file path.
"""
function parse_structure_string(content::AbstractString; format=PDBFormat)::Vector{ParsedResidue}
    parse_structure(read(IOBuffer(content), format))
end

"""
    fetch_pdb(pdb_id::AbstractString; dir=joinpath(tempdir(), "rfdiffusion_pdb_cache")) -> Vector{ParsedResidue}

Download a real structure from the RCSB PDB (via `BioStructures.retrievepdb`,
which caches to `dir` so repeated calls for the same ID don't re-download)
and parse it into `ParsedResidue` records — the real-data counterpart to
`parse_structure_string`'s synthetic fixtures, used for Phase 1's real-PDB
training harness (see `docs/PLAN.md`).
"""
function fetch_pdb(pdb_id::AbstractString; dir=joinpath(tempdir(), "rfdiffusion_pdb_cache"))::Vector{ParsedResidue}
    mkpath(dir)
    struc = retrievepdb(pdb_id; dir=dir)
    parse_structure(struc)
end

"""
    restrict_to_chain(residues, chain_id) -> Vector{ParsedResidue}

Keep only residues belonging to `chain_id`. Real PDB entries are often
multi-chain and/or carry crystallization additives across many short hetero
"chains"; restricting to one polymer chain keeps the O(N^2) pairwise
representation a tractable size for early training runs (see
`docs/PLAN.md`'s note on the encoder/decoder being unbatched across
structures of different `N` in v1).
"""
restrict_to_chain(residues::AbstractVector{ParsedResidue}, chain_id::AbstractString) =
    filter(r -> r.chain_id == chain_id, residues)

"""
    from_atom_array(chain_id, res_id, res_name, atom_name, element, coord, hetero) -> Vector{ParsedResidue}

Convert structure data shaped like Biotite's `AtomArray` — the representation
AtomWorks (`atomworks.io`/`atomworks.ml`, github.com/RosettaCommons/atomworks)
is built on, and the data layer RFdiffusion3/foundry trains against — into
`ParsedResidue` records. This is the actual "AtomWorks compatibility" this
package offers: not a Python<->Julia bridge (this codebase has no Python
dependency), but a Julia-side adapter for the *data shape* AtomWorks produces,
so an AtomWorks-curated dataset can be fed into this pipeline once its
`AtomArray` is converted to plain Julia vectors.

All seven arguments are length-`N` (one entry per atom), matching Biotite's
`AtomArray` annotation categories exactly:
- `chain_id::AbstractVector{<:AbstractString}`
- `res_id::AbstractVector{<:Integer}`
- `res_name::AbstractVector{<:AbstractString}`
- `atom_name::AbstractVector{<:AbstractString}`
- `element::AbstractVector{<:AbstractString}` (e.g. `"C"`, `"FE"` — case-insensitive)
- `coord::AbstractMatrix` — `N x 3` (Biotite's row-per-atom convention, *not*
  this package's internal `3 x N`; the conversion happens inside this function)
- `hetero::AbstractVector{Bool}`

Atoms are grouped into residues by contiguous runs of identical
`(chain_id, res_id, res_name)`, exactly matching how Biotite/AtomWorks lay out
an `AtomArray` (sorted, one contiguous block per residue) — this function does
*not* re-sort or group non-contiguous atoms, so the input must already be in
that order (as any real `AtomArray` is). Water (`HOH` and the other names in
`BioStructures.waterresnames`) is dropped, exactly as in `parse_structure`.

To bridge from a real Biotite `AtomArray` via PythonCall.jl (not exercised in
this codebase's test suite, since it requires `pip install atomworks` /
`biotite` in the Python environment PythonCall.jl resolves to):

```julia
using PythonCall
atom_array = ...  # however your AtomWorks pipeline produces one
residues = from_atom_array(
    pyconvert(Vector{String}, atom_array.chain_id),
    pyconvert(Vector{Int}, atom_array.res_id),
    pyconvert(Vector{String}, atom_array.res_name),
    pyconvert(Vector{String}, atom_array.atom_name),
    pyconvert(Vector{String}, atom_array.element),
    pyconvert(Matrix{Float64}, atom_array.coord),
    pyconvert(Vector{Bool}, atom_array.hetero),
)
```
"""
function from_atom_array(
    chain_id::AbstractVector{<:AbstractString},
    res_id::AbstractVector{<:Integer},
    res_name::AbstractVector{<:AbstractString},
    atom_name::AbstractVector{<:AbstractString},
    element::AbstractVector{<:AbstractString},
    coord::AbstractMatrix,
    hetero::AbstractVector{Bool},
)::Vector{ParsedResidue}
    n = length(chain_id)
    size(coord) == (n, 3) || throw(ArgumentError("coord must be N x 3 (Biotite's row-per-atom convention), got $(size(coord))"))
    residues = ParsedResidue[]

    i = 1
    while i <= n
        j = i
        while j < n && chain_id[j+1] == chain_id[i] && res_id[j+1] == res_id[i] && res_name[j+1] == res_name[i]
            j += 1
        end

        rname = res_name[i]
        if !(rname in waterresnames)
            present_atoms = Dict{String,NTuple{3,Float64}}()
            elements = Dict{String,Symbol}()
            for k in i:j
                present_atoms[atom_name[k]] = (Float64(coord[k, 1]), Float64(coord[k, 2]), Float64(coord[k, 3]))
                elements[atom_name[k]] = normalize_element(element[k])
            end
            modality = classify_residue(rname, hetero[i], j - i + 1)
            push!(residues, ParsedResidue(chain_id[i], Int(res_id[i]), rname, modality, present_atoms, elements))
        end

        i = j + 1
    end
    residues
end

end # module
