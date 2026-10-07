"""
Shared data-loading helpers for `train_cuda.jl` and `preprocess_dataset.jl`:
turns a PDB ID (RCSB fetch) or a local PDB/mmCIF file path into one
preprocessed training example (`tokens`, `feat`, `relpos`, `x1`,
`cond_features`, `n_atoms`). Kept in its own file so `preprocess_dataset.jl`
doesn't need to pull in `train_cuda.jl`'s optional CUDA/WandB dependency
loading, and so the two scripts can't silently drift out of sync on what
"preprocessed" means.
"""

const CANDIDATE_TARGETS = [
    ("1CRN","A"),("1UBQ","A"),("5PTI","A"),("1L2Y","A"),("1VII","A"),("2GB1","A"),
    ("1ENH","A"),("1PGB","A"),("1SHG","A"),("2CI2","A"),("1IGD","A"),("1TEN","A"),
    ("2TRX","A"),("1AHO","A"),("1BTA","A"),("1CSP","A"),("1FNA","A"),("1MJC","A"),
    ("1PIN","A"),("1HHP","A"),("2PTL","A"),("1UBI","A"),("1COA","A"),("1FXD","A"),
]

function tokenize_example(label, residues)
    tokens = tokenize_structure(residues)
    isempty(tokens) && error("no tokens after parsing/filtering")
    feat = featurize(tokens)
    (pdb_id        = label,
     tokens        = tokens,
     feat          = feat,
     relpos        = relpos_buckets(feat),
     x1            = target_coordinates(tokens),
     cond_features = constraint_features(no_constraints(length(tokens))),
     n_atoms       = length(tokens))
end

function load_rcsb_example((pdb_id, chain_id))
    tokenize_example(pdb_id, restrict_to_chain(fetch_pdb(pdb_id), chain_id))
end

function load_local_example(path)
    res = parse_structure(path)
    isempty(res) && error("no residues parsed")
    ch = largest_chain(res)
    tokenize_example(splitext(basename(path))[1], restrict_to_chain(res, ch))
end
