# Export existing Julia tokens to a portable, linear-size JSON corpus.
# Keeps the exact categorical vocabulary; no Python reimplementation of parsing.
using DiffuseBioMol, JSON, SHA

function export_python_corpus(input, output)
    ispath(output) && error("Choose a new output directory: $output")
    files = list_structure_files(input; recursive=true)
    isempty(files) && error("No coordinate files in $input")
    mkpath(output)
    F = DiffuseBioMol.Model.Features
    entries = []
    skipped = []
    for (i, path) in enumerate(files)
        try
            residues = parse_structure(path)
            tokens = tokenize_structure(restrict_to_chain(residues, largest_chain(residues)))
            isempty(tokens) && error("Empty chain")
            feat = featurize(tokens)
            any(.!feat.is_virtual) || error("No observed atoms")
            name = "source_$(lpad(i, 8, '0')).json"
            record = Dict("element" => feat.element_idx .- 1,
                "modality" => feat.modality_idx .- 1, "polymer" => feat.polymer_atom_idx .- 1,
                "chain" => feat.chain_idx, "residue" => feat.res_index,
                "virtual" => feat.is_virtual,
                "xyz" => [collect(t.coord === nothing ? (0., 0., 0.) : t.coord) for t in tokens])
            payload = JSON.json(record)
            write(joinpath(output, name), payload)
            push!(entries, Dict("file" => name, "source" => abspath(path),
                "label" => splitext(basename(path))[1], "atoms" => length(tokens),
                "sha256" => bytes2hex(sha256(payload))))
        catch err
            push!(skipped, Dict("source" => abspath(path), "error" => sprint(showerror, err)))
        end
    end
    isempty(entries) && error("No usable structures; see input files")
    vocab = sort([(index=i-1, residue=r, atom=a) for ((r,a),i) in F.POLYMER_ATOM_VOCAB]; by=x->x.index)
    manifest = Dict("schema_version" => 1, "index_base" => 0,
        "vocab_sizes" => [F.N_ELEMENTS, F.N_MODALITIES, F.N_POLYMER_ATOM_TYPES, F.N_RELPOS_BUCKETS],
        "polymer_vocabulary" => vocab, "entries" => entries, "skipped" => skipped)
    write(joinpath(output, "manifest.json"), JSON.json(manifest))
    println("Exported $(length(entries)) sources; $(length(skipped)) skipped to $output")
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("usage: export_python_corpus.jl INPUT_DIR NEW_OUTPUT_DIR")
    export_python_corpus(ARGS...)
end
