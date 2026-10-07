# Small full-network forward/gradient reference, independent of corpus loading.
using DiffuseBioMol, Random, JSON, Zygote, LinearAlgebra

pack(x) = Dict("shape" => collect(size(x)), "values" => vec(x))

function export_python_parity(path)
    BLAS.set_num_threads(2)
    rng = MersenneTwister(21)
    cfg = ModelConfig(d_single=16, d_pair=8, n_heads=4, n_pairformer_layers=1,
        n_dit_layers=1, d_time=16, d_hidden_mult=2)
    model = build_model(cfg)
    ps, st = DiffuseBioMol.Model.Network.Lux.setup(rng, model)
    # Nonzero head exercises the entire network and gradient path.
    ps.head.weight .= 0.1f0 .* randn(rng, Float32, size(ps.head.weight))
    F = DiffuseBioMol.Model.Features
    n, b = 7, 2
    elem = rand(rng, 1:F.N_ELEMENTS, n, b)
    mod = rand(rng, 1:F.N_MODALITIES, n, b)
    poly = rand(rng, 1:F.N_POLYMER_ATOM_TYPES, n, b)
    rp = rand(rng, 1:F.N_RELPOS_BUCKETS, n, n, b)
    x = randn(rng, Float32, 3, n, b)
    t = rand(rng, Float32, b)
    cond = randn(rng, Float32, 4, n, b)
    pad = zeros(Float32, n, n, b)
    input = (elem, mod, poly, rp, x, t, cond, pad)
    output, _ = model(input, ps, st)
    loss(p) = sum(abs2, first(model(input, p, st))) / length(output)
    grad = Zygote.gradient(loss, ps)[1]
    weights = Dict{String,Any}()
    function dense(prefix, layer)
        weights[prefix * ".weight"] = pack(layer.weight)
        weights[prefix * ".bias"] = pack(vec(layer.bias))
    end
    function norm(prefix, layer)
        weights[prefix * ".weight"] = pack(vec(layer.scale))
        weights[prefix * ".bias"] = pack(vec(layer.bias))
    end
    for (dest, src) in (("element", :elem_emb), ("modality", :mod_emb),
                        ("polymer", :poly_emb), ("relpos", :relpos_emb))
        weights[dest * ".weight"] = pack(permutedims(getproperty(ps, src).weight))
    end
    for (dest, src) in (("condition", :cond_in), ("coord", :coord_in),
                        ("time_mlp.0", :time_mlp1), ("time_mlp.2", :time_mlp2), ("head", :head))
        dense(dest, getproperty(ps, src))
    end
    for (prefix, layer) in (("encoder.0", ps.encoder.layer_1), ("decoder.0", ps.decoder.layer_1))
        for (dest, src) in (("attn.q", :wq), ("attn.k", :wk), ("attn.v", :wv),
                           ("attn.out", :wo), ("attn.bias", :pair_bias), ("mlp.0", :mlp1), ("mlp.2", :mlp2))
            dense(prefix * "." * dest, getproperty(layer, src))
        end
    end
    enc = ps.encoder.layer_1
    for (dest, src) in (("a", :pair_a), ("b", :pair_b), ("mix", :pair_mix))
        dense("encoder.0." * dest, getproperty(enc, src))
    end
    for (dest, src) in (("norm1", :ln_attn), ("norm2", :ln_mlp), ("pair_norm", :ln_pair))
        norm("encoder.0." * dest, getproperty(enc, src))
    end
    dense("decoder.0.ada1", ps.decoder.layer_1.ada1)
    dense("decoder.0.ada2", ps.decoder.layer_1.ada2)
    batch = Dict("element" => pack(permutedims(elem .- 1)), "modality" => pack(permutedims(mod .- 1)),
        "polymer" => pack(permutedims(poly .- 1)), "relpos" => pack(permutedims(rp .- 1, (3,1,2))))
    reference = Dict("weights" => weights, "batch" => batch,
        "vocab" => [F.N_ELEMENTS,F.N_MODALITIES,F.N_POLYMER_ATOM_TYPES,F.N_RELPOS_BUCKETS],
        "x" => pack(permutedims(x, (3,2,1))), "t" => pack(t),
        "cond" => pack(permutedims(cond, (3,2,1))),
        "output" => pack(permutedims(output, (3,2,1))), "loss" => loss(ps),
        "grad_coord" => pack(grad.coord_in.weight), "grad_head" => pack(grad.head.weight))
    mkpath(dirname(abspath(path)))
    write(path, JSON.json(reference))
    println("Wrote full-network parity reference to $path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("usage: export_python_parity.jl OUTPUT.json")
    export_python_parity(only(ARGS))
end
