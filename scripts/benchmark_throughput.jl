"""
Performance benchmarking harness — distinct from `scripts/benchmark_validity.jl`
(which measures *correctness*: clash count, bond RMSD). This one measures
*speed*: forward-only and forward+backward wall-clock, with first-call
(compile) time always reported separately from steady-state (the central
lesson from the Zygote compile-time investigation in `docs/PLAN.md` — these
two numbers tell very different stories and must never be conflated).

Throughput is normalized as atoms/sec and structures/sec rather than just
"iterations/sec", since structure size and (once `src/Model/Batching.jl`
lands) batch size both vary across the runs this harness is meant to compare
— atoms/sec is the one number that stays comparable from unbatched-CPU
through batched-GPU through Polaris-vs-H100.

Run with: `julia --project=. scripts/benchmark_throughput.jl`

Current scope: B=1 (unbatched) CPU baseline, swept over real structures of
different sizes and a couple of model scales. This is the number every later
step (batching, CUDA.jl, Polaris/H100) diffs against — see `docs/PLAN.md`'s
sequencing gates. Extend `run_one` with a `batch_size` parameter once
`Model.Batching` exists.
"""

using DiffuseBioMol
using Random, Zygote, Optimisers

const BENCH_TARGETS = [("1CRN", "A"), ("1UBQ", "A"), ("5PTI", "A")]

const BENCH_CONFIGS = [
    (label="toy (77K params)", cfg=ModelConfig(d_single=32, d_pair=16, n_heads=4, n_pairformer_layers=2, n_dit_layers=2, d_time=32)),
    (label="modest (2.5M params)", cfg=ModelConfig(d_single=128, d_pair=64, n_heads=8, n_pairformer_layers=4, n_dit_layers=6, d_time=128)),
]

function load_example(pdb_id, chain_id)
    residues = restrict_to_chain(fetch_pdb(pdb_id), chain_id)
    tokens = tokenize_structure(residues)
    feat = featurize(tokens)
    relpos = relpos_buckets(feat)
    x1 = target_coordinates(tokens)
    cond_features = constraint_features(no_constraints(length(tokens)))
    (pdb_id=pdb_id, feat=feat, relpos=relpos, x1=x1, cond_features=cond_features, n_atoms=length(tokens))
end

"""
    time_call(f) -> (result, seconds)

Wall-clock time for one call, returning the call's result alongside the
elapsed seconds (so the same helper covers both the "must inspect the output"
and "just need the timing" cases without calling `f` twice).
"""
function time_call(f)
    t0 = time()
    result = f()
    (result, time() - t0)
end

"""
    benchmark_step(model, ps, st, ex, rng; n_steady=5) -> NamedTuple

Measures, for one (model, example) pair at `B=1`:
- `fwd_compile_s` / `fwd_steady_s`: plain forward pass, first call vs. mean of
  `n_steady` subsequent calls.
- `pullback_compile_s` / `back_compile_s`: first-call `Zygote.pullback`
  forward and `back()` execution, kept separate (this distinction is exactly
  what the original Zygote investigation needed and didn't have until it was
  built ad hoc — never conflate these two numbers).
- `fwdbwd_steady_s`: mean steady-state combined forward+backward time.
- `atoms_per_sec` / `structures_per_sec`: derived from `fwdbwd_steady_s`.
"""
function benchmark_step(model, ps, st, ex, rng; n_steady::Int=5)
    example = prepare_training_example(ex.feat, ex.x1, ex.cond_features, rng)

    _, fwd_compile_s = time_call() do
        cfm_loss(model, ps, st, ex.feat, ex.relpos, example)
    end
    _, fwd_steady_s_total = time_call() do
        for _ in 1:n_steady
            cfm_loss(model, ps, st, ex.feat, ex.relpos, example)
        end
    end
    fwd_steady_s = fwd_steady_s_total / n_steady

    pullback_result, pullback_compile_s = time_call() do
        Zygote.pullback(p -> cfm_loss(model, p, st, ex.feat, ex.relpos, example)[1], ps)
    end
    _, back = pullback_result
    _, back_compile_s = time_call() do
        back(1.0f0)
    end

    _, fwdbwd_steady_s_total = time_call() do
        for _ in 1:n_steady
            _, back2 = Zygote.pullback(p -> cfm_loss(model, p, st, ex.feat, ex.relpos, example)[1], ps)
            back2(1.0f0)
        end
    end
    fwdbwd_steady_s = fwdbwd_steady_s_total / n_steady

    (
        pdb_id=ex.pdb_id, n_atoms=ex.n_atoms,
        fwd_compile_s=round(fwd_compile_s; digits=3), fwd_steady_s=round(fwd_steady_s; digits=4),
        pullback_compile_s=round(pullback_compile_s; digits=2), back_compile_s=round(back_compile_s; digits=2),
        fwdbwd_steady_s=round(fwdbwd_steady_s; digits=4),
        atoms_per_sec=round(ex.n_atoms / fwdbwd_steady_s; digits=1),
        structures_per_sec=round(1 / fwdbwd_steady_s; digits=4),
    )
end

"""
    benchmark_batched_step(model, ps, st, ex, batch_size, rng; n_steady=5, device=identity) -> NamedTuple

Same measurements as `benchmark_step`, but at a real batch size `B =
batch_size`, built from `ex` repeated `batch_size` times (so there's no
padding to confound the comparison — this isolates the throughput effect of
batching itself, on identical-size structures, before worrying about
padding waste from a realistic mixed-size dataset).

`device`: pass `Lux.gpu_device()` (after `using CUDA` — or `AMDGPU`/etc. —
in the calling session; DiffuseBioMol.jl itself adds no GPU dependency, see
`Model.Batching.to_device`'s docstring) to run this step on GPU instead of
CPU. `ps`/`st` must already live on the same device as `device` produces
(move them once, outside this function, with `ps = ps |> device` — moving
parameters every step would itself dominate the timing).
"""
function benchmark_batched_step(model, ps, st, ex, batch_size::Int, rng; n_steady::Int=5, device=identity)
    # Build everything CPU-side first (padding/masking logic is plain Julia,
    # not worth porting to GPU on its own), then move the handful of tensors
    # the model actually consumes to `device` in one pass.
    batched_feat_cpu = batch_features([ex.feat for _ in 1:batch_size])
    relpos_cpu = batch_relpos([ex.relpos for _ in 1:batch_size])
    cond_cpu = batch_cond_features([ex.cond_features for _ in 1:batch_size])
    x1_cpu = batch_coords([ex.x1 for _ in 1:batch_size])
    example_cpu = prepare_training_example(batched_feat_cpu, x1_cpu, cond_cpu, rng)

    batched_feat = to_device(batched_feat_cpu, device)
    relpos_batched = device(relpos_cpu)
    pad_bias = device(attention_pad_bias(batched_feat_cpu.pad_mask))
    example = BatchedTrainingExample(
        device(example_cpu.x_t), example_cpu.t, device(example_cpu.target_v),
        device(Array{Bool}(example_cpu.mask)), device(example_cpu.cond_features),
    )

    # Warmup: pay this batch size's first-call compile cost once, uncounted,
    # so the steady-state numbers below are comparable across batch sizes
    # regardless of which one happens to run first in this process.
    _, back0 = Zygote.pullback(p -> cfm_loss(model, p, st, batched_feat, relpos_batched, pad_bias, example)[1], ps)
    back0(1.0f0)

    _, fwd_compile_s = time_call() do
        cfm_loss(model, ps, st, batched_feat, relpos_batched, pad_bias, example)
    end
    _, fwd_steady_s_total = time_call() do
        for _ in 1:n_steady
            cfm_loss(model, ps, st, batched_feat, relpos_batched, pad_bias, example)
        end
    end
    fwd_steady_s = fwd_steady_s_total / n_steady

    _, fwdbwd_steady_s_total = time_call() do
        for _ in 1:n_steady
            _, back = Zygote.pullback(p -> cfm_loss(model, p, st, batched_feat, relpos_batched, pad_bias, example)[1], ps)
            back(1.0f0)
        end
    end
    fwdbwd_steady_s = fwdbwd_steady_s_total / n_steady
    total_atoms = ex.n_atoms * batch_size

    (
        pdb_id=ex.pdb_id, batch_size=batch_size, n_atoms=ex.n_atoms,
        fwd_compile_s=round(fwd_compile_s; digits=3), fwd_steady_s=round(fwd_steady_s; digits=4),
        fwdbwd_steady_s=round(fwdbwd_steady_s; digits=4),
        atoms_per_sec=round(total_atoms / fwdbwd_steady_s; digits=1),
        structures_per_sec=round(batch_size / fwdbwd_steady_s; digits=4),
    )
end

"""
    run_batch_size_sweep(; pdb_id="1CRN", chain="A", batch_sizes=(1, 4, 8, 16), device=identity)

Section 2's verification gate: confirm atoms/sec improves with batch size on
CPU before moving to CUDA.jl (porting an architecture that doesn't benefit
from batching would just move the bottleneck onto more expensive hardware).

**This is also the Section 4 entry point for real Polaris/H100 numbers**:
on a CUDA-capable node, run `using CUDA` before this script, then call
`run_batch_size_sweep(device=Lux.gpu_device())` (`Lux` is re-exported
transitively via `DiffuseBioMol.Model.Network.Lux` — `import
DiffuseBioMol.Model.Network.Lux as Lux`). This sandbox has no GPU, so this
path is written but not executable/verified here — see `docs/PLAN.md`'s
Section 4 for the handoff.
"""
function run_batch_size_sweep(; pdb_id="1CRN", chain="A", batch_sizes=(1, 4, 8, 16), seed=0, device=identity)
    rng = Random.Xoshiro(seed)
    ex = load_example(pdb_id, chain)
    cfg = ModelConfig(d_single=32, d_pair=16, n_heads=4, n_pairformer_layers=2, n_dit_layers=2, d_time=32)
    model = build_model(cfg)
    ps, st = DiffuseBioMol.Model.Network.Lux.setup(rng, model)
    ps, st = device(ps), device(st)

    println("\nBatch-size sweep on $pdb_id ($(ex.n_atoms) atoms), toy (77K param) config, device=$device:")
    println(rpad("batch_size", 12), rpad("fwdbwd_steady_s", 18), rpad("atoms/sec", 12), "structures/sec")
    reports = NamedTuple[]
    for b in batch_sizes
        r = benchmark_batched_step(model, ps, st, ex, b, rng; device=device)
        push!(reports, r)
        println(rpad(string(b), 12), rpad(string(r.fwdbwd_steady_s), 18), rpad(string(r.atoms_per_sec), 12), string(r.structures_per_sec))
    end
    reports
end

function main(; seed=0)
    rng = Random.Xoshiro(seed)
    println("Loading benchmark structures...")
    examples = [load_example(id, chain) for (id, chain) in BENCH_TARGETS]
    for ex in examples
        println("  $(ex.pdb_id): $(ex.n_atoms) atoms")
    end

    reports = NamedTuple[]
    for (label, cfg) in BENCH_CONFIGS
        println("\nBuilding model: $label ($cfg)")
        model = build_model(cfg)
        ps, st = DiffuseBioMol.Model.Network.Lux.setup(rng, model)
        for ex in examples
            println("  benchmarking $(ex.pdb_id)...")
            r = benchmark_step(model, ps, st, ex, rng)
            push!(reports, merge((config=label,), r))
        end
    end

    println("\n", "="^130)
    println(
        rpad("config", 24), rpad("pdb_id", 8), rpad("n_atoms", 9), rpad("fwd_compile_s", 15),
        rpad("fwd_steady_s", 14), rpad("pullback_compile_s", 20), rpad("back_compile_s", 16),
        rpad("fwdbwd_steady_s", 17), rpad("atoms/sec", 11), "struct/sec",
    )
    println("-"^130)
    for r in reports
        println(
            rpad(r.config, 24), rpad(r.pdb_id, 8), rpad(string(r.n_atoms), 9), rpad(string(r.fwd_compile_s), 15),
            rpad(string(r.fwd_steady_s), 14), rpad(string(r.pullback_compile_s), 20), rpad(string(r.back_compile_s), 16),
            rpad(string(r.fwdbwd_steady_s), 17), rpad(string(r.atoms_per_sec), 11), string(r.structures_per_sec),
        )
    end
    println("="^130)
    println("\nThis is the B=1, CPU, unbatched baseline -- the number every later step")
    println("(batching, CUDA.jl, Polaris/H100) diffs against. See docs/PLAN.md's sequencing gates.")

    run_batch_size_sweep(; seed=seed)
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
