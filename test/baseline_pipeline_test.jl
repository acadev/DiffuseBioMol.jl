using TOML

# The baseline runner is intentionally a script rather than package API: it
# owns experiment files/checkpoints while the package remains model-focused.
include(joinpath(@__DIR__, "..", "scripts", "train_baseline.jl"))

@testset "baseline pipeline: local corpus manifest, deterministic split, and gates" begin
    mktempdir() do dir
        data_dir = joinpath(dir, "structures")
        mkpath(data_dir)
        for i in 1:3
            write(joinpath(data_dir, "structure_$i.pdb"), SAMPLE_PDB)
        end

        cache_path = joinpath(dir, "corpus_cache.jls")
        examples, skipped = load_corpus(data_dir; max_atoms=100, cache_path, progress_every=1)
        @test length(examples) == 3
        @test isempty(skipped)
        @test isfile(cache_path)
        source = first(examples)
        cap = maximum(length, residue_groups(source))
        sequence_crop = crop_example(source, cap, "sequence", MersenneTwister(3); crop_id=1)
        spatial_crop = crop_example(source, cap, "spatial", MersenneTwister(3); crop_id=1)
        @test n_atoms(sequence_crop) <= cap
        @test n_atoms(spatial_crop) <= cap
        @test sequence_crop.source == source.source
        @test spatial_crop.source == source.source
        @test all(length(group) <= cap for group in residue_groups(sequence_crop))
        epoch_a = materialize_crops([source], cap, "mixed", 7, 1)
        epoch_b = materialize_crops([source], cap, "mixed", 7, 2)
        @test only(epoch_a).source == source.source
        @test only(epoch_b).source == source.source
        @test only(epoch_a).label != only(epoch_b).label
        crop_sources, _ = load_corpus(data_dir; max_atoms=cap, oversize_policy="crop", progress_every=1)
        @test length(crop_sources) == 3
        crop_train, crop_val = split_examples(crop_sources, 7, 0.34)
        crop_train_examples = materialize_crops(crop_train, cap, "mixed", 7, 1)
        crop_val_examples = materialize_crops(crop_val, cap, "mixed", 7, 0)
        @test isempty(intersect(Set(ex.source for ex in crop_train_examples), Set(ex.source for ex in crop_val_examples)))
        limited_examples, _ = load_corpus(data_dir; max_atoms=100, max_candidate_files=2, selection_seed=7,
            progress_every=1)
        @test length(limited_examples) == 2
        cached_examples, cached_skipped = load_corpus(data_dir; max_atoms=100, cache_path, progress_every=1)
        @test [ex.source for ex in cached_examples] == [ex.source for ex in examples]
        @test cached_skipped == skipped
        # A source-file change invalidates the cache rather than silently
        # training on stale coordinates.
        open(joinpath(data_dir, "structure_1.pdb"), "a") do io
            write(io, "\n")
        end
        rebuilt_examples, rebuilt_skipped = load_corpus(data_dir; max_atoms=100, cache_path, progress_every=1)
        @test length(rebuilt_examples) == 3
        @test isempty(rebuilt_skipped)

        prepare_config = joinpath(dir, "prepare.toml")
        open(prepare_config, "w") do io
            TOML.print(io, Dict(
                "data" => Dict("data_dir" => data_dir, "max_atoms" => 100,
                    "min_structures" => 3, "validation_fraction" => 0.34, "load_progress_every" => 1),
                "model" => Dict{String,Any}(), "training" => Dict("seed" => 7),
                "evaluation" => Dict("sentinel_size" => 1), "wandb" => Dict("enabled" => false),
            ))
        end
        prepare_dir = joinpath(dir, "prepared-run")
        prepared = main(prepare_config, prepare_dir; prepare_only=true)
        @test length(prepared.training) == 2
        @test isfile(joinpath(prepare_dir, "corpus_cache.jls"))
        train_a, val_a = split_examples(examples, 7, 0.34)
        train_b, val_b = split_examples(examples, 7, 0.34)
        @test [ex.source for ex in train_a] == [ex.source for ex in train_b]
        @test [ex.source for ex in val_a] == [ex.source for ex in val_b]

        config = (path="fixture.toml",)
        manifest = joinpath(dir, "manifest.toml")
        write_manifest(manifest, config, train_a, val_a, skipped)
        saved = TOML.parsefile(manifest)
        @test length(saved["training"]) == 2
        @test length(saved["validation"]) == 1

        initial = [
            (condition="prior", label="x", n_atoms=9, rmsd=10.0, clashes=9.0, bond_rmsd=3.0, chirality_bad=2.0),
            (condition="model", label="x", n_atoms=9, rmsd=8.0, clashes=7.0, bond_rmsd=2.0, chirality_bad=1.0),
        ]
        final = [
            (condition="model", label="x", n_atoms=9, rmsd=6.0, clashes=5.0, bond_rmsd=1.0, chirality_bad=0.0),
            (condition="model_guided", label="x", n_atoms=9, rmsd=6.0, clashes=4.0, bond_rmsd=0.8, chirality_bad=0.0),
        ]
        gates = gate_report(initial, final, 0.1)
        @test gates["reconstruction_pass"]
        @test gates["trained_beats_prior_on_all_metrics"]
        @test gates["guidance_non_regression"]
        @test gates["all_passed"]
    end
end
