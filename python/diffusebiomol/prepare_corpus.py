"""Build a training corpus directly from local PDB/mmCIF files using AtomWorks."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from .tokenizer import VOCABULARY, VOCAB_SIZES, parse_structure, tokenize

EXTENSIONS = (".pdb", ".ent", ".cif", ".mmcif")


def prepare_corpus(source, output, *, chain="largest"):
    source, output = Path(source), Path(output)
    files = [source] if source.is_file() else sorted(
        p for p in source.rglob("*") if p.is_file()
        and p.name.lower().removesuffix(".gz").endswith(EXTENSIONS))
    if not files:
        raise ValueError(f"No PDB/mmCIF files found in {source}")
    # Fail before creating output if the required parser is not installed.
    parser_version = version("atomworks")
    output.mkdir(parents=True, exist_ok=False)
    entries, skipped = [], []
    for i, path in enumerate(files, 1):
        try:
            record = tokenize(parse_structure(path), chain=chain)
            payload = json.dumps(record, allow_nan=False, separators=(",", ":")).encode()
        except Exception as exc:
            skipped.append(dict(source=str(path.resolve()), error=f"{type(exc).__name__}: {exc}"))
            continue
        name = f"source_{i:08d}.json"
        (output / name).write_bytes(payload)
        entries.append(dict(file=name, source=str(path.resolve()), label=path.name,
            atoms=len(record["element"]), sha256=hashlib.sha256(payload).hexdigest()))
    manifest = dict(schema_version=1, index_base=0, vocab_sizes=VOCAB_SIZES,
        tokenizer_version=VOCABULARY["version"],
        polymer_vocabulary=VOCABULARY["polymer_vocabulary"],
        parser=dict(name="atomworks", version=parser_version, preset="minimal",
                    model=1, assembly="asymmetric_unit", altloc="first",
                    hydrogens="remove", missing_atoms="virtual", chain=chain),
        entries=entries, skipped=skipped)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not entries:
        raise ValueError(f"No usable structures; see {output / 'manifest.json'}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Local structure file or recursively scanned directory")
    parser.add_argument("output", help="New corpus directory")
    parser.add_argument("--chain", default="largest", help="largest, all, or exact chain ID")
    args = parser.parse_args()
    manifest = prepare_corpus(args.source, args.output, chain=args.chain)
    print(f"Prepared {len(manifest['entries'])} sources; {len(manifest['skipped'])} skipped. See {args.output}/manifest.json")


if __name__ == "__main__":
    main()
