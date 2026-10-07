"""AtomWorks structures to versioned, atom-level training records.

Canonical polymer slots and embedding IDs are frozen in vocabulary.json.
No Julia runtime, generated coordinates, or external CCD mirror is required.
"""
import json
from pathlib import Path

import numpy as np

VOCABULARY = json.loads(Path(__file__).with_name("vocabulary.json").read_text())
ELEMENTS = {e.upper(): i for i, e in enumerate(VOCABULARY["elements"])}
POLYMER = {(v["residue"], v["atom"]): v["index"]
           for v in VOCABULARY["polymer_vocabulary"]}
SLOTS = {}
for entry in VOCABULARY["polymer_vocabulary"]:
    SLOTS.setdefault(entry["residue"], []).append(entry["atom"])
VOCAB_SIZES = [len(ELEMENTS) + 1, 6, len(POLYMER) + 1, 66]
RNA = {"A", "C", "G", "U"}
DNA = {"DA", "DC", "DG", "DT"}
PTMS = {"SEP", "TPO", "PTR", "MSE", "MLY", "M3L", "MLZ", "ALY",
        "CSO", "CSD", "HYP", "PCA"}
WATERS = {"HOH", "DOD", "WAT", "H2O"}
ION_ELEMENTS = {"LI", "NA", "K", "RB", "CS", "MG", "CA", "SR", "BA", "MN",
                "FE", "CO", "NI", "CU", "ZN", "CD", "HG", "F", "CL", "BR", "I"}


def parse_structure(path):
    """Read the first model/asymmetric unit, with deterministic first altlocs."""
    from atomworks.io import parse
    from atomworks.io.config import ParseConfig

    suffix = str(path).lower().removesuffix(".gz")
    file_type = "pdb" if suffix.endswith((".pdb", ".ent")) else "cif"
    config = ParseConfig.from_preset(
        "minimal", model=1, build_assembly=None, file_type=file_type,
        hydrogen_policy="remove", remove_waters=True,
        fix_ligands_at_symmetry_centers=False, ccd_mirror_path=None,
    )
    atoms = parse(str(path), config=config)["asym_unit"]
    return atoms[0] if atoms.coord.ndim == 3 else atoms


def tokenize(atoms, *, chain="largest"):
    """Convert a Biotite AtomArray into the trainer's seven parallel arrays.

    ``chain`` is ``largest`` (most canonical polymer residues), ``all``, or an
    exact chain ID. Chain IDs are recoded in encounter order. Residue numbers
    preserve positive gaps; insertion codes/repeated numbers get distinct,
    increasing indices so crops never merge adjacent residues.
    Unknown polymer components/PTMs use observed atoms and the catch-all ID.
    """
    if atoms.coord.ndim != 2:
        raise ValueError("Select one model before tokenization")
    annotations = set(atoms.get_annotation_categories())
    groups = {}
    for i in range(len(atoms)):
        name, element = str(atoms.res_name[i]), str(atoms.element[i]).upper()
        if name in WATERS or element in {"H", "D", "T"}:
            continue
        key = (str(atoms.chain_id[i]), int(atoms.res_id[i]),
               str(atoms.ins_code[i]), name, bool(atoms.hetero[i]))
        groups.setdefault(key, []).append(i)
    if chain == "largest":
        counts = {}
        for key in groups:
            if key[3] in SLOTS:
                counts[key[0]] = counts.get(key[0], 0) + 1
        if not counts:
            raise ValueError("No canonical polymer chain; use chain='all' for non-polymers")
        chain = min(counts, key=lambda c: (-counts[c], c))
    selected = [(key, indices) for key, indices in groups.items()
                if chain == "all" or key[0] == chain]
    record = {k: [] for k in ("element", "modality", "polymer", "chain", "residue", "virtual", "xyz")}
    chains, last_residue = {}, {}
    for (chain_id, number, insertion, name, hetero), indices in selected:
        chain_index = chains.setdefault(chain_id, len(chains) + 1)
        residue_index = max(number, last_residue.get(chain_id, number - 1) + 1)
        last_residue[chain_id] = residue_index
        by_name = {}
        for i in indices:
            atom_name = str(atoms.atom_name[i])
            if atom_name in by_name:
                raise ValueError(f"Duplicate atom {chain_id}/{number}{insertion}/{name}/{atom_name}")
            by_name[atom_name] = i
        if name in SLOTS:
            modality = 1 if name in RNA else 2 if name in DNA else 0
            names = SLOTS[name]
        else:
            is_polymer = (bool(np.any(atoms.is_polymer[indices]))
                          if "is_polymer" in annotations else not hetero)
            modality = (5 if name in PTMS or is_polymer else
                        4 if len(indices) == 1 and str(atoms.element[indices[0]]).upper() in ION_ELEMENTS else 3)
            names = sorted(by_name)
        for atom_name in names:
            i = by_name.get(atom_name)
            observed = i is not None and bool(np.isfinite(atoms.coord[i]).all())
            if observed and "occupancy" in annotations:
                observed = bool(np.isfinite(atoms.occupancy[i]) and atoms.occupancy[i] > 0)
            if observed and "is_atom_unresolved" in annotations:
                observed = not bool(atoms.is_atom_unresolved[i])
            element = str(atoms.element[i]).upper() if i is not None else atom_name[0]
            record["element"].append(ELEMENTS.get(element, len(ELEMENTS)))
            record["modality"].append(modality)
            record["polymer"].append(POLYMER.get((name, atom_name), len(POLYMER)))
            record["chain"].append(chain_index)
            record["residue"].append(residue_index)
            record["virtual"].append(not observed)
            record["xyz"].append(atoms.coord[i].tolist() if observed else [0., 0., 0.])
    if not record["element"] or all(record["virtual"]):
        raise ValueError("Selection has no observed heavy atoms")
    return record
