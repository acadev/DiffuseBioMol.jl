"""Lazy linear-size source records; pair features are built only after cropping."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


class Corpus:
    def __init__(self, root):
        self.root = Path(root)
        raw = (self.root / "manifest.json").read_bytes()
        self.signature = hashlib.sha256(raw).hexdigest()
        self.manifest = json.loads(raw)
        if self.manifest["schema_version"] != 1 or self.manifest["index_base"] != 0:
            raise ValueError("Unsupported corpus schema")
        self.entries = self.manifest["entries"]
        self.vocab = self.manifest["vocab_sizes"]

    def load(self, index, cap, rng):
        entry = self.entries[index]
        raw = (self.root / entry["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"Source checksum mismatch: {entry['file']}")
        record = {k: np.asarray(v) for k, v in json.loads(raw).items()}
        n = len(record["element"])
        boundaries = np.flatnonzero(np.r_[True, (np.diff(record["residue"]) != 0) | (np.diff(record["chain"]) != 0), True])
        groups = [np.arange(a, b) for a, b in zip(boundaries[:-1], boundaries[1:])]
        if n > cap:
            starts = [i for i, g in enumerate(groups) if len(g) <= cap]
            if not starts:
                raise ValueError("No complete residue fits crop budget")
            start = int(rng.choice(starts))
            selected, used = [], 0
            for group in groups[start:]:
                if used + len(group) > cap:
                    break
                selected.extend(group)
                used += len(group)
            record = {k: v[selected] for k, v in record.items()}
        if np.all(record["virtual"]):
            raise ValueError("Crop has no observed atoms")
        return record


def collate(records):
    b, n = len(records), max(len(r["element"]) for r in records)
    out = {k: torch.zeros(b, n, dtype=torch.long) for k in ("element", "modality", "polymer", "chain", "residue")}
    out["xyz"] = torch.zeros(b, n, 3)
    out["valid"] = torch.zeros(b, n, dtype=torch.bool)
    out["virtual"] = torch.ones(b, n, dtype=torch.bool)
    for i, r in enumerate(records):
        size = len(r["element"])
        for key in r:
            out[key][i, :size] = torch.as_tensor(r[key], dtype=out[key].dtype)
        out["valid"][i, :size] = True
    delta = out["residue"][:, :, None] - out["residue"][:, None, :]
    out["relpos"] = (delta.clamp(-32, 32) + 32).masked_fill(
        out["chain"][:, :, None] != out["chain"][:, None, :], 65)
    return out
