import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, set_structure

from diffusebiomol.data import Corpus, collate
from diffusebiomol.tokenizer import VOCAB_SIZES, parse_structure, tokenize
from diffusebiomol.prepare_corpus import prepare_corpus


def atoms(rows):
    """Rows: chain, residue ID, insertion, component, atom, element, hetero."""
    array = AtomArray(len(rows))
    for i, row in enumerate(rows):
        (array.chain_id[i], array.res_id[i], array.ins_code[i], array.res_name[i],
         array.atom_name[i], array.element[i], array.hetero[i]) = row
    array.coord = np.arange(len(rows)*3, dtype=np.float32).reshape(-1, 3)
    array.set_annotation("occupancy", np.ones(len(rows)))
    return array


class TokenizerTests(unittest.TestCase):
    def test_slots_masks_and_vocabulary(self):
        a = atoms([("A", 10, "", "GLY", "CA", "C", False),
                   ("A", 10, "", "GLY", "O", "O", False)])
        a.occupancy[1] = 0
        r = tokenize(a)
        self.assertEqual(VOCAB_SIZES, [17, 6, 335, 66])
        self.assertEqual(r["polymer"], [57, 58, 59, 60])
        self.assertEqual(r["element"], [1, 0, 0, 2])
        self.assertEqual(r["virtual"], [True, False, True, True])
        self.assertEqual(r["xyz"][3], [0., 0., 0.])
        a.coord[0] = np.nan
        with self.assertRaisesRegex(ValueError, "no observed"):
            tokenize(a)

    def test_modalities_elements_and_filters(self):
        a = atoms([("P", 1, "", "ALA", "CA", "C", False),
                   ("R", 1, "", "A", "P", "P", False),
                   ("D", 1, "", "DG", "P", "P", False),
                   ("L", 1, "", "LIG", "CL1", "CL", True),
                   ("L", 1, "", "LIG", "C1", "C", True),
                   ("I", 1, "", "ZN", "ZN", "ZN", True),
                   ("M", 1, "", "MSE", "SE", "SE", True),
                   ("U", 1, "", "UNK", "CA", "C", False),
                   ("W", 1, "", "HOH", "O", "O", True),
                   ("P", 1, "", "ALA", "H", "H", False)])
        r = tokenize(a, chain="all")
        self.assertEqual(set(r["modality"]), set(range(6)))
        self.assertIn(11, r["element"])  # chlorine, not carbon
        self.assertIn(7, r["element"])   # zinc
        self.assertNotIn(5, r["element"])  # hydrogen removed
        self.assertEqual(sum(not v for v in r["virtual"]), 8)
        self.assertEqual(set(tokenize(a, chain="R")["modality"]), {1})
        with self.assertRaises(ValueError):
            tokenize(a, chain="absent")

    def test_insertions_gaps_and_duplicate_atoms(self):
        a = atoms([("A", n, ins, "GLY", "CA", "C", False)
                   for n, ins in [(10, ""), (10, "A"), (10, "B"), (15, "")]])
        r = tokenize(a)
        self.assertEqual(r["residue"][::4], [10, 11, 12, 15])
        with self.assertRaisesRegex(ValueError, "Duplicate atom"):
            tokenize(a[[0, 0]])

    def test_pdb_cif_gzip_and_corpus(self):
        a = atoms([("A", n, "", "GLY", atom, element, False)
                   for n in [1, 2] for atom, element in [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")]])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdb = PDBFile(); pdb.set_structure(a); pdb.write(root / "a.pdb")
            cif = CIFFile(); set_structure(cif, a); cif.write(root / "b.cif")
            with gzip.open(root / "c.cif.gz", "wb") as f:
                f.write((root / "b.cif").read_bytes())
            with gzip.open(root / "d.pdb.gz", "wb") as f:
                f.write((root / "a.pdb").read_bytes())
            expected = tokenize(a)
            for name in ["a.pdb", "b.cif", "c.cif.gz", "d.pdb.gz"]:
                self.assertEqual(tokenize(parse_structure(root / name)), expected)
            (root / "broken.pdb").write_text("not a structure\n")
            manifest = prepare_corpus(root, root / "corpus")
            self.assertEqual(len(manifest["entries"]), 4)
            self.assertEqual(len(manifest["skipped"]), 1)
            corpus = Corpus(root / "corpus")
            crop = corpus.load(0, 4, np.random.default_rng(1))
            self.assertEqual(len(crop["element"]), 4)
            self.assertEqual(len(np.unique(crop["residue"])), 1)
            self.assertEqual(tuple(collate([crop])["xyz"].shape), (1, 4, 3))
            with self.assertRaises(FileExistsError):
                prepare_corpus(root, root / "corpus")

    def test_all_failed_writes_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bad.pdb").write_text("invalid\n")
            with self.assertRaisesRegex(ValueError, "No usable"):
                prepare_corpus(root, root / "out")
            manifest = json.loads((root / "out/manifest.json").read_text())
            self.assertEqual(len(manifest["skipped"]), 1)

    def test_first_model_and_alternate_conformer(self):
        a = atoms([("A", 1, "", "GLY", "CA", "C", False)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alternate.pdb"
            pdb = PDBFile(); pdb.set_structure(a); pdb.write(path)
            line = next(line for line in path.read_text().splitlines() if line.startswith("ATOM"))
            alt_a = line[:16] + "A" + line[17:]
            alt_b = line[:16] + "B" + line[17:30] + f"{50.:8.3f}" + line[38:]
            path.write_text("MODEL        1\n" + alt_a + "\n" + alt_b +
                            "\nENDMDL\nMODEL        2\n" + alt_b + "\nENDMDL\nEND\n")
            record = tokenize(parse_structure(path))
            self.assertEqual(record["xyz"][1], [0., 1., 2.])
            self.assertEqual(sum(not v for v in record["virtual"]), 1)


if __name__ == "__main__":
    unittest.main()
