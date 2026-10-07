import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from diffusebiomol.data import Corpus, collate
from diffusebiomol.flow import cfm_loss, prepare, sample_flow
from diffusebiomol.model import FlowModel, ModelConfig
from diffusebiomol.train import train


def record(n):
    return dict(element=np.arange(n) % 3, modality=np.zeros(n, dtype=int),
        polymer=np.arange(n) % 10, chain=np.ones(n, dtype=int),
        residue=np.arange(n)//4, virtual=np.zeros(n, dtype=bool),
        xyz=np.arange(n*3, dtype=np.float32).reshape(n, 3)/5)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(5)
        self.model = FlowModel(ModelConfig(), [17, 6, 20, 66])

    def test_padding_and_gradient_invariance(self):
        # Nonzero output weights ensure this exercises attention and pair updates.
        torch.nn.init.normal_(self.model.head.weight, std=0.1)
        short = collate([record(8)])
        padded = collate([record(8), record(12)])
        x = torch.randn(1, 8, 3)
        t = torch.tensor([0.3])
        expected = self.model(short, x, t)
        expanded = torch.randn(2, 12, 3)
        expanded[0, :8] = x[0]
        actual = self.model(padded, expanded, t.expand(2))
        torch.testing.assert_close(expected[0], actual[0, :8], atol=1e-6, rtol=1e-5)
        expected.sum().backward()
        grad = self.model.coord.weight.grad.clone()
        self.model.zero_grad()
        actual[0, :8].sum().backward()
        torch.testing.assert_close(grad, self.model.coord.weight.grad, atol=2e-6, rtol=1e-5)

    def test_loss_masks_and_linear_path(self):
        r = record(8)
        r["virtual"][3] = True
        batch = collate([r, record(12)])
        example = prepare(batch, np.random.default_rng(1))
        self.assertEqual(int(example["mask"].sum()), 19)
        self.assertTrue(torch.all(example["target"][~example["mask"]] == 0))
        # Initial head is zero; check exact normalization by real coordinate count.
        expected = example["target"].square().sum() / (3*19)
        torch.testing.assert_close(cfm_loss(self.model, batch, example), expected)
        self.assertTrue(torch.isfinite(sample_flow(self.model, batch, np.random.default_rng(2), 3)).all())

    def test_fixed_example_learns(self):
        corpus_path = os.getenv("DBM_CORPUS")
        if corpus_path:
            corpus = Corpus(corpus_path)
            self.model = FlowModel(ModelConfig(), corpus.vocab)
            data = corpus.load(0, 128, np.random.default_rng(2))
        else:
            data = record(16)
        batch = collate([data])
        example = prepare(batch, np.random.default_rng(4))
        before = cfm_loss(self.model, batch, example).item()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        for _ in range(150):
            optimizer.zero_grad()
            loss = cfm_loss(self.model, batch, example)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
        after = cfm_loss(self.model, batch, example).item()
        print(f"Fixed-example loss: {before:.6f} -> {after:.6f}")
        self.assertLess(after, before / 5)

    def test_resume_and_corpus_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus_path = os.getenv("DBM_CORPUS")
            if not corpus_path:
                corpus_path = root / "corpus"
                corpus_path.mkdir()
                entries = []
                for i in range(6):
                    r = record(12 + 4*i)
                    raw = json.dumps({k: v.tolist() for k, v in r.items()}).encode()
                    name = f"{i}.json"
                    (corpus_path / name).write_bytes(raw)
                    entries.append(dict(file=name, atoms=len(r["element"]), source=name,
                                        sha256=hashlib.sha256(raw).hexdigest()))
                (corpus_path / "manifest.json").write_text(json.dumps(dict(
                    schema_version=1, index_base=0, vocab_sizes=[17,6,20,66], entries=entries)))
            train(corpus_path, root / "full", epochs=3)
            train(corpus_path, root / "resume", epochs=2)
            train(corpus_path, root / "resume", epochs=3, resume=True)
            a = torch.load(root / "full/checkpoint.pt", weights_only=True)
            b = torch.load(root / "resume/checkpoint.pt", weights_only=True)
            def equal(x, y):
                if isinstance(x, torch.Tensor):
                    self.assertTrue(torch.equal(x, y))
                elif isinstance(x, dict):
                    self.assertEqual(x.keys(), y.keys())
                    for key in x:
                        equal(x[key], y[key])
                elif isinstance(x, list):
                    self.assertEqual(len(x), len(y))
                    for xx, yy in zip(x, y):
                        equal(xx, yy)
                else:
                    self.assertEqual(x, y)
            equal(a, b)
            self.assertEqual(a["presentations"], 12)
            with self.assertRaises(ValueError):
                train(corpus_path, root / "resume", epochs=4, resume=True, max_atoms=64)
            corpus = Corpus(corpus_path)
            data = corpus.load(0, 24, np.random.default_rng(4))
            self.assertLessEqual(len(data["element"]), 24)
            # Every selected residue is complete relative to the full source.
            source = corpus.load(0, 100000, np.random.default_rng(4))
            for residue in np.unique(data["residue"]):
                self.assertEqual(sum(data["residue"] == residue), sum(source["residue"] == residue))
            corpus.entries = copy.deepcopy(corpus.entries)
            corpus.entries[0]["sha256"] = "bad"
            with self.assertRaises(ValueError):
                corpus.load(0, 128, np.random.default_rng(2))


if __name__ == "__main__":
    unittest.main()
