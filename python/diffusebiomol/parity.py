"""Compare identical Julia/PyTorch weights, forward outputs and selected gradients."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import FlowModel, ModelConfig


def unpack(record, dtype=torch.float32):
    return torch.tensor(np.array(record["values"]).reshape(record["shape"], order="F").copy(), dtype=dtype)


def verify(path):
    torch.set_num_threads(2)
    ref = json.loads(Path(path).read_text())
    model = FlowModel(ModelConfig(), ref["vocab"])
    state = {k: unpack(v) for k, v in ref["weights"].items()}
    state["freq"] = model.freq
    model.load_state_dict(state, strict=True)
    batch = {k: unpack(v, torch.long) for k, v in ref["batch"].items()}
    batch["valid"] = torch.ones_like(batch["element"], dtype=torch.bool)
    output = model(batch, unpack(ref["x"]), unpack(ref["t"]), unpack(ref["cond"]))
    loss = output.square().mean()
    loss.backward()
    pairs = dict(output=(output, unpack(ref["output"])),
                 loss=(loss, torch.tensor(ref["loss"])),
                 grad_coord=(model.coord.weight.grad, unpack(ref["grad_coord"])),
                 grad_head=(model.head.weight.grad, unpack(ref["grad_head"])))
    errors = {}
    for key, (actual, expected) in pairs.items():
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4, msg=key)
        errors[key + "_max_abs_error"] = (actual-expected).abs().max().item()
    print(json.dumps(errors, indent=2))
    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    verify(parser.parse_args().reference)
