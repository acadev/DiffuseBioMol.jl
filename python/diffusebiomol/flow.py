"""Linear-path CFM, polymer random-walk prior, SE(3) augmentation and Euler sampling."""
import numpy as np
import torch


def prior(batch, rng):
    chain = batch["chain"].cpu().numpy()
    result = np.zeros((*chain.shape, 3), dtype=np.float32)
    for b in range(len(chain)):
        for i in range(chain.shape[1]):
            if i and chain[b, i] == chain[b, i-1]:
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                result[b, i] = result[b, i-1] + (3.8 + rng.normal()) * direction
            else:
                result[b, i] = 10 * rng.normal(size=3)
    return torch.from_numpy(result)


def prepare(batch, rng):
    # Randomness stays on CPU for repeatable CPU/CUDA input preparation.
    real = batch["valid"] & ~batch["virtual"]
    xyz = batch["xyz"].numpy().copy()
    for b in range(len(xyz)):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q *= np.sign(np.diag(r))[None, :]
        if np.linalg.det(q) < 0:
            q[:, -1] *= -1
        xyz[b] = (xyz[b] - xyz[b, real[b].numpy()].mean(axis=0)) @ q.T
    x0 = prior(batch, rng)
    x1 = torch.from_numpy(xyz)
    x1 = torch.where(real[..., None], x1, x0)
    t = torch.tensor(rng.random(len(xyz)), dtype=torch.float32)
    return {"x": (1-t[:, None, None])*x0 + t[:, None, None]*x1,
            "t": t, "target": x1-x0, "mask": real}


def cfm_loss(model, batch, example):
    predicted = model(batch, example["x"], example["t"])
    error = (predicted - example["target"]).square() * example["mask"][..., None]
    return error.sum() / (3 * example["mask"].sum().clamp_min(1))


@torch.no_grad()
def sample_flow(model, host_batch, rng, steps=40):
    if steps <= 0:
        raise ValueError("steps must be positive")
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in host_batch.items()}
    x = prior(host_batch, rng).to(device)
    for i in range(steps):
        t = torch.full((len(x),), i / steps, device=device)
        x = x + model(batch, x, t) / steps
    return x.masked_fill(~batch["valid"][..., None], 0).cpu()
