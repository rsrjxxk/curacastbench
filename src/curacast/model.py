"""Reaction-difference retrieval encoder, contrastive training, and inference."""

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class MultiTokenRetriever(nn.Module):
    """Bidirectional block-attention enzyme encoder and reaction-difference projector."""

    def __init__(self) -> None:
        super().__init__()
        self.sequence = nn.Linear(160, 256)
        self.text = nn.Linear(128, 256)
        self.sequence_position = nn.Parameter(torch.randn(1, 8, 256) * 0.02)
        self.text_position = nn.Parameter(torch.randn(1, 6, 256) * 0.02)
        self.st = nn.MultiheadAttention(256, 4, dropout=0.1, batch_first=True)
        self.ts = nn.MultiheadAttention(256, 4, dropout=0.1, batch_first=True)
        self.sn = nn.LayerNorm(256)
        self.tn = nn.LayerNorm(256)
        self.enzyme = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
        )
        self.reaction = nn.Sequential(
            nn.Linear(3584, 1024),
            nn.GELU(),
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
        )
        self.decoder = nn.Linear(256, 3584)

    def encode_enzyme(self, sequence: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        s = self.sequence(sequence.reshape(-1, 8, 160)) + self.sequence_position
        t = self.text(text.reshape(-1, 6, 128)) + self.text_position
        sa = self.sn(s + self.st(s, t, t, need_weights=True)[0])
        ta = self.tn(t + self.ts(t, s, s, need_weights=True)[0])
        pooled = torch.cat((sa.mean(dim=1), ta.mean(dim=1)), dim=1)
        return F.normalize(self.enzyme(pooled), dim=1)

    def encode_reaction(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.reaction(x), dim=1)


def top_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Stable top-k selection breaking exact ties by ascending column index."""
    vals, idx = scores.topk(k + 1, dim=1, sorted=True)
    idx = idx[:, :k].sort(dim=1).values
    gathered = scores.gather(1, idx)
    order = torch.argsort(gathered, dim=1, descending=True, stable=True)
    idx = idx.gather(1, order)
    tied = torch.where(vals[:, k - 1] == vals[:, k])[0]
    for block in tied.split(64):
        idx[block] = torch.argsort(scores[block], dim=1, descending=True, stable=True)[:, :k]
    return idx


def multi_positive_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Symmetric multi-positive InfoNCE objective."""
    positives = logits.masked_fill(~mask, -torch.inf)
    row_loss = (torch.logsumexp(logits, dim=1) - torch.logsumexp(positives, dim=1)).mean()
    col_loss = (torch.logsumexp(logits, dim=0) - torch.logsumexp(positives, dim=0)).mean()
    return 0.5 * (row_loss + col_loss)


def reaction_vectors(molecular: np.ndarray, fingerprints: np.ndarray, sides: list) -> tuple[np.ndarray, np.ndarray]:
    """Assemble IDF-weighted reactant, product, and difference features (3584-D)."""
    df = Counter(i for pair in sides if pair for i in set(pair[0] + pair[1]))
    weights = np.array(
        [np.log((len(sides) + 1) / (df.get(i, 0) + 1)) for i in range(len(molecular))],
        dtype=np.float32,
    )
    result = np.zeros((len(sides), 3584), dtype=np.float32)
    valid = np.zeros(len(sides), dtype=bool)

    def unit(vec: np.ndarray) -> np.ndarray:
        return vec / max(float(np.linalg.norm(vec)), 1e-8)

    for i, pair in enumerate(sides):
        if not pair or not pair[0] or not pair[1]:
            continue
        side_vecs = []
        side_counts = []
        for ids in pair:
            w = weights[ids]
            w = w / w.sum()
            side_vecs.append((np.asarray(molecular[ids]) * w[:, None]).sum(axis=0))
            side_counts.append((np.asarray(fingerprints[ids]) * w[:, None]).sum(axis=0))
        left, right = side_vecs
        result[i] = np.concatenate(
            [unit(left), unit(right), unit(right - left), unit(side_counts[1] - side_counts[0])]
        )
        valid[i] = True
    return result, valid


def load_inputs(path: Path, device: str):
    path = Path(path)
    s = np.load(path / "sequence.npy", mmap_mode="r")
    t = np.load(path / "text.npy", mmap_mode="r")
    r = np.load(path / "reaction.npy", mmap_mode="r")
    known = np.asarray(np.load(path / "known.npy"), dtype=np.int64)
    valid = np.asarray(np.load(path / "valid.npy"), dtype=bool)

    if s.ndim != 2 or s.shape[1] != 1280 or t.shape != (len(s), 768) or r.ndim != 2 or r.shape[1] != 3584:
        raise ValueError("Expected sequence (N, 1280), text (N, 768), and reaction (T, 3584)")
    if valid.shape != (len(r),) or known.ndim != 2 or known.shape[1] != 2:
        raise ValueError("Invalid valid mask or known-pair array shape")
    if len(known) and (known.min() < 0 or known[:, 0].max() >= len(s) or known[:, 1].max() >= len(r)):
        raise ValueError("Known pair index out of bounds")

    known = np.unique(known, axis=0)
    tensors = [torch.as_tensor(np.array(arr), device=device) for arr in (s, t, r)]
    return (*tensors, valid, known)


def fit(
    path: Path,
    checkpoint: Path,
    seed: int = 20261112,
    epochs: int = 20,
    batch: int = 1024,
    device: str = "cuda",
) -> MultiTokenRetriever:
    """Train MultiTokenRetriever with target-degree logit adjustment and auxiliary reconstruction."""
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    s, t, r, valid, known = load_inputs(path, device)
    model = MultiTokenRetriever().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    pairs = known[valid[known[:, 1]]]
    if not len(pairs):
        raise ValueError("No supported training pairs")

    keys = torch.as_tensor(np.sort(known[:, 0] * len(r) + known[:, 1]), device=device)
    degree = torch.as_tensor(np.bincount(known[:, 1], minlength=len(r)), dtype=torch.float32, device=device)
    supported = np.flatnonzero(valid)

    for epoch in range(epochs):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        perm = rng.permutation(len(pairs))
        total_loss = 0.0
        for begin in range(0, len(pairs), batch):
            q, y = pairs[perm[begin : begin + batch]].T
            qt = torch.as_tensor(q, device=device)
            yt = torch.as_tensor(y, device=device)
            proposed = qt[:, None] * len(r) + yt[None, :]
            ix = torch.searchsorted(keys, proposed)
            mask = (ix < len(keys)) & (keys[ix.clamp_max(len(keys) - 1)] == proposed)

            e = model.encode_enzyme(s[qt].float(), t[qt].float())
            z = model.encode_reaction(r[yt].float())
            logits = (e @ z.T) / 0.07 - 0.5 * torch.log1p(degree[yt])[None, :]

            sample_rxn = torch.as_tensor(
                rng.choice(supported, min(256, len(supported)), replace=False),
                device=device,
            )
            recon = F.mse_loss(model.decoder(model.encode_reaction(r[sample_rxn].float())), r[sample_rxn].float())
            objective = multi_positive_loss(logits, mask) + 0.1 * recon
            if not torch.isfinite(objective):
                raise ValueError("Non-finite training loss")

            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(objective.detach()) * len(q)

        print(f"epoch {epoch + 1}/{epochs} loss {total_loss / len(pairs):.6f}", flush=True)

    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    return model.eval()


@torch.inference_mode()
def retrieve(
    path: Path,
    checkpoint: Path,
    depth: int = 100,
    batch: int = 2048,
    device: str = "cuda",
) -> np.ndarray:
    """Encode queries and targets and return top-k candidate indices per query."""
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    s, t, r, valid, known = load_inputs(path, device)
    model = MultiTokenRetriever().to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

    e = torch.cat(
        [model.encode_enzyme(s[i : i + batch].float(), t[i : i + batch].float()) for i in range(0, len(s), batch)]
    )
    z = torch.cat([model.encode_reaction(r[i : i + 512].float()) for i in range(0, len(r), 512)])

    from .ops import search

    op = "gpu" if device.startswith("cuda") else "cpu"
    return search(e.cpu().numpy(), z.cpu().numpy(), known, valid, operator=op, depth=depth, batch=batch)
