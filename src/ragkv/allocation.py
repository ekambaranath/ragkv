"""Turn retriever scores into a per-chunk token budget.

Three steps: scores -> weights, weights -> fractional budgets (water-filling so
short chunks don't hoard budget they can't use), fractional -> integer budgets
that add up exactly.
"""
import numpy as np


def weights_from_scores(scores, mode="softmax", temperature=1.0):
    """Higher score -> larger weight. Weights sum to 1.

    softmax: z-score the retriever scores first so the temperature means the
             same thing for a cosine similarity and for a reranker logit.
    rank:    only the ordering matters, weight ~ (rank + 1) ** (-1 / T).
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 1:
        return np.ones(1)

    if mode == "softmax":
        spread = scores.std()
        if spread < 1e-8:                       # retriever gave everything the same score
            return np.full(n, 1.0 / n)
        z = (scores - scores.mean()) / spread / temperature
        z -= z.max()
        w = np.exp(z)
    elif mode == "rank":
        order = np.argsort(-scores, kind="stable")
        rank = np.empty(n)
        rank[order] = np.arange(n)
        w = (rank + 1.0) ** (-1.0 / temperature)
    else:
        raise ValueError(f"unknown weight mode: {mode}")
    return w / w.sum()


def allocate(total, lengths, weights, floor=0):
    """Split `total` kept tokens across chunks. Returns ints that sum to
    min(total, sum(lengths)), with 0 <= b_i <= lengths_i.

    Every chunk first gets min(floor, length) tokens. The rest is shared out
    proportionally to `weights`; whatever a chunk can't take (because it is
    shorter than its share) is handed to the others.
    """
    lengths = np.asarray(lengths, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    n = len(lengths)
    total = int(min(total, lengths.sum()))

    base = np.minimum(floor, lengths)
    if base.sum() > total:
        # budget is smaller than the floors: shrink the floors evenly
        base = base * (total / base.sum())
    alloc = base.copy()
    remaining = total - alloc.sum()

    for _ in range(n + 1):
        if remaining < 1e-9:
            break
        room = lengths - alloc
        active = room > 1e-9
        if not active.any():
            break
        w = np.where(active, weights, 0.0)
        if w.sum() < 1e-12:                     # e.g. oracle weights ran out of gold chunks
            w = active.astype(np.float64)
        share = remaining * w / w.sum()
        take = np.minimum(share, room)
        alloc += take
        remaining -= take.sum()

    return _round_to_total(alloc, lengths, total)


def _round_to_total(alloc, lengths, total):
    """Largest-remainder rounding that respects the per-chunk caps."""
    out = np.floor(alloc + 1e-9).astype(int)
    caps = lengths.astype(int)
    out = np.minimum(out, caps)
    leftover = total - out.sum()
    if leftover > 0:
        frac = alloc - out
        # prefer chunks with the biggest fractional part, but only if they have room
        order = np.argsort(-frac, kind="stable")
        while leftover > 0:
            progressed = False
            for i in order:
                if leftover == 0:
                    break
                if out[i] < caps[i]:
                    out[i] += 1
                    leftover -= 1
                    progressed = True
            if not progressed:
                break
    return out
