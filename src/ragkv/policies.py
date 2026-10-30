"""Which tokens of the retrieved chunks survive.

A policy is an allocator plus a scorer, written "allocator+scorer":

  allocator  how many tokens each chunk may keep
    global     no chunk structure, rank every chunk token against every other
    uniform    equal share per chunk
    retriever  share follows the retriever score         <- the idea of this repo
    oracle     share goes to the gold chunks (upper bound for a perfect retriever)

  scorer     which tokens to keep inside the allowed share
    knorm      lowest key L2 norm first (Devoto et al. 2024), needs no attention
    recency    most recent first
    random
    snapkv     attention paid by the question tokens, average-pooled (Li et al. 2024)

So "global+snapkv" is SnapKV, "global+recency" is StreamingLLM-style, and
"retriever+knorm" is ours. The prefix and the question are never evicted.
"""
import numpy as np
import torch
import torch.nn.functional as F

from . import cache_ops
from .allocation import allocate, weights_from_scores

ALLOCATORS = ("global", "uniform", "retriever", "oracle")
SCORERS = ("knorm", "recency", "random", "snapkv")

ALIASES = {
    "streaming": "global+recency",
    "snapkv": "global+snapkv",
    "knorm": "global+knorm",
    "random": "global+random",
    "ours": "retriever+knorm",
}


class Policy:
    def __init__(self, name, temperature=1.0, mode="softmax", floor=8, pool=7):
        name = ALIASES.get(name, name)
        self.name = name
        self.temperature = temperature
        self.mode = mode
        self.floor = floor
        self.pool = pool

        self.is_full = name == "full"
        if self.is_full:
            self.allocator = self.scorer = None
        else:
            try:
                self.allocator, self.scorer = name.split("+")
            except ValueError:
                raise ValueError(f"policy must look like 'allocator+scorer' or 'full', got {name!r}")
            if self.allocator not in ALLOCATORS or self.scorer not in SCORERS:
                raise ValueError(f"unknown policy {name!r}; allocators={ALLOCATORS} scorers={SCORERS}")

    @property
    def needs_attention(self):
        return self.scorer == "snapkv"

    # ---- budgets -------------------------------------------------------

    def chunk_budgets(self, layout, total):
        """Per-chunk token budgets, or None for the chunk-agnostic global allocator."""
        lengths = layout.chunk_lengths
        if self.allocator == "global":
            return None
        if self.allocator == "uniform":
            return allocate(total, lengths, np.ones(len(lengths)))
        if self.allocator == "retriever":
            w = weights_from_scores(layout.scores, self.mode, self.temperature)
            return allocate(total, lengths, w, floor=self.floor)
        if self.allocator == "oracle":
            w = layout.gold.astype(float)
            if w.sum() == 0:
                w = np.ones(len(lengths))
            return allocate(total, lengths, w, floor=self.floor)

    # ---- selection -----------------------------------------------------

    def select(self, cache, attn, layout, ratio, seed=0):
        """Returns (keep, budgets). keep[layer] is a (n_kv_heads, n_keep) tensor
        of sorted cache positions."""
        n_layers = cache_ops.n_layers(cache)
        T = cache_ops.seq_length(cache)
        n_heads = cache_ops.layer_kv(cache, 0)[0].shape[1]
        total = min(int(round(ratio * layout.n_chunk_tokens)), layout.n_chunk_tokens)

        if self.is_full or total == layout.n_chunk_tokens:
            everything = torch.arange(T).expand(n_heads, -1)
            return [everything] * n_layers, None

        budgets = self.chunk_budgets(layout, total)
        protected = torch.cat([torch.arange(layout.n_prefix), torch.arange(layout.n_context, T)])
        gen = torch.Generator().manual_seed(seed)

        keep = []
        for layer in range(n_layers):
            k = cache_ops.layer_kv(cache, layer)[0]
            attn_l = attn[layer] if attn is not None else None
            scores = self._token_scores(k, attn_l, layout, gen)
            picked = self._pick(scores, layout, budgets, total)
            prot = protected.to(picked.device).expand(n_heads, -1)
            keep.append(torch.cat([prot, picked], dim=1).sort(dim=1).values)
        return keep, budgets

    def _token_scores(self, k, attn_l, layout, gen):
        """(n_kv_heads, n_context) score per context token, higher = keep."""
        n_ctx = layout.n_context
        h = k.shape[1]
        if self.scorer == "knorm":
            return -k[0, :, :n_ctx].float().norm(dim=-1)
        if self.scorer == "recency":
            return torch.arange(n_ctx, device=k.device, dtype=torch.float32).expand(h, -1)
        if self.scorer == "random":
            return torch.rand(h, n_ctx, generator=gen).to(k.device)
        if self.scorer == "snapkv":
            if attn_l is None:
                raise RuntimeError("snapkv scoring needs attention weights; load the model with attn='eager'")
            a = attn_l[0, :, :, :n_ctx].float()            # (n_heads, q, n_ctx)
            group = a.shape[0] // h                        # query heads per kv head (GQA)
            a = a.reshape(h, group, a.shape[1], n_ctx).mean(dim=(1, 2))
            a = F.avg_pool1d(a[None], self.pool, stride=1, padding=self.pool // 2,
                             count_include_pad=False)[0]
            return a

    def _pick(self, scores, layout, budgets, total):
        h = scores.shape[0]
        if total == 0:
            return torch.empty(h, 0, dtype=torch.long, device=scores.device)
        if budgets is None:
            pos = layout.chunk_positions().to(scores.device)
            top = scores[:, pos].topk(total, dim=1).indices
            return pos[top]
        parts = []
        for (start, end), b in zip(layout.spans, budgets):
            if b > 0:
                parts.append(scores[:, start:end].topk(int(b), dim=1).indices + start)
        return torch.cat(parts, dim=1)
