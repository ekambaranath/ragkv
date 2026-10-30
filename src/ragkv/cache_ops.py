"""Small helpers for poking at a transformers KV cache.

The internal layout of DynamicCache has moved around between releases, so all
the version-specific access lives here and nothing else in the repo touches it.
Only `cache.update(k, v, layer)` and `cache.get_seq_length()` are used as the
stable public surface.
"""
import torch
from transformers import DynamicCache


def n_layers(cache):
    return len(cache)


def seq_length(cache):
    return cache.get_seq_length()


def layer_kv(cache, i):
    """(keys, values) of layer i, each shaped (1, n_kv_heads, seq, head_dim)."""
    if hasattr(cache, "layers"):
        layer = cache.layers[i]
        return layer.keys, layer.values
    if hasattr(cache, "key_cache"):
        return cache.key_cache[i], cache.value_cache[i]
    k, v = cache[i]
    return k, v


def build_cache(pairs):
    cache = DynamicCache()
    for i, (k, v) in enumerate(pairs):
        cache.update(k, v, i)
    return cache


def gather(cache, keep):
    """New cache holding only the positions in keep[layer], a (n_kv_heads, n) long tensor.

    Each kv head can keep a different set of positions. Keys stay rotated by the
    position they were computed at, so nothing needs re-encoding.
    """
    pairs = []
    for i, idx in enumerate(keep):
        k, v = layer_kv(cache, i)
        ix = idx.to(k.device)[None, :, :, None].expand(1, -1, -1, k.shape[-1])
        pairs.append((k.gather(2, ix), v.gather(2, ix)))
    return build_cache(pairs)


def kv_bytes(cache):
    total = 0
    for i in range(n_layers(cache)):
        k, v = layer_kv(cache, i)
        total += (k.numel() + v.numel()) * k.element_size()
    return total
