"""Prefill -> compress -> decode.

The prompt is run in two chunks so we can look at the question's attention
before deciding what to evict, and so the expensive part is shared between
policies:

  stage 1  context (prefix + chunks) through the model          -> full cache
  stage 2  question, minus its last token, against the full cache  (attention
           weights are captured here if the policy wants them)
  select   pick the positions to keep, build a smaller cache
  stage 3  feed the last prompt token to the *compressed* cache -> first answer token
  decode   greedy, until newline / eos / max_new_tokens

Holding back the last prompt token matters: otherwise the first answer token
would be predicted from the full cache and every policy would look better than
it is (for short QA answers that first token is most of the answer).

New tokens keep counting positions from the original prompt length, so the
rotary phase of kept keys and fresh queries stays consistent.
"""
import time
from dataclasses import dataclass

import torch

from . import cache_ops


@dataclass
class Prefilled:
    layout: object
    cache: object          # full cache: context + question[:-1]
    attn: object           # per-layer attention of the question tokens, or None
    last_id: int
    last_pos: int
    prefill_ms: float


def _now(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def eos_ids(model, tok):
    ids = set()
    e = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(e, int):
        ids.add(e)
    elif e:
        ids.update(e)
    if getattr(tok, "eos_token_id", None) is not None:
        ids.add(tok.eos_token_id)
    return ids


@torch.no_grad()
def prefill(model, layout, need_attention=False):
    dev = next(model.parameters()).device
    ids = layout.ids.to(dev)
    n_ctx = layout.n_context

    t0 = _now(dev)
    out = model.model(input_ids=ids[:, :n_ctx], use_cache=True)
    cache = out.past_key_values

    q_ids = ids[:, n_ctx:-1]
    pos = torch.arange(n_ctx, n_ctx + q_ids.shape[1], device=dev)[None]
    out = model.model(input_ids=q_ids, past_key_values=cache, position_ids=pos,
                      use_cache=True, output_attentions=need_attention)
    cache = out.past_key_values
    attn = out.attentions if need_attention else None
    if need_attention and attn is None:
        raise RuntimeError("model returned no attention weights; load it with attn='eager'")
    ms = (_now(dev) - t0) * 1000

    return Prefilled(layout, cache, attn, int(ids[0, -1]), layout.total - 1, ms)


@torch.no_grad()
def decode(model, tok, cache, last_id, last_pos, max_new_tokens=32):
    dev = next(model.parameters()).device
    stop = eos_ids(model, tok)
    cur = torch.tensor([[last_id]], device=dev)
    pos = last_pos
    out_ids = []

    for _ in range(max_new_tokens):
        mask = torch.ones(1, cache_ops.seq_length(cache) + 1, dtype=torch.long, device=dev)
        h = model.model(input_ids=cur, past_key_values=cache, attention_mask=mask,
                        position_ids=torch.tensor([[pos]], device=dev), use_cache=True)
        nxt = int(model.lm_head(h.last_hidden_state[:, -1]).argmax(-1))
        if nxt in stop:
            break
        out_ids.append(nxt)
        # answers are one line; stop at the first newline after some content
        if "\n" in tok.decode(out_ids, skip_special_tokens=True).lstrip():
            break
        cur = torch.tensor([[nxt]], device=dev)
        pos += 1
    return out_ids


@torch.no_grad()
def run_policy(model, tok, pre, policy, ratio, seed=0, max_new_tokens=32):
    """Compress a prefilled prompt with `policy` and answer from the small cache.

    The prefilled cache is left untouched, so many (policy, ratio) pairs can
    reuse one prefill.
    """
    dev = next(model.parameters()).device

    t0 = _now(dev)
    keep, budgets = policy.select(pre.cache, pre.attn, pre.layout, ratio, seed)
    small = cache_ops.gather(pre.cache, keep)
    select_ms = (_now(dev) - t0) * 1000

    kv_mb = cache_ops.kv_bytes(small) / 2**20
    kept = int(keep[0].shape[1])

    t0 = _now(dev)
    out_ids = decode(model, tok, small, pre.last_id, pre.last_pos, max_new_tokens)
    decode_ms = (_now(dev) - t0) * 1000

    text = tok.decode(out_ids, skip_special_tokens=True).strip().split("\n")[0].strip()
    return {
        "text": text,
        "n_generated": len(out_ids),
        "kept_tokens": kept,
        "kept_frac": kept / pre.layout.total,
        "kv_mb": kv_mb,
        "select_ms": select_ms,
        "decode_ms_per_tok": decode_ms / max(1, len(out_ids) + 1),
        "budgets": None if budgets is None else [int(b) for b in budgets],
    }
