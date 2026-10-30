"""Phase 1: is the retriever score actually a good guide for the KV budget?

For every example we run the prompt once with attention switched on and ask,
chunk by chunk:

  * how much attention do the question tokens pay to this chunk? (the thing
    attention-based eviction like SnapKV relies on)
  * how well does each cheap signal separate gold chunks from distractors?

Signals compared by AUROC (gold vs. non-gold, per example, then averaged):
  retriever   the score handed to us for free
  attn_mass   attention mass the question puts on the chunk (needs attention)
  attn_dens   same, per token
  neg_knorm   minus the mean key norm of the chunk (attention-free)
  length      chunk length (sanity check; should sit near 0.5)
  random

If `retriever` is not clearly above `neg_knorm` and close to `attn_mass`, the
paper's premise is in trouble and there is no point running the full sweep.

    python scripts/prior_analysis.py --model Qwen/Qwen2.5-1.5B-Instruct \\
        --data data/hotpot_bm25.jsonl --n 300 --out results/prior_qwen.json
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

from ragkv import cache_ops
from ragkv.data import load_examples, reorder_chunks
from ragkv.layout import build_layout
from ragkv.modeling import load_model
from ragkv.pipeline import prefill


def auroc(signal, gold):
    gold = np.asarray(gold, dtype=bool)
    n_pos, n_neg = gold.sum(), (~gold).sum()
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(signal)
    return (ranks[gold].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def chunk_signals(pre):
    layout = pre.layout
    n_ctx = layout.n_context

    # attention of the question tokens over context tokens, averaged over layers, heads, queries
    per_token = torch.zeros(n_ctx)
    for a in pre.attn:
        per_token += a[0, :, :, :n_ctx].float().mean(dim=(0, 1)).cpu()
    per_token /= len(pre.attn)

    # mean key norm per context token, averaged over layers and heads
    knorm = torch.zeros(n_ctx)
    n_layers = cache_ops.n_layers(pre.cache)
    for l in range(n_layers):
        k = cache_ops.layer_kv(pre.cache, l)[0][0, :, :n_ctx].float()
        knorm += k.norm(dim=-1).mean(dim=0).cpu()
    knorm /= n_layers

    mass = np.array([per_token[s:e].sum().item() for s, e in layout.spans])
    return {
        "retriever": layout.scores,
        "attn_mass": mass,
        "attn_dens": mass / layout.chunk_lengths,
        "neg_knorm": -np.array([knorm[s:e].mean().item() for s, e in layout.spans]),
        "length": layout.chunk_lengths.astype(float),
    }, mass


def bootstrap_ci(values, n_boot=2000, seed=0):
    values = np.asarray(values, dtype=float)
    rng = np.random.RandomState(seed)
    means = [values[rng.randint(len(values), size=len(values))].mean() for _ in range(n_boot)]
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--order", default="shuffle")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model, tok = load_model(args.model, args.dtype, args.device, attn="eager")
    examples = load_examples(args.data, args.n)
    rng = np.random.RandomState(args.seed)

    aucs = {k: [] for k in ["retriever", "attn_mass", "attn_dens", "neg_knorm", "length", "random"]}
    top1 = {k: [] for k in aucs}
    spear, gold_mass = [], []

    for i, ex in enumerate(examples):
        chunks = reorder_chunks(ex["chunks"], args.order, seed=args.seed * 100003 + i)
        layout = build_layout(tok, ex["question"], chunks)
        if not layout.gold.any() or layout.gold.all():
            continue
        pre = prefill(model, layout, need_attention=True)
        signals, mass = chunk_signals(pre)
        signals["random"] = rng.rand(len(chunks))

        for name, sig in signals.items():
            a = auroc(sig, layout.gold)
            if a is not None:
                aucs[name].append(a)
            top1[name].append(float(layout.gold[int(np.argmax(sig))]))
        if np.std(layout.scores) > 0 and np.std(mass) > 0:
            spear.append(spearmanr(layout.scores, mass)[0])
        gold_mass.append(mass[layout.gold].sum() / mass.sum())
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(examples)}", flush=True)

    report = {"n_examples": len(aucs["retriever"]), "model": args.model, "data": args.data, "signals": {}}
    print(f"\n{'signal':<11} {'AUROC (gold vs rest)':<28} top-1 is gold")
    for name in aucs:
        m, lo, hi = bootstrap_ci(aucs[name])
        t = float(np.mean(top1[name]))
        report["signals"][name] = {"auroc": m, "ci95": [lo, hi], "top1_gold": t}
        print(f"{name:<11} {m:.3f}  [{lo:.3f}, {hi:.3f}]{'':<8} {t:.3f}")
    m, lo, hi = bootstrap_ci(spear)
    report["spearman_retriever_vs_attention_mass"] = {"mean": m, "ci95": [lo, hi]}
    gm = bootstrap_ci(gold_mass)
    report["attention_mass_on_gold_chunks"] = {"mean": gm[0], "ci95": [gm[1], gm[2]]}
    print(f"\nSpearman(retriever score, attention mass): {m:.3f} [{lo:.3f}, {hi:.3f}]")
    print(f"share of question attention that lands on gold chunks: {gm[0]:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
