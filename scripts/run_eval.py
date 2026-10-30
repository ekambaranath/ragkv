"""Sweep policies x KV budgets over a scored dataset.

    python scripts/run_eval.py --model Qwen/Qwen2.5-1.5B-Instruct --data data/hotpot_bm25.jsonl \\
        --policies full global+knorm uniform+knorm retriever+knorm snapkv \\
        --ratios 0.1 0.2 0.3 0.5 --n 500 --out results/hotpot_qwen.jsonl

One JSON line per (example, policy, ratio). Re-running the same command skips
what is already in the output file, so a dead Colab session costs nothing.
`ratio` is the fraction of the *chunk* tokens that is kept; the prefix and the
question are always kept on top of that.
"""
import argparse
import json
import os
import time

from ragkv import metrics
from ragkv.data import load_examples, reorder_chunks
from ragkv.layout import build_layout
from ragkv.modeling import load_model
from ragkv.pipeline import prefill, run_policy
from ragkv.policies import Policy


def read_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["id"], r["policy"], r["ratio"], r["tag"]))
    return done


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model name, or tiny-random for an offline smoke test")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--policies", nargs="+", default=["full", "global+knorm", "uniform+knorm", "retriever+knorm"])
    p.add_argument("--ratios", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5])
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--order", default="shuffle", choices=["shuffle", "rank", "reverse_rank", "original"])
    p.add_argument("--temperature", type=float, default=1.0, help="softmax temperature of the retriever prior")
    p.add_argument("--weights", default="softmax", choices=["softmax", "rank"])
    p.add_argument("--floor", type=int, default=8, help="minimum tokens every chunk keeps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--device", default=None)
    p.add_argument("--attn", default=None, help="eager or sdpa; default picks eager only if a policy needs attention")
    args = p.parse_args()

    policies = [Policy(n, args.temperature, args.weights, args.floor) for n in args.policies]
    attn = args.attn or ("eager" if any(pol.needs_attention for pol in policies) else "sdpa")
    model, tok = load_model(args.model, args.dtype, args.device, attn)

    examples = load_examples(args.data, args.n)
    tag = f"{args.order}|{args.weights}|T{args.temperature}|floor{args.floor}|seed{args.seed}"
    done = read_done(args.out)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    start = time.time()
    with open(args.out, "a") as out:
        for i, ex in enumerate(examples):
            jobs = []
            for pol in policies:
                for r in ([1.0] if pol.is_full else args.ratios):
                    if (ex["id"], pol.name, r, tag) not in done:
                        jobs.append((pol, r))
            if not jobs:
                continue

            chunks = reorder_chunks(ex["chunks"], args.order, seed=args.seed * 100003 + i)
            layout = build_layout(tok, ex["question"], chunks)
            pre = prefill(model, layout, need_attention=any(pol.needs_attention for pol, _ in jobs))

            for pol, r in jobs:
                res = run_policy(model, tok, pre, pol, r, seed=args.seed * 100003 + i,
                                 max_new_tokens=args.max_new_tokens)
                rec = {
                    "id": ex["id"], "policy": pol.name, "ratio": r, "tag": tag,
                    "order": args.order, "seed": args.seed, "temperature": args.temperature,
                    "weights": args.weights, "floor": args.floor,
                    **metrics.score(res["text"], ex["answers"]),
                    "pred": res["text"], "answers": ex["answers"],
                    "n_prompt_tokens": layout.total, "n_chunk_tokens": layout.n_chunk_tokens,
                    "prefill_ms": pre.prefill_ms,
                    **{k: v for k, v in res.items() if k != "text"},
                }
                out.write(json.dumps(rec) + "\n")
            out.flush()

            if (i + 1) % 10 == 0:
                rate = (i + 1) / (time.time() - start)
                print(f"{i + 1}/{len(examples)} examples  ({rate:.2f}/s)", flush=True)

    print("done ->", args.out)


if __name__ == "__main__":
    main()
