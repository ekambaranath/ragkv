"""Paired F1 differences between two policies, with a bootstrap interval.

Every question is scored under both policies, so the comparison is made
question by question. That is much tighter than comparing two averages.

    python scripts/paired_diff.py --results results/qwen_bge_heldout.jsonl \\
        --pair retriever+random:uniform+random \\
        --pair retriever+snapkv:global+snapkv \\
        --pair retriever+snapkv:full \\
        --ratios 0.2 0.5

A pair is "a:b" and the printed number is mean(F1 of a) - mean(F1 of b).
Use policy names as they appear in the results file (aliases are stored
resolved, so write global+snapkv, not snapkv). When b is "full", it is
compared at ratio 1.0, the only budget the full cache has.
"""
import argparse

import numpy as np
import pandas as pd


def paired(per_example, a, ra, b, rb, n_boot=5000, seed=0):
    for key in [(a, ra), (b, rb)]:
        if key not in per_example.columns:
            have = sorted({f"{p} @ {r}" for p, r in per_example.columns})
            raise SystemExit(f"{key[0]} @ {key[1]} is not in the results. Available: {', '.join(have)}")
    d = (per_example[(a, ra)] - per_example[(b, rb)]).dropna().values
    if len(d) == 0:
        raise SystemExit(f"no questions have both {a} @ {ra} and {b} @ {rb}")
    rng = np.random.RandomState(seed)
    means = [d[rng.randint(len(d), size=len(d))].mean() for _ in range(n_boot)]
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5), len(d)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", nargs="+", required=True, help="one or more run_eval.py output files")
    p.add_argument("--pair", action="append", required=True, help="a:b, repeat for several pairs")
    p.add_argument("--ratios", nargs="+", type=float, default=[0.2, 0.5])
    p.add_argument("--tag", default=None, help="keep only rows with this tag (needed if files mix settings)")
    args = p.parse_args()

    df = pd.concat([pd.read_json(path, lines=True) for path in args.results], ignore_index=True)
    if args.tag is not None:
        df = df[df["tag"] == args.tag]
    per_example = df.groupby(["id", "policy", "ratio"]).f1.mean().unstack(["policy", "ratio"])

    for spec in args.pair:
        a, b = spec.split(":")
        for r in args.ratios:
            rb = 1.0 if b == "full" else r
            m, lo, hi, n = paired(per_example, a, r, b, rb)
            print(f"{a} - {b} @ {r}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  (n={n})")


if __name__ == "__main__":
    main()
