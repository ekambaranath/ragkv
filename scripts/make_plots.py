"""Turn run_eval.py output into the figure and tables for the paper.

    python scripts/make_plots.py --results results/hotpot_qwen.jsonl --out results/hotpot_qwen

Writes <out>_f1.png, <out>_em.png and <out>_summary.md. Every number in the
plots is also in the markdown tables. Confidence intervals are bootstrap
over examples (2000 resamples). The figure shades the interval for our method
only, to stay readable; the tables carry every interval. The paired deltas
bootstrap the per-example difference, which is what to look at when deciding
whether a method is actually better.

The x-axis is the share of the whole prompt's KV cache that is kept (prefix and
question included). The tables are indexed by the budget `ratio`, the share of
the chunk tokens kept, which is what run_eval.py sweeps.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# colour follows the policy, not its rank in whatever subset is on screen
SLOTS = {
    "retriever+knorm": ("#2a78d6", "o", "retriever + key-norm (ours)"),
    "global+snapkv":   ("#eb6834", "s", "SnapKV"),
    "global+knorm":    ("#1baf7a", "^", "key-norm (global)"),
    "uniform+knorm":   ("#eda100", "D", "uniform + key-norm"),
    "global+recency":  ("#e87ba4", "v", "StreamingLLM-style"),
    "global+random":   ("#008300", "P", "random"),
    "oracle+knorm":    ("#6250d6", "X", "oracle + key-norm"),
    "retriever+snapkv": ("#e34948", "h", "retriever + SnapKV"),
}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
OURS = "retriever+knorm"


def load(paths, tag=None):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(line) for line in f]
    df = pd.DataFrame(rows)
    if tag is not None:
        return df[df["tag"] == tag]
    # runs that differ only in their seed are replicates: keep them all, per_example() averages them
    configs = sorted({t.rsplit("|seed", 1)[0] for t in df["tag"].unique()})
    if len(configs) > 1:
        raise SystemExit("results mix several configs, pick one with --tag:\n  "
                         + "\n  ".join(sorted(df["tag"].unique())))
    return df


def per_example(df):
    """Average over seeds so every example counts once per (policy, ratio)."""
    cols = ["em", "f1", "kv_mb", "kept_frac", "select_ms", "decode_ms_per_tok"]
    return df.groupby(["policy", "ratio", "id"], as_index=False)[cols].mean()


def boot(values, n_boot=2000, seed=0):
    values = np.asarray(values, dtype=float)
    rng = np.random.RandomState(seed)
    idx = rng.randint(len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return values.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def curve(pe, policy, metric):
    sub = pe[pe.policy == policy]
    out = []
    for ratio, g in sub.groupby("ratio"):
        m, lo, hi = boot(g[metric])
        out.append((ratio, g["kept_frac"].mean(), m, lo, hi))
    return np.array(out)


def plot_metric(pe, metric, path):
    label = {"f1": "F1", "em": "exact match"}[metric]
    fig, ax = plt.subplots(figsize=(9.0, 4.6), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    present = set(pe.policy.unique()) - {"full"}
    unknown = sorted(present - set(SLOTS))
    if unknown:
        raise SystemExit(f"no colour slot for {unknown}; add them to SLOTS")
    policies = [p for p in SLOTS if p in present]        # legend follows SLOTS order, ours first

    for pol in policies:
        color, marker, name = SLOTS[pol]
        c = curve(pe, pol, metric)
        if pol == OURS:         # one band keeps the figure readable; every CI is in the summary table
            ax.fill_between(c[:, 1], c[:, 3], c[:, 4], color=color, alpha=0.14, linewidth=0)
        ax.plot(c[:, 1], c[:, 2], color=color, lw=1.8, marker=marker, ms=6.5,
                markeredgecolor=SURFACE, markeredgewidth=1.4, label=name,
                zorder=3 if pol == OURS else 2)

    if "full" in set(pe.policy):
        m, lo, hi = boot(pe[pe.policy == "full"][metric])
        ax.axhline(m, color=INK_2, lw=1.2, ls=(0, (5, 3)), zorder=1, label="full cache")

    ax.set_xlabel("fraction of the prompt's KV cache kept", color=INK_2)
    ax.set_ylabel(label, color=INK_2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def summary(pe):
    ratios = sorted(r for r in pe.ratio.unique() if r < 1.0)
    policies = [p for p in pe.policy.unique() if p != "full"]
    n = pe.id.nunique()
    out = [f"{n} examples. Cells are mean [95% bootstrap CI].\n"]

    for metric, title in (("f1", "F1"), ("em", "Exact match")):
        rows = []
        if "full" in set(pe.policy):
            m, lo, hi = boot(pe[pe.policy == "full"][metric])
            rows.append(["full cache"] + [f"{m:.3f} [{lo:.3f}, {hi:.3f}]"] * len(ratios))
        for pol in policies:
            cells = []
            for r in ratios:
                g = pe[(pe.policy == pol) & (pe.ratio == r)][metric]
                m, lo, hi = boot(g) if len(g) else (np.nan,) * 3
                cells.append(f"{m:.3f} [{lo:.3f}, {hi:.3f}]")
            rows.append([SLOTS.get(pol, (0, 0, pol))[2]] + cells)
        out += [f"### {title} by fraction of chunk tokens kept\n",
                md_table(["policy"] + [f"{r:g}" for r in ratios], rows), ""]

    if OURS in policies:
        rows = []
        for pol in policies:
            if pol == OURS:
                continue
            cells = []
            for r in ratios:
                a = pe[(pe.policy == OURS) & (pe.ratio == r)].set_index("id")["f1"]
                b = pe[(pe.policy == pol) & (pe.ratio == r)].set_index("id")["f1"]
                d = (a - b).dropna()
                m, lo, hi = boot(d) if len(d) else (np.nan,) * 3
                star = "*" if lo > 0 or hi < 0 else ""
                cells.append(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}")
            rows.append([SLOTS.get(pol, (0, 0, pol))[2]] + cells)
        out += ["### Paired F1 difference, ours minus baseline (* = CI excludes 0)\n",
                md_table(["baseline"] + [f"{r:g}" for r in ratios], rows), ""]

    rows = []
    for pol in ["full"] + policies:
        if pol not in set(pe.policy):
            continue
        g = pe[pe.policy == pol]
        rows.append([SLOTS.get(pol, (0, 0, pol))[2] if pol != "full" else "full cache",
                     f"{g.kv_mb.mean():.1f}", f"{g.select_ms.mean():.2f}", f"{g.decode_ms_per_tok.mean():.2f}"])
    out += ["### Cost (averaged over all budgets shown for the policy)\n",
            md_table(["policy", "KV cache MB", "select+prune ms", "decode ms/token"], rows), ""]
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--out", required=True, help="output prefix")
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    pe = per_example(load(args.results, args.tag))
    plot_metric(pe, "f1", args.out + "_f1.png")
    plot_metric(pe, "em", args.out + "_em.png")
    with open(args.out + "_summary.md", "w") as f:
        f.write(summary(pe))
    print("wrote", args.out + "_f1.png,", args.out + "_em.png,", args.out + "_summary.md")


if __name__ == "__main__":
    main()
