"""Draw the three result figures used in the README.

The numbers below are copied from the evaluation runs, so the figures can be
redrawn without a GPU:

  held-out F1 and paired differences   results/qwen_bge_heldout.jsonl  (questions 200-399)
  gold-text coverage                   data/hotpot_bm25.jsonl, data/hotpot_bge.jsonl  (first 200 questions)

    python docs/make_result_figures.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
os.makedirs(OUT, exist_ok=True)

BLUE = "#2a78d6"
LIGHT = "#b7d3f6"
GREY = "#8a8984"
PALE = "#d9d8d3"
AMBER = "#c98500"
INK = "#1b1b1a"

plt.rcParams.update({
    "font.size": 10.5,
    "axes.edgecolor": GREY,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# ---------------------------------------------------------------- figure 1
# paired F1 differences on the held-out questions, mean [95% CI], n = 200
ROWS = [
    ("Retriever-weighted split vs equal split\n(both pick tokens at random)",
     (0.087, 0.041, 0.134), (0.045, 0.005, 0.086)),
    ("Retriever-weighted split + SnapKV\nvs SnapKV alone",
     (0.047, 0.008, 0.088), (0.018, -0.014, 0.048)),
    ("Retriever-weighted split + SnapKV\nvs the full cache",
     (0.002, -0.038, 0.041), (0.000, -0.023, 0.024)),
    ("Perfect split from gold labels vs equal split\n(upper bound, for reference)",
     (0.149, 0.097, 0.201), (0.059, 0.019, 0.102)),
]


def forest():
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    n = len(ROWS)
    for i, (label, a, b) in enumerate(ROWS):
        y = n - 1 - i
        for off, (m, lo, hi), color, name in [(0.14, a, BLUE, "20% of chunk tokens kept"),
                                              (-0.14, b, AMBER, "50% of chunk tokens kept")]:
            ax.plot([lo, hi], [y + off, y + off], color=color, lw=2.4, solid_capstyle="round")
            ax.plot(m, y + off, "o", color=color, ms=7, label=name if i == 0 else None)
            ax.text(hi + 0.004, y + off, f"{m:+.3f}", va="center", fontsize=9, color=color)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in ROWS][::-1])
    ax.set_xlabel("difference in answer F1  (right of the line = better)")
    ax.set_xlim(-0.06, 0.24)
    ax.set_ylim(-0.6, n - 0.4)
    ax.grid(axis="x", color="#ececea")
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, frameon=False)
    ax.set_title("Weighting the KV budget by retriever score, 200 held-out questions\n"
                 "dot = mean, bar = 95% bootstrap interval, paired by question",
                 loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "results_paired.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
FULL = 0.393
POLICIES = [
    ("Equal split\n+ random", (0.272, 0.357), PALE),
    ("Retriever split\n+ random", (0.359, 0.402), LIGHT),
    ("SnapKV\nalone", (0.348, 0.376), PALE),
    ("Retriever split\n+ SnapKV", (0.395, 0.394), BLUE),
    ("Gold-label split\n+ random", (0.420, 0.416), GREY),
]


def f1_bars():
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), sharey=True)
    for k, (ax, title) in enumerate(zip(axes, ["Keep 20% of chunk tokens", "Keep 50% of chunk tokens"])):
        names = [p[0] for p in POLICIES]
        vals = [p[1][k] for p in POLICIES]
        cols = [p[2] for p in POLICIES]
        bars = ax.bar(range(len(vals)), vals, color=cols, edgecolor=GREY, linewidth=0.8, width=0.72)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=9)
        ax.axhline(FULL, color=AMBER, ls="--", lw=1.4)
        ax.text(-0.45, FULL + 0.008, "full cache 0.393", color=AMBER, ha="left", fontsize=9)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(names, fontsize=8.4)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylim(0, 0.5)
        ax.grid(axis="y", color="#ececea")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("answer F1")
    fig.suptitle("Answer quality on 200 held-out HotpotQA questions (Qwen2.5-1.5B-Instruct)",
                 x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "results_f1.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
# share of the gold paragraphs' tokens that receive budget, first 200 questions, tau = 0.5
COVER = [
    ("Equal split", (0.27, 0.63), PALE),
    ("Weighted by BM25", (0.54, 0.80), LIGHT),
    ("Weighted by bge reranker", (0.69, 0.97), BLUE),
    ("Gold-label split", (0.85, 1.00), GREY),
]


def coverage():
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    w = 0.19
    for j, (name, vals, color) in enumerate(COVER):
        xs = [0 + (j - 1.5) * w, 1 + (j - 1.5) * w]
        bars = ax.bar(xs, vals, width=w * 0.92, color=color, edgecolor=GREY, linewidth=0.8, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["20% of chunk tokens kept", "50% of chunk tokens kept"])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("share of the supporting paragraphs'\ntokens that get budget")
    ax.grid(axis="y", color="#ececea")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9)
    ax.set_title("A better reranker puts the budget where the evidence is", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "results_coverage.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    forest()
    f1_bars()
    coverage()
    print("wrote", ", ".join(["results_paired.png", "results_f1.png", "results_coverage.png"]), "->", OUT)
