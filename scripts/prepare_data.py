"""Build a scored JSONL of retrieval-QA examples.

    python scripts/prepare_data.py --source synthetic --n 200 --out data/syn.jsonl
    python scripts/prepare_data.py --source hotpot --n 500 --retriever bm25 --out data/hotpot_bm25.jsonl
    python scripts/prepare_data.py --source hotpot --n 500 --retriever BAAI/bge-reranker-base --out data/hotpot_bge.jsonl
"""
import argparse
import os

import numpy as np

from ragkv.data import save_examples
from ragkv.data.retrievers import add_scores
from ragkv.data.synthetic import make_dataset


def retriever_report(examples):
    """How good are the scores we are about to build on?"""
    top1, rr, both_in_top2 = [], [], []
    for ex in examples:
        gold = np.array([c["gold"] for c in ex["chunks"]])
        if not gold.any():
            continue
        order = np.argsort(-np.array([c["score"] for c in ex["chunks"]]), kind="stable")
        ranks = np.flatnonzero(gold[order])
        top1.append(gold[order[0]])
        rr.append(1.0 / (ranks[0] + 1))
        if gold.sum() >= 2:
            both_in_top2.append(gold[order[:2]].sum() == min(2, gold.sum()))
    print(f"retriever: top-1 is gold {np.mean(top1):.3f} | MRR {np.mean(rr):.3f}"
          + (f" | all of 2 gold in top-2 {np.mean(both_in_top2):.3f}" if both_in_top2 else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["hotpot", "synthetic"], default="hotpot")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--retriever", default="bm25", help="bm25, or a sentence-transformers cross-encoder name (hotpot only)")
    p.add_argument("--k", type=int, default=8, help="chunks per example (synthetic)")
    p.add_argument("--filler", type=int, default=4, help="filler sentences per chunk (synthetic)")
    p.add_argument("--noise", type=float, default=1.0, help="retriever noise (synthetic)")
    p.add_argument("--signal", type=float, default=2.0, help="score bump of the gold chunk (synthetic)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if args.source == "synthetic":
        examples = make_dataset(args.n, args.k, args.filler, args.noise, args.signal, args.seed)
    else:
        from ragkv.data.hotpot import load_hotpot
        examples = load_hotpot(args.n, seed=args.seed)
        add_scores(examples, args.retriever)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_examples(examples, args.out)
    print(f"wrote {len(examples)} examples to {args.out}")
    retriever_report(examples)


if __name__ == "__main__":
    main()
