"""HotpotQA (distractor setting) as retrieval QA.

Each question comes with 10 paragraphs: the 2 that hold the supporting facts
plus 8 distractors. We treat the 10 as the retrieved set; scores are added
afterwards by a retriever (see retrievers.add_scores).
"""
import numpy as np


def load_hotpot(n=500, split="validation", seed=0):
    from datasets import load_dataset

    try:
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    except Exception:
        # newer `datasets` releases refuse script-based datasets; the Hub keeps a parquet mirror
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, revision="refs/convert/parquet")
    picks = np.random.RandomState(seed).permutation(len(ds))[:n]

    examples = []
    for i in picks:
        row = ds[int(i)]
        gold_titles = set(row["supporting_facts"]["title"])
        chunks = []
        for title, sents in zip(row["context"]["title"], row["context"]["sentences"]):
            chunks.append({
                "text": f"{title}. {' '.join(s.strip() for s in sents)}",
                "score": 0.0,
                "gold": title in gold_titles,
            })
        examples.append({
            "id": row["id"],
            "question": row["question"],
            "answers": [row["answer"]],
            "chunks": chunks,
        })
    return examples
