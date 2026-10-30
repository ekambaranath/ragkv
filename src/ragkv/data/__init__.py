"""Examples are plain dicts, stored as JSON lines:

    {"id": ..., "question": ..., "answers": [...],
     "chunks": [{"text": ..., "score": <retriever score>, "gold": <bool>}, ...]}

`score` comes from whatever retriever sits in front of the LLM. `gold` marks the
chunks that actually contain the answer; only the oracle policy and the
analysis scripts look at it.
"""
import json

import numpy as np


def save_examples(examples, path):
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def load_examples(path, n=None):
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
            if n and len(out) >= n:
                break
    return out


def reorder_chunks(chunks, how="shuffle", seed=0):
    """Chunk order in the prompt.

    shuffle        random but reproducible (default; keeps position from
                   being confounded with relevance)
    rank           best retriever score first
    reverse_rank   best retriever score last, right next to the question
    original       whatever order the dataset gave
    """
    chunks = list(chunks)
    if how == "original":
        return chunks
    if how == "rank":
        return sorted(chunks, key=lambda c: -c["score"])
    if how == "reverse_rank":
        return sorted(chunks, key=lambda c: c["score"])
    if how == "shuffle":
        perm = np.random.RandomState(seed).permutation(len(chunks))
        return [chunks[i] for i in perm]
    raise ValueError(f"unknown order: {how}")
