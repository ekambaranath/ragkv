"""Retrieval scores for chunks that came without one.

bm25 needs nothing but numpy. Any sentence-transformers cross-encoder name
(e.g. BAAI/bge-reranker-base) works too and is the more realistic reranker.
"""
import math
import re
from collections import Counter

import numpy as np


def _tokens(text):
    return re.findall(r"\w+", text.lower())


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        tokenized = [_tokens(d) for d in docs]
        self.avgdl = np.mean([len(t) for t in tokenized])
        df = Counter()
        for t in tokenized:
            df.update(set(t))
        n = len(tokenized)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def score(self, query, doc):
        terms = Counter(_tokens(doc))
        dl = sum(terms.values())
        s = 0.0
        for w in set(_tokens(query)):
            if w not in terms:
                continue
            tf = terms[w]
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += self.idf.get(w, 0.0) * tf * (self.k1 + 1) / denom
        return s


def add_scores(examples, retriever="bm25", batch_size=32):
    """Fill in chunk['score'] for every chunk, in place."""
    if retriever == "bm25":
        bm25 = BM25([c["text"] for ex in examples for c in ex["chunks"]])
        for ex in examples:
            for c in ex["chunks"]:
                c["score"] = bm25.score(ex["question"], c["text"])
        return examples

    from sentence_transformers import CrossEncoder
    model = CrossEncoder(retriever)
    pairs = [(ex["question"], c["text"]) for ex in examples for c in ex["chunks"]]
    flat = model.predict(pairs, batch_size=batch_size, show_progress_bar=True)
    it = iter(flat)
    for ex in examples:
        for c in ex["chunks"]:
            c["score"] = float(next(it))
    return examples
