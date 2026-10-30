"""How a RAG prompt is laid out in token space.

The whole method depends on knowing which token positions belong to which
retrieved chunk, so we tokenize piece by piece and remember the spans instead
of tokenizing one big string and trying to recover them afterwards.

    [bos] instructions | [1] chunk | [2] chunk | ... | [K] chunk | question
    '----- prefix -----' '------- compressible -------' '- protected -'
"""
from dataclasses import dataclass

import numpy as np
import torch

PREFIX = "Answer the question using only the documents below. Reply with a short answer.\n\n"


@dataclass
class Layout:
    ids: torch.Tensor      # (1, T) the full prompt
    n_prefix: int          # tokens before the first chunk (bos + instructions)
    spans: list            # [(start, end)] per chunk, in prompt order
    n_context: int         # prefix + every chunk; the question starts here
    scores: np.ndarray     # retriever score per chunk, prompt order
    gold: np.ndarray       # bool per chunk. Only the oracle policy and the analysis read this

    @property
    def total(self):
        return self.ids.shape[1]

    @property
    def n_question(self):
        return self.total - self.n_context

    @property
    def chunk_lengths(self):
        return np.array([e - s for s, e in self.spans])

    @property
    def n_chunk_tokens(self):
        return int(self.chunk_lengths.sum())

    def chunk_positions(self):
        """All token positions that belong to some chunk, in order."""
        return torch.cat([torch.arange(s, e) for s, e in self.spans])


def _encode(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def build_layout(tok, question, chunks, prefix=PREFIX):
    """chunks: list of dicts with 'text', 'score' and optionally 'gold'."""
    ids = []
    bos = getattr(tok, "bos_token_id", None)
    if bos is not None:
        ids.append(bos)
    ids += _encode(tok, prefix)
    n_prefix = len(ids)

    spans = []
    for i, chunk in enumerate(chunks):
        piece = _encode(tok, f"[{i + 1}] {chunk['text'].strip()}\n")
        spans.append((len(ids), len(ids) + len(piece)))
        ids += piece
    n_context = len(ids)

    ids += _encode(tok, f"\nQuestion: {question.strip()}\nShort answer:")

    return Layout(
        ids=torch.tensor([ids], dtype=torch.long),
        n_prefix=n_prefix,
        spans=spans,
        n_context=n_context,
        scores=np.array([float(c["score"]) for c in chunks]),
        gold=np.array([bool(c.get("gold", False)) for c in chunks]),
    )
