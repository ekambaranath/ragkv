import numpy as np
import pytest

from ragkv import metrics
from ragkv.data import load_examples, reorder_chunks, save_examples
from ragkv.data.retrievers import BM25, add_scores
from ragkv.data.synthetic import make_dataset
from ragkv.layout import build_layout
from ragkv.tiny import CharTokenizer


# ---- layout ---------------------------------------------------------------

def test_spans_tile_the_context_without_gaps(examples):
    tok = CharTokenizer()
    ex = examples[0]
    lay = build_layout(tok, ex["question"], ex["chunks"])
    assert lay.spans[0][0] == lay.n_prefix
    for (s0, e0), (s1, e1) in zip(lay.spans, lay.spans[1:]):
        assert e0 == s1
    assert lay.spans[-1][1] == lay.n_context
    assert lay.total > lay.n_context
    assert lay.n_chunk_tokens == lay.n_context - lay.n_prefix


def test_chunk_tokens_decode_back_to_their_text(examples):
    tok = CharTokenizer()
    ex = examples[1]
    lay = build_layout(tok, ex["question"], ex["chunks"])
    for i, (s, e) in enumerate(lay.spans):
        text = tok.decode(lay.ids[0, s:e].tolist())
        assert ex["chunks"][i]["text"].strip() in text


def test_layout_keeps_scores_and_gold_aligned_with_prompt_order(examples):
    tok = CharTokenizer()
    ex = examples[2]
    chunks = reorder_chunks(ex["chunks"], "rank")
    lay = build_layout(tok, ex["question"], chunks)
    assert list(lay.scores) == sorted(lay.scores, reverse=True)
    assert lay.gold.sum() == 1


def test_chunk_positions_cover_exactly_the_chunk_tokens(layout):
    pos = layout.chunk_positions()
    assert len(pos) == layout.n_chunk_tokens
    assert pos.min() == layout.n_prefix and pos.max() == layout.n_context - 1


# ---- data -----------------------------------------------------------------

def test_synthetic_has_one_gold_chunk_that_contains_the_answer(examples):
    for ex in examples:
        gold = [c for c in ex["chunks"] if c["gold"]]
        assert len(gold) == 1
        assert ex["answers"][0] in gold[0]["text"]
        others = [c["text"] for c in ex["chunks"] if not c["gold"]]
        assert all(ex["answers"][0] not in t for t in others)


def test_synthetic_is_reproducible():
    assert make_dataset(3, seed=7) == make_dataset(3, seed=7)
    assert make_dataset(3, seed=7) != make_dataset(3, seed=8)


def test_noise_controls_retriever_quality():
    def top1(noise):
        data = make_dataset(300, k=8, noise=noise, signal=2.0, seed=0)
        return np.mean([max(ex["chunks"], key=lambda c: c["score"])["gold"] for ex in data])
    assert top1(0.2) > top1(1.0) > top1(5.0)
    assert top1(5.0) < 0.5


@pytest.mark.parametrize("how", ["shuffle", "rank", "reverse_rank", "original"])
def test_reorder_is_a_permutation(examples, how):
    chunks = examples[0]["chunks"]
    out = reorder_chunks(chunks, how, seed=2)
    assert sorted(c["text"] for c in out) == sorted(c["text"] for c in chunks)


def test_save_load_roundtrip(tmp_path, examples):
    path = tmp_path / "x.jsonl"
    save_examples(examples, path)
    assert load_examples(path) == examples
    assert len(load_examples(path, n=2)) == 2


def test_bm25_puts_the_gold_chunk_first_on_synthetic_questions():
    data = make_dataset(40, k=6, noise=1.0, seed=5)
    add_scores(data, "bm25")
    hits = [max(ex["chunks"], key=lambda c: c["score"])["gold"] for ex in data]
    assert np.mean(hits) > 0.9


def test_bm25_scores_a_matching_doc_above_a_non_matching_one():
    docs = ["the cat sat on the mat", "dogs bark loudly at night", "a cat and a dog"]
    bm = BM25(docs)
    assert bm.score("cat mat", docs[0]) > bm.score("cat mat", docs[1])


# ---- metrics --------------------------------------------------------------

def test_normalisation_and_em():
    assert metrics.exact_match("The Eiffel Tower.", "eiffel tower") == 1.0
    assert metrics.exact_match("Paris", "London") == 0.0


def test_f1_partial_credit():
    assert metrics.f1("new york city", "new york") == pytest.approx(0.8)
    assert metrics.f1("", "") == 1.0
    assert metrics.f1("yes", "no") == 0.0


def test_score_takes_the_best_reference():
    s = metrics.score("Paris", ["London", "paris"])
    assert s["em"] == 1.0 and s["f1"] == 1.0
