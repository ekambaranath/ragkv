import numpy as np
import pytest
import torch

from ragkv import cache_ops
from ragkv.pipeline import decode, prefill, run_policy
from ragkv.policies import Policy


def keep_everything(pre):
    T = cache_ops.seq_length(pre.cache)
    n_kv = cache_ops.layer_kv(pre.cache, 0)[0].shape[1]
    return [torch.arange(T).expand(n_kv, -1)] * cache_ops.n_layers(pre.cache)


def last_token_logits(model, pre, cache):
    mask = torch.ones(1, cache_ops.seq_length(cache) + 1, dtype=torch.long)
    h = model.model(input_ids=torch.tensor([[pre.last_id]]), past_key_values=cache,
                    attention_mask=mask, position_ids=torch.tensor([[pre.last_pos]]))
    return model.lm_head(h.last_hidden_state[:, -1])


# ---- the two-stage prefill must be the same computation as one forward ----

@torch.no_grad()
def test_two_stage_prefill_matches_a_single_forward(tiny, layout):
    model, _ = tiny
    pre = prefill(model, layout)
    full = cache_ops.gather(pre.cache, keep_everything(pre))
    got = last_token_logits(model, pre, full)
    ref = model(layout.ids).logits[:, -1]
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_full_cache_decoding_matches_hf_generate(tiny, layout):
    model, _ = tiny

    class NoNewlineTok:                     # keep the stop rule from ending the comparison early
        eos_token_id = None
        def decode(self, ids, skip_special_tokens=True):
            return ""

    pre = prefill(model, layout)
    full = cache_ops.gather(pre.cache, keep_everything(pre))
    got = decode(model, NoNewlineTok(), full, pre.last_id, pre.last_pos, max_new_tokens=8)
    ref = model.generate(layout.ids, attention_mask=torch.ones_like(layout.ids), do_sample=False,
                         max_new_tokens=8, pad_token_id=0)[0, layout.total:].tolist()
    assert len(got) >= 1
    assert got == ref[:len(got)]


# ---- evicting must be exactly "the last token cannot see the evicted keys" ----

@torch.no_grad()
def test_pruned_cache_equals_full_forward_with_evicted_keys_masked(tiny, layout):
    model, _ = tiny
    pre = prefill(model, layout)
    T = layout.total

    pol = Policy("global+recency")          # same positions kept in every head, so a plain 2D mask can express it
    keep, _ = pol.select(pre.cache, pre.attn, layout, 0.4)
    small = cache_ops.gather(pre.cache, keep)
    got = last_token_logits(model, pre, small)

    kept_positions = set(keep[0][0].tolist())
    evicted = [p for p in range(T - 1) if p not in kept_positions]
    assert evicted, "test needs something to evict"

    neg = torch.finfo(torch.float32).min
    mask = torch.zeros(T, T).masked_fill(torch.ones(T, T).triu(1).bool(), neg)
    mask[T - 1, evicted] = neg              # only the last query row loses the evicted keys
    try:
        ref = model(input_ids=layout.ids, attention_mask=mask[None, None]).logits[:, -1]
    except (TypeError, ValueError, RuntimeError) as e:
        pytest.skip(f"this transformers version does not take 4D masks here: {e}")
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4)

    # and the check has teeth: the wrong position id (compressed length) gives something else
    h = model.model(input_ids=torch.tensor([[pre.last_id]]), past_key_values=cache_ops.gather(pre.cache, keep),
                    attention_mask=torch.ones(1, cache_ops.seq_length(small) + 1, dtype=torch.long),
                    position_ids=torch.tensor([[cache_ops.seq_length(small)]]))
    wrong = model.lm_head(h.last_hidden_state[:, -1])
    assert not torch.allclose(wrong, ref, atol=1e-4, rtol=1e-4)


# ---- what the policies keep ----------------------------------------------

@pytest.mark.parametrize("name", ["global+knorm", "uniform+knorm", "retriever+knorm", "oracle+knorm",
                                  "global+recency", "global+random"])
def test_cache_size_matches_the_budget(tiny, layout, name):
    model, tok = tiny
    pre = prefill(model, layout)
    ratio = 0.3
    out = run_policy(model, tok, pre, Policy(name), ratio, seed=1, max_new_tokens=2)
    budget = round(ratio * layout.n_chunk_tokens)
    expected = layout.n_prefix + (layout.total - 1 - layout.n_context) + budget
    assert out["kept_tokens"] == expected


@pytest.mark.parametrize("name", ["global+knorm", "uniform+knorm", "retriever+knorm", "global+random"])
def test_prefix_and_question_are_never_evicted(tiny, layout, name):
    model, _ = tiny
    pre = prefill(model, layout)
    keep, _ = Policy(name).select(pre.cache, pre.attn, layout, 0.2, seed=0)
    T = cache_ops.seq_length(pre.cache)
    protected = set(range(layout.n_prefix)) | set(range(layout.n_context, T))
    for layer_keep in keep:
        for head in layer_keep:
            h = head.tolist()
            assert protected <= set(h)
            assert h == sorted(h) and len(set(h)) == len(h)


def test_each_chunk_keeps_exactly_its_budget_in_every_head(tiny, layout):
    model, _ = tiny
    pre = prefill(model, layout)
    keep, budgets = Policy("retriever+knorm").select(pre.cache, pre.attn, layout, 0.3)
    for layer_keep in keep:
        for head in layer_keep:
            kept = np.array(head.tolist())
            for (s, e), b in zip(layout.spans, budgets):
                assert ((kept >= s) & (kept < e)).sum() == b


def test_retriever_policy_spends_more_on_higher_scored_chunks(tiny, layout):
    model, _ = tiny
    pre = prefill(model, layout)
    _, budgets = Policy("retriever+knorm", temperature=0.5).select(pre.cache, pre.attn, layout, 0.3)
    best, worst = int(np.argmax(layout.scores)), int(np.argmin(layout.scores))
    lengths = layout.chunk_lengths
    assert budgets[best] / lengths[best] >= budgets[worst] / lengths[worst]


def test_oracle_policy_prioritises_the_gold_chunk(tiny, layout):
    model, _ = tiny
    pre = prefill(model, layout)
    _, budgets = Policy("oracle+knorm", floor=2).select(pre.cache, pre.attn, layout, 0.3)
    gold = int(np.flatnonzero(layout.gold)[0])
    assert budgets[gold] == max(budgets)


@torch.no_grad()
def test_knorm_keeps_the_lowest_norm_keys_inside_a_chunk(tiny, layout):
    model, _ = tiny
    pre = prefill(model, layout)
    keep, budgets = Policy("uniform+knorm").select(pre.cache, pre.attn, layout, 0.25)
    layer, head = 1, 0
    k = cache_ops.layer_kv(pre.cache, layer)[0][0, head]
    kept = set(keep[layer][head].tolist())
    for (s, e), b in zip(layout.spans, budgets):
        norms = k[s:e].float().norm(dim=-1)
        lowest = set((torch.argsort(norms)[:b] + s).tolist())
        assert lowest == {p for p in kept if s <= p < e}


def test_random_policy_depends_on_the_seed_only(tiny, layout):
    model, _ = tiny
    pre = prefill(model, layout)
    pol = Policy("global+random")
    a = pol.select(pre.cache, pre.attn, layout, 0.3, seed=1)[0]
    b = pol.select(pre.cache, pre.attn, layout, 0.3, seed=1)[0]
    c = pol.select(pre.cache, pre.attn, layout, 0.3, seed=2)[0]
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not all(torch.equal(x, y) for x, y in zip(a, c))


def test_full_ratio_is_the_same_as_the_full_policy(tiny, layout):
    model, tok = tiny
    pre = prefill(model, layout)
    full = run_policy(model, tok, pre, Policy("full"), 1.0, max_new_tokens=6)
    ours = run_policy(model, tok, pre, Policy("retriever+knorm"), 1.0, max_new_tokens=6)
    assert full["text"] == ours["text"]
    assert full["kept_tokens"] == ours["kept_tokens"] == layout.total - 1


def test_decoding_does_not_touch_the_prefilled_cache(tiny, layout):
    model, tok = tiny
    pre = prefill(model, layout)
    before = cache_ops.seq_length(pre.cache)
    k0 = cache_ops.layer_kv(pre.cache, 0)[0].clone()
    for name in ["full", "retriever+knorm", "global+knorm"]:
        run_policy(model, tok, pre, Policy(name), 0.3, max_new_tokens=4)
    assert cache_ops.seq_length(pre.cache) == before
    assert torch.equal(cache_ops.layer_kv(pre.cache, 0)[0], k0)


def test_compressed_cache_is_smaller_in_bytes(tiny, layout):
    model, tok = tiny
    pre = prefill(model, layout)
    big = run_policy(model, tok, pre, Policy("full"), 1.0, max_new_tokens=1)["kv_mb"]
    small = run_policy(model, tok, pre, Policy("retriever+knorm"), 0.2, max_new_tokens=1)["kv_mb"]
    assert small < big


# ---- snapkv needs attention weights --------------------------------------

@torch.no_grad()
def test_snapkv_runs_with_gqa_and_respects_the_budget(tiny_eager, layout):
    model, tok = tiny_eager
    pol = Policy("global+snapkv")
    pre = prefill(model, layout, need_attention=pol.needs_attention)
    assert pre.attn is not None
    q = layout.total - 1 - layout.n_context
    assert pre.attn[0].shape[2:] == (q, layout.n_context + q)
    out = run_policy(model, tok, pre, pol, 0.25, max_new_tokens=3)
    budget = round(0.25 * layout.n_chunk_tokens)
    assert out["kept_tokens"] == layout.n_prefix + q + budget


def test_snapkv_without_attention_weights_says_so(tiny, layout):
    model, tok = tiny
    pre = prefill(model, layout, need_attention=False)
    with pytest.raises(RuntimeError, match="attention"):
        run_policy(model, tok, pre, Policy("global+snapkv"), 0.3)


def test_snapkv_prefers_tokens_the_question_attends_to(tiny_eager, layout):
    model, _ = tiny_eager
    pol = Policy("retriever+snapkv")
    pre = prefill(model, layout, need_attention=True)
    keep, budgets = pol.select(pre.cache, pre.attn, layout, 0.3)
    k = cache_ops.layer_kv(pre.cache, 0)[0]
    scores = pol._token_scores(k, pre.attn[0], layout, torch.Generator())
    s, e = layout.spans[0]
    b = budgets[0]
    kept_here = [p for p in keep[0][0].tolist() if s <= p < e]
    assert len(kept_here) == b
    assert set(kept_here) == set((scores[0, s:e].topk(b).indices + s).tolist())


# ---- policy names ---------------------------------------------------------

def test_aliases_and_errors():
    assert Policy("ours").name == "retriever+knorm"
    assert Policy("snapkv").name == "global+snapkv"
    assert Policy("streaming").name == "global+recency"
    assert Policy("full").is_full
    assert Policy("global+snapkv").needs_attention and not Policy("global+knorm").needs_attention
    for bad in ["nonsense", "retriever+nothing", "what+knorm", "a+b+c"]:
        with pytest.raises(ValueError):
            Policy(bad)
