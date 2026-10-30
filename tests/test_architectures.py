"""The target models are Llama-3.2 and Qwen2.5 style. Check that the cache surgery
and the two-stage prefill hold up on both layouts (and in half precision)."""
import pytest
import torch

from ragkv import cache_ops
from ragkv.data import reorder_chunks
from ragkv.data.synthetic import make_dataset
from ragkv.layout import build_layout
from ragkv.pipeline import prefill, run_policy
from ragkv.policies import Policy
from ragkv.tiny import tiny_model

ARCHS = ["llama", "qwen2"]


def _setup(arch, dtype=torch.float32, attn="sdpa"):
    model, tok = tiny_model(arch=arch, dtype=dtype, attn=attn)
    ex = make_dataset(2, k=5, filler=2, seed=11)[0]
    layout = build_layout(tok, ex["question"], reorder_chunks(ex["chunks"], "shuffle", seed=2))
    return model, tok, layout


@pytest.mark.parametrize("arch", ARCHS)
@torch.no_grad()
def test_two_stage_prefill_equals_single_forward(arch):
    model, _, layout = _setup(arch)
    pre = prefill(model, layout)
    T = cache_ops.seq_length(pre.cache)
    n_kv = cache_ops.layer_kv(pre.cache, 0)[0].shape[1]
    keep = [torch.arange(T).expand(n_kv, -1)] * cache_ops.n_layers(pre.cache)
    full = cache_ops.gather(pre.cache, keep)
    mask = torch.ones(1, T + 1, dtype=torch.long)
    h = model.model(input_ids=torch.tensor([[pre.last_id]]), past_key_values=full,
                    attention_mask=mask, position_ids=torch.tensor([[pre.last_pos]]))
    got = model.lm_head(h.last_hidden_state[:, -1])
    ref = model(layout.ids).logits[:, -1]
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("name", ["global+knorm", "retriever+knorm", "uniform+knorm", "global+recency"])
def test_policies_run_and_hit_the_budget(arch, name):
    model, tok, layout = _setup(arch)
    pre = prefill(model, layout)
    out = run_policy(model, tok, pre, Policy(name), 0.3, max_new_tokens=3)
    q = layout.total - 1 - layout.n_context
    assert out["kept_tokens"] == layout.n_prefix + q + round(0.3 * layout.n_chunk_tokens)


@pytest.mark.parametrize("arch", ARCHS)
def test_snapkv_on_both_architectures(arch):
    model, tok, layout = _setup(arch, attn="eager")
    pre = prefill(model, layout, need_attention=True)
    out = run_policy(model, tok, pre, Policy("retriever+snapkv"), 0.25, max_new_tokens=3)
    assert out["kept_tokens"] > layout.n_prefix


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_runs_and_halves_the_cache(dtype):
    def kv_mb(dt):
        model, tok, layout = _setup("qwen2", dtype=dt)
        pre = prefill(model, layout)
        return run_policy(model, tok, pre, Policy("retriever+knorm"), 0.3, max_new_tokens=1)["kv_mb"]

    assert kv_mb(dtype) == pytest.approx(kv_mb(torch.float32) / 2, rel=1e-6)
