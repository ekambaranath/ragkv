import torch


def load_model(name, dtype="auto", device=None, attn="sdpa"):
    """Returns (model, tokenizer).

    attn='eager' is only needed for scorers that read attention weights
    (snapkv); the attention-free policies are happy with 'sdpa'.
    """
    if name == "tiny-random":
        from .tiny import tiny_model
        return tiny_model(attn=attn)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype == "auto":
        dtype = torch.float16 if device == "cuda" else torch.float32
    else:
        dtype = getattr(torch, dtype)

    tok = AutoTokenizer.from_pretrained(name)
    kwargs = dict(attn_implementation=attn)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **kwargs)
    except (TypeError, ValueError):         # older transformers call it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, **kwargs)
    return model.to(device).eval(), tok
