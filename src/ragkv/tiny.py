"""A throwaway model and tokenizer so the whole pipeline runs with no downloads.

The weights are random, so the answers it produces are noise. It exists to check
plumbing (shapes, positions, cache surgery), not to measure anything.
"""
import torch
from transformers import AutoModelForCausalLM, LlamaConfig

_NEWLINE = 98
_UNKNOWN = 99


class CharTokenizer:
    """One token per printable ASCII character. Just enough of the HF API."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        ids = [self._enc(c) for c in text]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        return {"input_ids": ids}

    @staticmethod
    def _enc(ch):
        if ch == "\n":
            return _NEWLINE
        if 32 <= ord(ch) <= 126:
            return 3 + ord(ch) - 32
        return _UNKNOWN

    def decode(self, ids, skip_special_tokens=True):
        out = []
        for i in ids:
            if i == _NEWLINE:
                out.append("\n")
            elif 3 <= i < 3 + 95:
                out.append(chr(32 + i - 3))
        return "".join(out)


def tiny_model(seed=0, attn="sdpa", layers=3, arch="llama", dtype=torch.float32):
    """arch: 'llama' (Llama 3.x style) or 'qwen2' (Qwen2.5 style, has QKV biases).
    Both use grouped-query attention: 4 query heads share 2 kv heads."""
    shape = dict(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=8192, bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    if arch == "llama":
        cfg = LlamaConfig(**shape)
    elif arch == "qwen2":
        from transformers import Qwen2Config
        cfg = Qwen2Config(**shape)
    else:
        raise ValueError(f"unknown arch: {arch}")
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation=attn)
    return model.to(dtype).eval(), CharTokenizer()
