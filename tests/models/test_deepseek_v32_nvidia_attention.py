# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("deepseek_v32 NVIDIA attention requires CUDA", allow_module_level=True)

from vllm.models.deepseek_v32.nvidia import attention as attention_module


class _ForbiddenProjection:
    def __call__(self, *_args, **_kwargs):
        raise AssertionError("the indexer must not run while top-k is reused")


def test_mtp_skip_topk_disables_all_indexer_work(monkeypatch):
    attn = attention_module.DeepseekV32Attention.__new__(
        attention_module.DeepseekV32Attention
    )
    torch.nn.Module.__init__(attn)

    attn.indexer = SimpleNamespace(
        wk_weights_proj=_ForbiddenProjection(),
        wq_b=_ForbiddenProjection(),
    )
    attn.skip_topk = True
    attn.q_lora_rank = 2
    attn.kv_lora_rank = 1
    attn.qk_rope_head_dim = 1
    attn.qk_nope_head_dim = 1
    attn.qk_head_dim = 2
    attn.num_local_heads = 1
    attn.v_head_dim = 1
    attn.layer_name = "model.layers.78.self_attn.attn"
    attn.kv_cache_dtype = "fp8"
    attn._index_rope_interleave = True
    attn._fp8_query = True
    attn._q_scale = torch.ones(1)
    attn.W_UK_T = torch.ones(1, 1, 1)
    attn.topk_indices_buffer = torch.full((2, 4), 7, dtype=torch.int32)

    attn.q_a_layernorm = SimpleNamespace(weight=torch.ones(2), variance_epsilon=1e-6)
    attn.kv_a_layernorm = SimpleNamespace(weight=torch.ones(1), variance_epsilon=1e-6)
    attn.rotary_emb = SimpleNamespace(cos_sin_cache=torch.ones(1, 2))

    attn.fused_qkv_a_proj = lambda hidden: (
        torch.zeros(hidden.shape[0], 4, dtype=hidden.dtype),
    )
    attn.q_b_proj = lambda q_c: (torch.zeros(q_c.shape[0], 2, dtype=q_c.dtype),)
    attn.o_proj = lambda output: (output,)

    seen = {}

    def fake_fused_norm_rope(*args, **kwargs):
        seen["norm_has_indexer"] = kwargs["has_indexer"]
        seen["indexer_k_cache"] = kwargs["indexer_k_cache"]
        return args[1]

    def fake_fused_q(*args, **kwargs):
        seen["q_has_indexer"] = kwargs["has_indexer"]
        seen["index_q"] = args[3]
        seen["index_weights"] = args[7]
        return torch.empty(0), torch.empty(0), torch.zeros_like(args[1])

    def forbidden_sparse_indexer(*_args, **_kwargs):
        raise AssertionError("sparse_attn_indexer must not run while top-k is reused")

    monkeypatch.setattr(attention_module, "fused_norm_rope", fake_fused_norm_rope)
    monkeypatch.setattr(attention_module, "fused_q", fake_fused_q)
    monkeypatch.setattr(
        attention_module, "sparse_attn_indexer", forbidden_sparse_indexer
    )
    monkeypatch.setattr(
        attention_module,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None, slot_mapping={}),
    )

    output = attn(torch.arange(2), torch.zeros(2, 4))

    assert seen == {
        "norm_has_indexer": False,
        "indexer_k_cache": None,
        "q_has_indexer": False,
        "index_q": None,
        "index_weights": None,
    }
    assert torch.count_nonzero(output) == 0
    assert torch.all(attn.topk_indices_buffer == 7)
