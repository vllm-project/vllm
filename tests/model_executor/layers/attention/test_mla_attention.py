# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.attention.mla_attention import MLACommonImpl


class _FakeKVProj:
    def __init__(self, params_dtype: torch.dtype):
        self.params_dtype = params_dtype
        self.weight = torch.empty(4, dtype=torch.int32)
        self.seen_dtype = None

    def __call__(self, x: torch.Tensor):
        self.seen_dtype = x.dtype
        return (
            torch.zeros(
                x.shape[0],
                2,
                dtype=x.dtype,
                device=x.device,
            ),
            None,
        )


def test_compute_prefill_context_uses_kv_b_proj_params_dtype(monkeypatch):
    monkeypatch.setattr(ops, "gather_and_maybe_dequant_cache", lambda **kwargs: None)

    attn = SimpleNamespace(
        kv_cache_dtype="auto",
        kv_lora_rank=2,
        num_heads=1,
        qk_nope_head_dim=1,
        v_head_dim=1,
        kv_b_proj=_FakeKVProj(torch.bfloat16),
        _concat_k_nope_k_pe=lambda k_nope, k_pe: k_nope,
        _run_prefill_context_chunk=lambda **kwargs: (
            torch.zeros(2, 1, 1, dtype=torch.bfloat16),
            torch.zeros(2, 1, dtype=torch.float32),
        ),
    )

    workspace = torch.randn(2, 3, dtype=torch.float32)
    chunked_context = SimpleNamespace(
        seq_tot=[2],
        workspace=workspace,
        cu_seq_lens=[torch.tensor([0, 2], dtype=torch.int32)],
        token_to_seq=[torch.tensor([0, 0], dtype=torch.int32)],
        chunk_total_token=[2],
        starts=[torch.tensor([0], dtype=torch.int32)],
    )
    prefill = SimpleNamespace(
        q_data_type=torch.bfloat16,
        chunked_context=chunked_context,
        block_table=torch.zeros(1, 1, dtype=torch.int32),
    )
    attn_metadata = SimpleNamespace(prefill=prefill, num_prefills=1)

    q = torch.zeros(2, 1, 2, dtype=torch.bfloat16)
    kv_c_and_k_pe_cache = torch.zeros_like(workspace)
    k_scale = torch.tensor(1.0)

    MLACommonImpl._compute_prefill_context(
        attn, q, kv_c_and_k_pe_cache, attn_metadata, k_scale
    )

    assert attn.kv_b_proj.seen_dtype == torch.bfloat16
