# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The AITER MLA builder takes the KV cache dtype from its group's spec.

Builders are constructed with the target's VllmConfig, so a speculative draft
whose ``kv_cache_dtype`` differs from the target's must not inherit the
target's ``cache_config.cache_dtype``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

rocm_aiter_mla = pytest.importorskip(
    "vllm.v1.attention.backends.mla.rocm_aiter_mla",
    reason="ROCm AITER MLA backend not available",
)


def _build(monkeypatch, global_cache_dtype, group_cache_dtype, group_dtype):
    """Run the real builder __init__ on CPU without DCP or speculative decoding.

    Stubs only what needs a GPU, AITER, or a DCP group.
    """

    def init_common_builder(self, *args, **kwargs):
        self.num_heads = 16
        self.q_data_type = torch.bfloat16
        self.mla_dims = SimpleNamespace(
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        )
        self.dcp_world_size = 1
        self.reorder_batch_threshold = 1

    monkeypatch.setattr(
        rocm_aiter_mla.MLACommonMetadataBuilder, "__init__", init_common_builder
    )
    monkeypatch.setattr(rocm_aiter_mla, "_fp8_mla_prefill_supported", lambda: False)
    monkeypatch.setitem(
        sys.modules,
        "aiter",
        SimpleNamespace(
            dtypes=SimpleNamespace(fp8="fp8", fp16="fp16", bf16="bf16"),
            get_mla_metadata_info_v1=lambda *args, **kwargs: tuple(
                (1, torch.int32) for _ in range(6)
            ),
        ),
    )

    config = SimpleNamespace(
        speculative_config=None,
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, cp_kv_cache_interleave_size=1
        ),
        model_config=SimpleNamespace(max_model_len=16, dtype=torch.bfloat16),
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        cache_config=SimpleNamespace(
            cache_dtype=global_cache_dtype, num_gpu_blocks=None
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False)
        ),
    )
    return rocm_aiter_mla.AiterMLAMetadataBuilder(
        kv_cache_spec=SimpleNamespace(
            block_size=1, dtype=group_dtype, cache_dtype_str=group_cache_dtype
        ),
        layer_names=["layer.0"],
        vllm_config=config,
        device=torch.device("cpu"),
    )


def test_fp8_group_under_bf16_target(monkeypatch):
    builder = _build(
        monkeypatch,
        global_cache_dtype="auto",
        group_cache_dtype="fp8",
        group_dtype=torch.uint8,
    )

    assert builder._kv_cache_dtype_str == "fp8"
    assert builder._mla_kv_dtype == "fp8"
    assert builder._mla_q_dtype == "fp8"


def test_bf16_group_under_fp8_target(monkeypatch):
    builder = _build(
        monkeypatch,
        global_cache_dtype="fp8",
        group_cache_dtype="auto",
        group_dtype=torch.bfloat16,
    )

    assert builder._kv_cache_dtype_str == "auto"
    assert builder._mla_kv_dtype == "bf16"
    assert builder._mla_q_dtype == torch.bfloat16
