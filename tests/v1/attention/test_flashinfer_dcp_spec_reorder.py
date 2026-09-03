# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer GQA builder: reorder threshold under DCP with spec decode."""

import pytest

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("FlashInfer backend requires a CUDA platform.", allow_module_level=True)

import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.config import SpeculativeConfig, set_current_vllm_config
from vllm.v1.attention.backends import flashinfer as flashinfer_backend
from vllm.v1.attention.backends.flashinfer import (
    FlashInferDecodeKernel,
    FlashInferMetadataBuilder,
)
from vllm.v1.attention.backends.utils import PerLayerParameters
from vllm.v1.kv_cache_interface import FullAttentionSpec


def test_flashinfer_gqa_dcp_spec_decode_clamps_reorder_threshold(monkeypatch):
    """trtllm-gen decode receives no cp_rank/global-seq-len information, so its
    end-aligned causal mask is wrong for q_len > 1 over the DCP-interleaved
    local KV shard. The builder must keep reorder_batch_threshold at 1 under
    DCP so spec queries take the (DCP-aware) prefill path instead.
    """
    vllm_config = create_vllm_config(max_model_len=1024)
    vllm_config.parallel_config.decode_context_parallel_size = 2
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=3
    )

    monkeypatch.setattr(
        flashinfer_backend, "can_use_trtllm_attention", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        FlashInferMetadataBuilder,
        "_get_flashinfer_trtllm_api_decode_kernel",
        staticmethod(lambda: FlashInferDecodeKernel.TRTLLM_GEN),
    )
    monkeypatch.setattr(
        flashinfer_backend,
        "get_per_layer_parameters",
        lambda *args, **kwargs: {
            "layer.0": PerLayerParameters(
                window_left=-1, logits_soft_cap=None, sm_scale=0.1, has_sinks=False
            )
        },
    )

    kv_cache_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        ),
        head_size=vllm_config.model_config.get_head_size(),
        dtype=vllm_config.model_config.dtype,
    )
    with set_current_vllm_config(vllm_config):
        builder = FlashInferMetadataBuilder(
            kv_cache_spec,
            ["layer.0"],
            vllm_config,
            torch.device("cpu"),
        )

    # Guard against passing vacuously with the kernel disabled.
    assert (
        builder.flashinfer_trtllm_api_decode_kernel == FlashInferDecodeKernel.TRTLLM_GEN
    )
    assert builder.reorder_batch_threshold == 1
