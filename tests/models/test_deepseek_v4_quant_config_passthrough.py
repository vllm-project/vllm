# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Constructor contract for DSv4 compressor/indexer linear sub-modules.

The compressor's ``fused_wkv_wgate`` and the indexer's ``wq_b`` /
``weights_proj`` are part of the attention path and must honour the
model's quantization recipe. If they are built as unquantized BF16
linears while the checkpoint carries quantized weights (e.g. FP8 block
scales) for those prefixes, weight loading fails with a missing
``weight_scale`` key. The model-level ``quant_config`` therefore has to
reach these sub-module constructors so the compressed-tensors scheme
resolver can match their prefixes against the artifact's target
patterns and allocate the right parameters.

These tests pin down that contract for both the quantized case (the
config is propagated unchanged) and the unquantized case (``None`` is
preserved end-to-end, so non-quantized checkpoints still load via
``UnquantizedLinearMethod``).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def _vllm_config_ctx():
    """Set a current VllmConfig so RMSNorm / CustomOp dispatch works."""
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()) as cfg:
        yield cfg


def _make_compressor_stub_config(quant_config_marker):
    hf = SimpleNamespace(qk_rope_head_dim=64, rms_norm_eps=1e-6)
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, max_model_len=1024),
        scheduler_config=SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=64),
        compilation_config=SimpleNamespace(static_forward_context={}),
        quant_config=quant_config_marker,
    )


@pytest.mark.parametrize(
    "quant_config_marker",
    ["SENTINEL_QUANT_CFG", None],
    ids=["quantized", "unquantized"],
)
def test_compressor_passes_quant_config_to_fused_wkv_wgate(
    _vllm_config_ctx, quant_config_marker
):
    """``fused_wkv_wgate`` must receive the model-level ``quant_config``
    from ``vllm_config`` so quantized checkpoints can load its weights."""
    captured: list[dict] = []

    class FakeMergedLinear:
        def __init__(self, *args, **kwargs):
            captured.append(kwargs)

    with patch(
        "vllm.models.deepseek_v4.compressor.MergedColumnParallelLinear",
        FakeMergedLinear,
    ):
        from vllm.models.deepseek_v4.compressor import DeepseekCompressor

        DeepseekCompressor(
            vllm_config=_make_compressor_stub_config(quant_config_marker),
            compress_ratio=128,
            hidden_size=2048,
            head_dim=128,
            prefix="layers.0.attn.compressor",
        )

    assert len(captured) == 1, "fused_wkv_wgate should be constructed exactly once"
    kwargs = captured[0]
    assert kwargs.get("prefix") == "layers.0.attn.compressor.fused_wkv_wgate"
    assert kwargs.get("quant_config") is quant_config_marker, (
        "fused_wkv_wgate must be constructed with vllm_config.quant_config; "
        f"got {kwargs.get('quant_config')!r}"
    )


@pytest.mark.parametrize(
    "quant_config_marker",
    ["SENTINEL_QUANT_CFG", None],
    ids=["quantized", "unquantized"],
)
def test_indexer_passes_quant_config_to_weights_proj_and_wq_b(
    _vllm_config_ctx, quant_config_marker
):
    """Both ``wq_b`` and ``weights_proj`` must be constructed with the
    same ``quant_config`` the indexer itself was given, so the indexer's
    linear sub-modules participate in the same quantization scheme as
    the surrounding model."""
    captured: list[dict] = []

    class FakeReplicatedLinear:
        def __init__(self, *args, **kwargs):
            captured.append(kwargs)

    attn_mod = "vllm.models.deepseek_v4.nvidia.ops.attention"
    with (
        patch(f"{attn_mod}.ReplicatedLinear", FakeReplicatedLinear),
        patch(f"{attn_mod}.DeepseekV4IndexerCache", MagicMock()),
        patch(f"{attn_mod}.DeepseekCompressor", MagicMock()),
        patch(f"{attn_mod}.SparseAttnIndexer", MagicMock()),
        patch(f"{attn_mod}.get_max_prefill_buffer_size", return_value=4096),
        patch("torch.cuda.Event", MagicMock()),
    ):
        from vllm.models.deepseek_v4.nvidia.ops.attention import (
            DeepseekV4Indexer,
        )

        cfg = SimpleNamespace(
            index_topk=64,
            index_n_heads=8,
            index_head_dim=128,
            qk_rope_head_dim=64,
        )
        vllm_cfg = SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=1024),
            attention_config=SimpleNamespace(use_fp4_indexer_cache=False),
        )

        DeepseekV4Indexer(
            vllm_config=vllm_cfg,
            config=cfg,
            hidden_size=2048,
            q_lora_rank=1536,
            quant_config=quant_config_marker,
            cache_config=MagicMock(),
            topk_indices_buffer=None,
            compress_ratio=1,
            prefix="layers.0.attn.indexer",
        )

    by_prefix = {kw.get("prefix"): kw for kw in captured}
    assert "layers.0.attn.indexer.wq_b" in by_prefix
    assert "layers.0.attn.indexer.weights_proj" in by_prefix

    assert by_prefix["layers.0.attn.indexer.wq_b"]["quant_config"] is (
        quant_config_marker
    )
    assert by_prefix["layers.0.attn.indexer.weights_proj"]["quant_config"] is (
        quant_config_marker
    ), "weights_proj must be constructed with the indexer's quant_config"
