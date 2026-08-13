# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.quantization.utils import fp8_utils
from vllm.platforms import current_platform


def _get_fused_moe_configs(e, n, block_shape):
    if block_shape is None:
        return fused_moe.get_moe_configs(e, n, "fp8_w8a8")
    block_n, block_k = block_shape
    return fused_moe.get_moe_configs(e, n, "fp8_w8a8", block_n, block_k)


def test_rtx_pro_6000_variants_reuse_workstation_tuned_configs(monkeypatch):
    monkeypatch.setattr(
        current_platform,
        "get_device_name",
        lambda: "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    )
    monkeypatch.setattr(fused_moe.envs, "VLLM_BATCH_INVARIANT", False)
    fp8_utils.get_w8a8_block_fp8_configs.cache_clear()
    fused_moe.get_moe_configs.cache_clear()

    assert fp8_utils.get_w8a8_block_fp8_configs(1536, 4096, 128, 128) is not None
    assert _get_fused_moe_configs(256, 384, (128, 128)) is not None
