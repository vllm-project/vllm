# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.quantization.utils import fp8_utils
from vllm.platforms import current_platform

GB10_BLOCK_FP8_SHAPES = (
    (1536, 4096),
    (16384, 1024),
    (2048, 4096),
    (4096, 1024),
    (4096, 4096),
    (8192, 1024),
)

GB10_FUSED_MOE_SHAPES = (
    (128, 704, None),
    (129, 704, None),
    (20, 1536, None),
    (256, 384, (128, 128)),
    (256, 512, (128, 128)),
    (64, 1536, (128, 128)),
)


def _get_fused_moe_configs(e, n, block_shape):
    if block_shape is None:
        return fused_moe.get_moe_configs(e, n, "fp8_w8a8")
    block_n, block_k = block_shape
    return fused_moe.get_moe_configs(e, n, "fp8_w8a8", block_n, block_k)


def test_gb10_tuned_configs_cover_dense_and_fused_moe(monkeypatch):
    monkeypatch.setattr(current_platform, "get_device_name", lambda: "NVIDIA GB10")
    monkeypatch.setattr(fused_moe.envs, "VLLM_BATCH_INVARIANT", False)
    fp8_utils.get_w8a8_block_fp8_configs.cache_clear()
    fused_moe.get_moe_configs.cache_clear()

    missing_dense = [
        (n, k)
        for n, k in GB10_BLOCK_FP8_SHAPES
        if fp8_utils.get_w8a8_block_fp8_configs(n, k, 128, 128) is None
    ]
    assert not missing_dense

    missing_moe = [
        (e, n, block_shape)
        for e, n, block_shape in GB10_FUSED_MOE_SHAPES
        if _get_fused_moe_configs(e, n, block_shape) is None
    ]
    assert not missing_moe
