# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-size dispatch tests for the RDNA3 (gfx1100) fused-MoE methods.

``RDNA3FusedMoEMixin`` is shared by the MXFP4 MoE method and the existing W4A16
(``moe_gptq_gemm_rdna3``) method. The MXFP4 kernel has a WMMA-16 tile; the W4A16
kernel only instantiates tiles {1,2,4,8} and rejects anything else. So the
W4A16 method must override tile selection to stay inside its set — otherwise any
>=16-token prefill / batched decode hard-crashes. These are pure-Python checks
(no GPU), so the regression is caught on any CI host.
"""

import pytest

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.rdna3_moe_common import (  # noqa: E501
    RDNA3FusedMoEMixin,
    select_block_size_m,
)

# Tiles each HIP kernel actually instantiates (the rest hit a TORCH_CHECK).
GPTQ_TILES = {1, 2, 4, 8}  # moe_gptq_gemm_rdna3 (W4A16)
MXFP4_TILES = {1, 2, 4, 8, 16}  # moe_mxfp4_gemm_rdna3

BATCH_SIZES = [1, 2, 3, 4, 8, 15, 16, 17, 32, 64, 256, 512, 4096]
TOP_KS = [1, 2, 4, 8]
NUM_EXPERTS = [8, 32, 128, 256]


def _w4a16_cls():
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16_rdna3 import (  # noqa: E501
        CompressedTensorsWNA16RDNA3MoEMethod,
    )

    return CompressedTensorsWNA16RDNA3MoEMethod


@pytest.mark.parametrize("num_tokens", BATCH_SIZES)
@pytest.mark.parametrize("top_k", TOP_KS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
def test_mxfp4_tile_in_supported_set(num_tokens, top_k, num_experts):
    bsm = select_block_size_m(num_tokens, top_k, num_experts)
    assert bsm in MXFP4_TILES
    # the WMMA-16 tile is only picked once a step can fill it
    if num_tokens < 16:
        assert bsm != 16


@pytest.mark.parametrize("num_tokens", BATCH_SIZES)
@pytest.mark.parametrize("top_k", TOP_KS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
def test_w4a16_never_selects_unsupported_tile(num_tokens, top_k, num_experts):
    """Regression guard: the W4A16 method must never hand moe_gptq_gemm_rdna3 a
    tile outside {1,2,4,8} (>=16-token batches used to select 16 and crash)."""
    # _select_block_size_m is stateless, so call it without building a method.
    bsm = _w4a16_cls()._select_block_size_m(None, num_tokens, top_k, num_experts)
    assert bsm in GPTQ_TILES, (
        f"W4A16 selected block_size_m={bsm} for num_tokens={num_tokens}; "
        f"moe_gptq_gemm_rdna3 only supports {sorted(GPTQ_TILES)}"
    )


def test_w4a16_override_differs_from_mixin_default():
    """The mixin default returns the WMMA-16 tile for a 32-token batch; the
    W4A16 override must replace it with a kernel-supported tile."""
    assert RDNA3FusedMoEMixin._select_block_size_m(None, 32, 8, 256) == 16
    w4a16 = _w4a16_cls()._select_block_size_m(None, 32, 8, 256)
    assert w4a16 != 16
    assert w4a16 in GPTQ_TILES


def test_degenerate_inputs_return_safe_tile():
    # num_experts/top_k <= 0 must not divide-by-zero; fall back to the 1-tile.
    assert select_block_size_m(32, 0, 256) == 1
    assert select_block_size_m(32, 8, 0) == 1
