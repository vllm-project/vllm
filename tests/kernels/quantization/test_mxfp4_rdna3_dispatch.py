# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-size dispatch tests for the RDNA3 (gfx1100) MXFP4 MoE experts backend.

``RDNA3Mxfp4Experts`` (the ``Mxfp4MoeBackend.RDNA3_MXFP4`` kernel) picks a GEMM
tile from expected occupancy: the WMMA-16 tile once a step has >=16 rows, else a
scalar tile (1/4). The ``moe_mxfp4_gemm_rdna3`` kernel only instantiates tiles
{1,2,4,8,16}, so the selector must never return anything else. Pure-Python (no
GPU), so this runs on any CI host.
"""

import pytest

from vllm.model_executor.layers.fused_moe.experts.rdna3_mxfp4_moe import (
    _select_block_size_m,
)

SUPPORTED_TILES = {1, 2, 4, 8, 16}  # moe_mxfp4_gemm_rdna3 instantiations
BATCH_SIZES = [1, 2, 3, 4, 8, 15, 16, 17, 32, 64, 256, 512, 4096]
TOP_KS = [1, 2, 4, 8]
NUM_EXPERTS = [8, 32, 128, 256]


@pytest.mark.parametrize("num_tokens", BATCH_SIZES)
@pytest.mark.parametrize("top_k", TOP_KS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
def test_tile_in_supported_set(num_tokens, top_k, num_experts):
    bsm = _select_block_size_m(num_tokens, top_k, num_experts)
    assert bsm in SUPPORTED_TILES
    # the WMMA-16 tile is only chosen once a step can fill it
    if num_tokens < 16:
        assert bsm != 16


def test_large_batches_use_wmma_tile():
    # >=16-row steps (prefill / high-concurrency decode) take the WMMA tile.
    assert _select_block_size_m(16, 8, 256) == 16
    assert _select_block_size_m(512, 8, 256) == 16


def test_degenerate_inputs_return_safe_tile():
    # top_k/num_experts <= 0 must not divide-by-zero; fall back to the 1-tile.
    assert _select_block_size_m(32, 0, 256) == 1
    assert _select_block_size_m(32, 8, 0) == 1
