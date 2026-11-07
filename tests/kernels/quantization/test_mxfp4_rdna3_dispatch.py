# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-size dispatch tests for the RDNA3 (gfx1100) MXFP4 MoE experts backend.

``RDNA3Mxfp4Experts`` (the ``Mxfp4MoeBackend.RDNA3_MXFP4`` kernel) picks a GEMM
tile from expected occupancy: the WMMA-16 tile once a step has >=16 rows, else a
scalar tile (1/4). The ``moe_mxfp4_gemm_rdna3`` kernel only instantiates tiles
{1,2,4,8,16}, so the selector must never return anything else. Pure-Python (no
GPU), so this runs on any CI host.

The oracle tests at the bottom run on real gfx1100 hardware: which backend a
checkpoint that asks for quantized activations lands on.
"""

from unittest.mock import patch

import pytest
import torch

from vllm.config.kernel import MoEBackend
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.rdna3_mxfp4_moe import (
    _select_block_size_m,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    select_mxfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
    kMxfp4Dynamic,
)
from vllm.platforms import current_platform

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


ON_GFX1100 = False
if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx1100

    ON_GFX1100 = on_gfx1100()

# W4A16 checkpoints, Quark W4A4 (e.g. amd/Qwen3-Coder-Next-MXFP4) and Quark W4A8
# (e.g. amd/gpt-oss-120b-w-mxfp4-a-fp8). gfx1100 can run none of the quantized
# activations natively.
ACTIVATION_KEYS = [None, kMxfp4Dynamic, kFp8StaticTensorSym]


def _make_moe_config(moe_backend: MoEBackend) -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=256,
        intermediate_size=256,
        num_local_experts=8,
        num_logical_experts=8,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.Renormalize,
        moe_backend=moe_backend,
    )


@pytest.fixture
def no_user_override():
    with patch(
        "vllm.model_executor.layers.fused_moe.oracle.mxfp4.get_current_vllm_config"
    ) as mock_get_config:
        mock_get_config.return_value.model_config.quantization_config = None
        yield


def _with_quark(installed: bool):
    return patch(
        "vllm.model_executor.layers.fused_moe.experts.ocp_mx_emulation_moe.has_quark",
        return_value=installed,
    )


@pytest.mark.skipif(not ON_GFX1100, reason="Requires gfx1100")
@pytest.mark.parametrize("activation_key", ACTIVATION_KEYS)
def test_explicit_rdna3_backend(no_user_override, activation_key: QuantKey | None):
    """--moe-backend rdna3 selects the native kernel whatever the checkpoint's
    activation scheme, running activations in BF16."""
    with _with_quark(True):
        backend, _ = select_mxfp4_moe_backend(
            _make_moe_config("rdna3"), activation_key=activation_key
        )
    assert backend == Mxfp4MoeBackend.RDNA3_MXFP4


@pytest.mark.skipif(not ON_GFX1100, reason="Requires gfx1100")
def test_auto_w4a16_selects_rdna3(no_user_override):
    with _with_quark(True):
        backend, _ = select_mxfp4_moe_backend(_make_moe_config("auto"))
    assert backend == Mxfp4MoeBackend.RDNA3_MXFP4


@pytest.mark.skipif(not ON_GFX1100, reason="Requires gfx1100")
@pytest.mark.parametrize("activation_key", [kMxfp4Dynamic, kFp8StaticTensorSym])
@pytest.mark.parametrize(
    "quark_installed, expected",
    [
        # Emulation honours the checkpoint's activation quantization, so it
        # wins under auto; the native kernel is the BF16 fallback.
        (True, Mxfp4MoeBackend.EMULATION),
        (False, Mxfp4MoeBackend.RDNA3_MXFP4),
    ],
)
def test_auto_quantized_activations(
    no_user_override, activation_key, quark_installed, expected
):
    with _with_quark(quark_installed):
        backend, _ = select_mxfp4_moe_backend(
            _make_moe_config("auto"), activation_key=activation_key
        )
    assert backend == expected
