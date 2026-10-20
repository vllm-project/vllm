# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""install_expert_pool rejects unsupported geometries and backends before
allocating anything, and bounds the planner width."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.expert_pool import install as inst
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4FusedMoE


def _method(backend=NvFp4MoeBackend.MARLIN):
    m = ModelOptNvFp4FusedMoE.__new__(ModelOptNvFp4FusedMoE)
    m.nvfp4_backend = backend
    return m


def _layer(E=8, top_k=2, size=4, method=None, use_ep=False):
    return SimpleNamespace(
        quant_method=method or _method(),
        local_num_experts=E,
        _moe_expert_pool_rows=size,
        moe_config=SimpleNamespace(
            experts_per_token=top_k,
            moe_parallel_config=SimpleNamespace(use_ep=use_ep),
        ),
    )


def test_consistent_layers_pass():
    inst.check_pool_layers([("a", _layer()), ("b", _layer())])


@pytest.mark.parametrize(
    "bad, match",
    [
        (_layer(E=16), "local_num_experts"),
        (_layer(top_k=4), "top_k"),
        (_layer(size=6), "_moe_expert_pool_rows"),
        (_layer(method=_method(NvFp4MoeBackend.VLLM_CUTLASS)), "Marlin"),
        (_layer(method=SimpleNamespace(nvfp4_backend=None)), "ModelOptNvFp4FusedMoE"),
        (_layer(use_ep=True), "EP"),
    ],
)
def test_mismatch_or_unsupported_backend_is_rejected(bad, match):
    with pytest.raises(ValueError, match=match):
        inst.check_pool_layers([("a", _layer()), ("b", bad)])


def test_planner_width_is_bounded_by_the_decode_lane_cap():
    top_k = 10
    cap_tokens = inst.MAX_DECODE_LANES // top_k
    for requested, expected in ((1, 1), (cap_tokens, cap_tokens), (256, cap_tokens)):
        tokens = max(1, min(requested, cap_tokens))
        assert tokens == expected
        assert inst._next_power_of_two(top_k * tokens) <= 2 * inst.MAX_DECODE_LANES


def test_bank_beyond_the_marlin_weight_index_is_rejected():
    # 4 MiB w13 rows (GLM-5.3-Flash, TP2): 8192 bank rows fill the 32 GiB.
    source = {
        "w13_weight": torch.empty(1, 4 << 20, dtype=torch.uint8, device="meta"),
        "w2_weight": torch.empty(1, 2 << 20, dtype=torch.uint8, device="meta"),
    }
    inst._check_bank_size(source, slots=193, num_layers=42, staging=64)
    with pytest.raises(ValueError, match="w13_weight .* <= 193"):
        inst._check_bank_size(source, slots=194, num_layers=42, staging=64)


def test_top_k_beyond_the_lane_cap_is_rejected():
    assert inst.install_expert_pool(torch.nn.Module(), torch.device("cpu")) is None
    for bad in (0, inst.MAX_DECODE_LANES + 1):
        with pytest.raises(ValueError, match="0 < top_k"):
            inst._check_top_k(bad)
    inst._check_top_k(inst.MAX_DECODE_LANES)


def _routed(rows=4, top_k=2, use_ep=False, dp_size=1, sp=False, method=None):
    # The layer-construction guard runs before create_weights; it reads only
    # these attributes, so a namespace stands in for the RoutedExperts.
    return SimpleNamespace(
        _moe_expert_pool_rows=rows,
        quant_method=method or _method(),
        moe_config=SimpleNamespace(
            experts_per_token=top_k,
            moe_parallel_config=SimpleNamespace(
                use_ep=use_ep,
                ep_size=2 if use_ep else 1,
                dp_size=dp_size,
                is_sequence_parallel=sp,
            ),
        ),
    )


def test_layer_guard_accepts_the_supported_geometry():
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    RoutedExperts._validate_expert_pool_supported(_routed())


@pytest.mark.parametrize(
    "bad, match",
    [
        (_routed(rows=1), "fewer than the 2 experts"),
        (_routed(use_ep=True), "expert parallelism"),
        (_routed(dp_size=2), "data parallelism"),
        (_routed(sp=True), "sequence parallelism"),
        (_routed(method=_method(NvFp4MoeBackend.VLLM_CUTLASS)), "Marlin"),
    ],
)
def test_layer_guard_rejects_before_any_weight_is_allocated(bad, match):
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    with pytest.raises(ValueError, match=match):
        RoutedExperts._validate_expert_pool_supported(bad)


def test_layer_guard_keeps_unquantized_layers_on_the_gpu():
    # e.g. the BF16 MTP head of an NVFP4 checkpoint: served outside the pool.
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )

    layer = _routed(method=UnquantizedFusedMoEMethod.__new__(UnquantizedFusedMoEMethod))
    RoutedExperts._validate_expert_pool_supported(layer)
    assert layer._moe_expert_pool_rows == 0
