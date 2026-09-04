# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

from vllm.model_executor.warmup.kernel_warmup import (
    _run_flashinfer_deferred_moe_autotune,
)

pytestmark = pytest.mark.cpu_test


class _FakeMoERunner:
    moe_config: Any
    routed_experts: Any


def _make_deferred_moe(hidden_dim: int = 3584):
    import torch

    moe_config = SimpleNamespace(
        use_deferred_moe_finalize=True,
        defer_moe_finalize_max_num_tokens=128,
        should_defer_moe_finalize=Mock(return_value=True),
        hidden_dim=hidden_dim,
        num_experts=896,
        in_dtype=torch.bfloat16,
        router_logits_dtype=torch.bfloat16,
    )
    moe_kernel = SimpleNamespace(supports_deferred_moe_finalize=Mock(return_value=True))
    routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(is_monolithic=True, moe_kernel=moe_kernel),
        w13_weight=torch.empty(0),
        forward_monolithic=Mock(),
    )
    moe = _FakeMoERunner()
    moe.moe_config = moe_config
    moe.routed_experts = routed_experts
    return moe


def _run_deferred_moe_autotune(modules, buckets):
    runner = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
        get_model=Mock(
            return_value=SimpleNamespace(modules=Mock(return_value=modules))
        ),
    )
    with (
        patch("vllm.model_executor.layers.fused_moe.MoERunner", _FakeMoERunner),
        patch(
            "vllm.utils.flashinfer.flashinfer_get_hybrid_num_tokens_buckets",
            return_value=buckets,
        ) as get_buckets,
        patch("vllm.utils.flashinfer.autotune") as autotune,
    ):
        _run_flashinfer_deferred_moe_autotune(runner)
    return get_buckets, autotune


def test_flashinfer_autotune_directly_tunes_deferred_moe_buckets():
    moe = _make_deferred_moe()
    buckets = (1, 2, 4, 8, 16, 32, 64, 128)

    get_buckets, autotune = _run_deferred_moe_autotune([object(), moe], buckets)

    get_buckets.assert_called_once_with(128)
    autotune.assert_called_once_with(tuning_buckets=buckets)
    moe_kernel = moe.routed_experts.quant_method.moe_kernel
    moe.moe_config.should_defer_moe_finalize.assert_called_once_with(128)
    moe_kernel.supports_deferred_moe_finalize.assert_called_once_with()
    moe.routed_experts.forward_monolithic.assert_called_once()
    call = moe.routed_experts.forward_monolithic.call_args.kwargs
    assert call["x"].shape == (128, 3584)
    assert call["router_logits"].shape == (128, 896)


def test_flashinfer_autotune_tunes_each_deferred_moe_geometry_once():
    # Identical layers share one cache key, so only the first needs a
    # dispatcher call; a differently shaped layer still gets its own.
    same_a, same_b = _make_deferred_moe(), _make_deferred_moe()
    other = _make_deferred_moe(hidden_dim=1792)
    buckets = (1, 2, 4, 8, 16, 32, 64, 128)

    _run_deferred_moe_autotune([same_a, same_b, other], buckets)

    same_a.routed_experts.forward_monolithic.assert_called_once()
    same_b.routed_experts.forward_monolithic.assert_not_called()
    other.routed_experts.forward_monolithic.assert_called_once()
