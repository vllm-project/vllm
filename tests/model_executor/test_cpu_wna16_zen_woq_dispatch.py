# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the W4A16 zentorch dispatch path in
``CPUWNA16LinearKernel``.

Coverage:
* ``_zentorch_woq_eligible``           — eligibility predicate
                                         (Zen + ops + CT-GPTQ format).
* ``_process_weights_for_zentorch_woq`` — weight repack + scale prep,
                                         dedicated ``_zentorch_woq_*``
                                         attribute caching, original
                                         storage release.
* ``apply_weights``                    — op dispatch + bias / zp
                                         pass-through.
* ``process_weights_after_loading``    — fall-through to the legacy
                                         ``cpu_gemm_wna16`` path when
                                         ineligible, idempotency.

Run ``pytest tests/model_executor/test_cpu_wna16_zen_woq_dispatch.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.quantization._zentorch_helpers import zentorch_ops_mock  # noqa: F401
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.mixed_precision import (
    CPUWNA16LinearKernel,
    MPLinearLayerConfig,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

try:
    from compressed_tensors.compressors.pack_quantized.helpers import (
        pack_to_int32,
    )
except ImportError:
    pytest.skip("compressed_tensors is required", allow_module_level=True)


# Zen CPU dispatch unit tests for compressed-tensors W4A16 GPTQ.


# ----- Helpers -------------------------------------------------------------


def _param_with_dims(
    tensor: torch.Tensor, *, input_dim: int, output_dim: int, packed_dim: int
) -> torch.nn.Parameter:
    """``nn.Parameter`` plus metadata read by ``CPUWNA16LinearKernel``."""
    p = torch.nn.Parameter(tensor, requires_grad=False)
    p.input_dim = input_dim  # type: ignore[attr-defined]
    p.output_dim = output_dim  # type: ignore[attr-defined]
    p.packed_dim = packed_dim  # type: ignore[attr-defined]
    return p


def _make_ct_symmetric_layer(
    *,
    n: int = 32,
    k: int = 32,
    weight_int8: torch.Tensor | None = None,
    scales_bf16: torch.Tensor | None = None,
    with_g_idx: bool = False,
) -> torch.nn.Module:
    """Build a minimal CT-style WNA16 layer (``input_dim == packed_dim == 1``).

    A real ``nn.Module`` (not ``SimpleNamespace``) is required because
    ``register_parameter`` is used to attach the weight tensors with their
    quantization-metadata attributes.
    """
    if weight_int8 is None:
        g = torch.Generator(device="cpu").manual_seed(0)
        weight_int8 = torch.randint(-7, 8, (n, k), dtype=torch.int8, generator=g)
    if scales_bf16 is None:
        g = torch.Generator(device="cpu").manual_seed(1)
        scales_bf16 = torch.rand(n, 1, dtype=torch.bfloat16, generator=g) * 0.02 + 0.01

    assert weight_int8.shape == (n, k)
    assert scales_bf16.shape == (n, 1)
    packed = pack_to_int32(weight_int8, 4, packed_dim=1)
    assert packed.shape == (n, k // 8)

    layer = torch.nn.Module()
    layer.register_parameter(
        "weight_packed",
        _param_with_dims(packed, input_dim=1, output_dim=0, packed_dim=1),
    )
    layer.register_parameter(
        "weight_scale",
        _param_with_dims(scales_bf16.clone(), output_dim=0, input_dim=1, packed_dim=0),
    )
    if with_g_idx:
        layer.register_parameter(
            "weight_g_idx",
            torch.nn.Parameter(torch.zeros(k, dtype=torch.int32), requires_grad=False),
        )
    return layer


def _make_marlin_style_layer(*, n: int = 32, k: int = 32) -> torch.nn.Module:
    """Build a Marlin-style layer (``input_dim == packed_dim == 0``).

    The zentorch fast path only handles the CT layout, so this layer must be
    rejected by the eligibility predicate.
    """
    g = torch.Generator(device="cpu").manual_seed(0)
    weight_int8 = torch.randint(-7, 8, (k, n), dtype=torch.int8, generator=g)
    scales_bf16 = torch.rand(1, n, dtype=torch.bfloat16, generator=g) * 0.02 + 0.01
    packed = pack_to_int32(weight_int8, 4, packed_dim=0)

    layer = torch.nn.Module()
    layer.register_parameter(
        "weight_packed",
        _param_with_dims(packed, input_dim=0, output_dim=1, packed_dim=0),
    )
    layer.register_parameter(
        "weight_scale",
        _param_with_dims(scales_bf16, output_dim=1, input_dim=0, packed_dim=0),
    )
    return layer


def _make_kernel(
    *,
    weight_type=scalar_types.uint4b8,
    full_shape: tuple[int, int] = (32, 32),
    partition_shape: tuple[int, int] = (32, 32),
    group_size: int = 32,
    zero_points: bool = False,
    has_g_idx: bool = False,
) -> CPUWNA16LinearKernel:
    """Construct a kernel without invoking ``can_implement`` (which gates on
    ``current_platform.is_cpu()`` and is unrelated to the dispatch logic
    exercised here)."""
    cfg = MPLinearLayerConfig(
        full_weight_shape=full_shape,
        partition_weight_shape=partition_shape,
        weight_type=weight_type,
        act_type=torch.bfloat16,
        group_size=group_size,
        zero_points=zero_points,
        has_g_idx=has_g_idx,
    )
    kernel = CPUWNA16LinearKernel.__new__(CPUWNA16LinearKernel)
    kernel.config = cfg
    kernel.w_q_name = "weight_packed"
    kernel.w_s_name = "weight_scale"
    kernel.w_zp_name = None
    kernel.w_gidx_name = "weight_g_idx" if has_g_idx else None
    return kernel


def _ct_repack_spy(counter: dict[str, int]):
    """Build a spy for ``zentorch_woq_repack_weight`` that records call
    counts and returns a tensor whose ``.t()`` matches the CT ``(out, in//8)``
    packing the kernel caches as ``_zentorch_woq_packed``."""

    def _repack(weight: torch.Tensor) -> torch.Tensor:
        counter["repack_calls"] = counter.get("repack_calls", 0) + 1
        return pack_to_int32(weight, 4, packed_dim=1).t()

    return _repack


def _patch_woq_repack(monkeypatch: pytest.MonkeyPatch, spy):
    """Override ``torch.ops.zentorch.zentorch_woq_repack_weight.default``.

    The kernel binds ``.default`` at call time, so wrapping the spy in a
    ``SimpleNamespace(default=spy)`` is the lightest-weight way to redirect
    dispatch without touching the ``torch.library`` registry."""
    monkeypatch.setattr(
        torch.ops.zentorch,
        "zentorch_woq_repack_weight",
        SimpleNamespace(default=spy),
    )


def _patch_woq_linear(monkeypatch: pytest.MonkeyPatch, spy):
    """Override ``torch.ops.zentorch.zentorch_woq_linear.default``."""
    monkeypatch.setattr(
        torch.ops.zentorch,
        "zentorch_woq_linear",
        SimpleNamespace(default=spy),
    )


# ----- _zentorch_woq_eligible: predicate -----------------------------------


def test_eligible_predicate_zen_cpu_ct_gptq_returns_true(
    monkeypatch,
    zentorch_ops_mock,  # noqa: F811
):
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)

    layer = _make_ct_symmetric_layer()
    kernel = _make_kernel()
    assert kernel._zentorch_woq_eligible(layer) is True


def test_eligible_predicate_off_zen_returns_false(
    monkeypatch,
    zentorch_ops_mock,  # noqa: F811
):
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: False)

    layer = _make_ct_symmetric_layer()
    kernel = _make_kernel()
    assert kernel._zentorch_woq_eligible(layer) is False


def test_eligible_predicate_op_missing_returns_false(monkeypatch):
    """Zen CPU but ``zentorch_woq_repack_weight`` is not registered."""
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)
    if hasattr(torch.ops, "zentorch") and hasattr(
        torch.ops.zentorch, "zentorch_woq_repack_weight"
    ):
        pytest.skip(
            "real zentorch build registers the op; can't test the missing-op "
            "fallback hermetically here"
        )

    layer = _make_ct_symmetric_layer()
    kernel = _make_kernel()
    assert kernel._zentorch_woq_eligible(layer) is False


def test_eligible_predicate_with_g_idx_returns_false(
    monkeypatch,
    zentorch_ops_mock,  # noqa: F811
):
    """GPTQ with explicit group index isn't supported by the zentorch op."""
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)

    layer = _make_ct_symmetric_layer(with_g_idx=True)
    kernel = _make_kernel(has_g_idx=True)
    assert kernel._zentorch_woq_eligible(layer) is False


def test_eligible_predicate_marlin_format_returns_false(
    monkeypatch,
    zentorch_ops_mock,  # noqa: F811
):
    """Marlin packs ``packed_dim == input_dim == 0``: not the CT layout the
    zentorch fast path expects."""
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)

    layer = _make_marlin_style_layer()
    kernel = _make_kernel()
    assert kernel._zentorch_woq_eligible(layer) is False


# ----- process_weights_after_loading: success paths ------------------------


def test_process_weights_after_loading_zen_caches_zentorch_attrs(
    monkeypatch,
    zentorch_ops_mock,  # noqa: F811
):
    """CT + Zen + ops present: zentorch repack runs; layer gets dedicated
    ``_zentorch_woq_*`` attributes and the original weight storage is freed.
    """
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)

    counter: dict[str, int] = {}
    _patch_woq_repack(monkeypatch, _ct_repack_spy(counter))

    n, k = 32, 32
    layer = _make_ct_symmetric_layer(n=n, k=k)
    kernel = _make_kernel()
    kernel.process_weights_after_loading(layer)

    assert counter["repack_calls"] == 1
    assert getattr(layer, "_zentorch_processed_weights", False) is True
    assert getattr(layer, "_zentorch_kind", None) == "compressed_tensors_w4a16_gptq"

    # Repacked tensors live on dedicated ``_zentorch_woq_*`` attributes.
    assert layer._zentorch_woq_packed.shape == (n, k // 8)
    assert layer._zentorch_woq_scale.shape == (1, n)
    assert layer._zentorch_woq_zero_point is None

    # The original Parameters are kept (so attribute access doesn't break)
    # but their storage is released.
    assert layer.weight_packed.shape == (0,)
    assert layer.weight_scale.shape == (0,)


def test_process_weights_after_loading_is_idempotent(
    monkeypatch,
    zentorch_ops_mock,  # noqa: F811
):
    """A second call after a successful zentorch path should be a no-op:
    the repack op is not invoked again."""
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)

    counter: dict[str, int] = {}
    _patch_woq_repack(monkeypatch, _ct_repack_spy(counter))

    layer = _make_ct_symmetric_layer()
    kernel = _make_kernel()
    kernel.process_weights_after_loading(layer)
    kernel.process_weights_after_loading(layer)

    assert counter["repack_calls"] == 1


# ----- process_weights_after_loading: fall-through paths preserve weight ---


def test_process_weights_after_loading_off_zen_uses_legacy_path(
    monkeypatch,
    zentorch_ops_mock,  # noqa: F811
):
    """Off Zen → legacy ``_process_gptq_weights`` runs; no zentorch attrs."""
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: False)

    counter: dict[str, int] = {}
    _patch_woq_repack(monkeypatch, _ct_repack_spy(counter))

    layer = _make_ct_symmetric_layer()
    kernel = _make_kernel()
    kernel.process_weights_after_loading(layer)

    assert counter.get("repack_calls", 0) == 0
    assert not getattr(layer, "_zentorch_processed_weights", False)
    assert not hasattr(layer, "_zentorch_woq_packed")
    # Legacy path sets ``isa_hint`` on the layer and keeps the weight buffer.
    assert hasattr(layer, "isa_hint")
    assert layer.weight_packed.numel() > 0


def test_process_weights_after_loading_op_missing_uses_legacy_path(monkeypatch):
    """Zen CPU but ``zentorch_woq_repack_weight`` is not registered."""
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)
    if hasattr(torch.ops, "zentorch") and hasattr(
        torch.ops.zentorch, "zentorch_woq_repack_weight"
    ):
        pytest.skip(
            "real zentorch build registers the op; can't test the missing-op "
            "fallback hermetically here"
        )

    layer = _make_ct_symmetric_layer()
    kernel = _make_kernel()
    kernel.process_weights_after_loading(layer)

    assert not getattr(layer, "_zentorch_processed_weights", False)
    assert not hasattr(layer, "_zentorch_woq_packed")
    assert hasattr(layer, "isa_hint")


# ----- apply(): zentorch dispatch ------------------------------------------


def test_apply_dispatches_to_zentorch_woq_linear(
    monkeypatch,
    zentorch_ops_mock,  # noqa: F811
):
    """When the layer has been processed for zen, ``apply_weights`` routes
    to ``zentorch_woq_linear`` (rather than ``cpu_gemm_wna16``)."""
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)

    counter: dict[str, int] = {}
    _patch_woq_repack(monkeypatch, _ct_repack_spy(counter))

    n, k = 32, 32
    layer = _make_ct_symmetric_layer(n=n, k=k)
    kernel = _make_kernel()
    kernel.process_weights_after_loading(layer)

    captured: dict = {}

    def linear_spy(*args, **kwargs):
        # zentorch schema:
        #   zentorch::zentorch_woq_linear(
        #       Tensor input, Tensor weight_packed, Tensor weight_scale,
        #       Tensor? weight_zero_point, Tensor? bias=None
        #   ) -> Tensor
        # Bind via *args/**kwargs so we don't depend on a brittle subset.
        captured["called"] = True
        captured["args"] = args
        captured["kwargs"] = kwargs
        inp = args[0] if args else kwargs["input"]
        out_features = layer._zentorch_woq_scale.shape[1]
        return torch.zeros(
            inp.shape[:-1] + (out_features,),
            dtype=inp.dtype,
            device=inp.device,
        )

    _patch_woq_linear(monkeypatch, linear_spy)

    x = torch.randn(4, k, dtype=torch.bfloat16)
    out = kernel.apply_weights(layer, x, None)

    assert captured.get("called") is True
    assert out.shape == (4, n)


def test_apply_passes_bias_and_zp_through_to_zentorch_woq_linear(
    monkeypatch,
    zentorch_ops_mock,  # noqa: F811
):
    """``bias`` and ``zero_point`` (``None`` for symmetric) reach the op
    unmodified."""
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)

    counter: dict[str, int] = {}
    _patch_woq_repack(monkeypatch, _ct_repack_spy(counter))

    n, k = 32, 32
    layer = _make_ct_symmetric_layer(n=n, k=k)
    kernel = _make_kernel()
    kernel.process_weights_after_loading(layer)

    captured: dict = {}

    def linear_spy(inp, weight_packed, weight_scale, weight_zp, bias=None):
        captured["bias"] = bias
        captured["weight_zp"] = weight_zp
        return torch.zeros(inp.shape[:-1] + (n,), dtype=inp.dtype, device=inp.device)

    _patch_woq_linear(monkeypatch, linear_spy)

    x = torch.randn(4, k, dtype=torch.bfloat16)
    bias = torch.randn(n, dtype=torch.bfloat16)
    kernel.apply_weights(layer, x, bias)

    assert captured["bias"] is bias
    assert captured["weight_zp"] is None


def test_apply_unprocessed_layer_falls_back_to_cpu_gemm_wna16(
    monkeypatch,
    zentorch_ops_mock,  # noqa: F811
):
    """When ``_zentorch_processed_weights`` is not set, ``apply_weights``
    falls back to the legacy ``ops.cpu_gemm_wna16`` path."""
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: False)

    layer = _make_ct_symmetric_layer()
    kernel = _make_kernel()
    kernel.process_weights_after_loading(layer)
    assert not getattr(layer, "_zentorch_processed_weights", False)

    woq_calls: dict[str, int] = {"linear_calls": 0}

    def linear_spy(*args, **kwargs):
        woq_calls["linear_calls"] += 1
        inp = args[0] if args else kwargs["input"]
        return torch.zeros(inp.shape[:-1] + (32,), dtype=inp.dtype)

    _patch_woq_linear(monkeypatch, linear_spy)

    cpu_gemm_calls: dict[str, int] = {"calls": 0}

    def cpu_gemm_spy(*args, **kwargs):
        cpu_gemm_calls["calls"] += 1
        inp: torch.Tensor = kwargs["input"]
        return torch.zeros(inp.shape[:-1] + (32,), dtype=inp.dtype)

    monkeypatch.setattr(ops, "cpu_gemm_wna16", cpu_gemm_spy)

    x = torch.randn(4, 32, dtype=torch.bfloat16)
    kernel.apply_weights(layer, x, None)

    assert woq_calls["linear_calls"] == 0
    assert cpu_gemm_calls["calls"] == 1


if __name__ == "__main__":
    pytest.main([__file__])
