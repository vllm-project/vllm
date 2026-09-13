# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch

from vllm.model_executor.kernels.linear import (
    _POSSIBLE_FP8_KERNELS,
    CutlassFP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    MarlinFP8ScaledMMLinearKernel,
    choose_scaled_mm_linear_kernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    cutlass_fp8_supported,
)
from vllm.platforms import PlatformEnum, current_platform

# The CUTLASS FP8 GEMM probe wants SM89 + CUDA 12.4, or SM90+ + CUDA 12.0. Model
# it as a plain capability threshold so the tests do not depend on the toolkit
# the wheel was built against.
_PROBE_MIN_CAPABILITY = 89


def _fake_probe(capability: int) -> bool:
    return capability >= _PROBE_MIN_CAPABILITY


def _patch_probe(monkeypatch, seen: list[int] | None = None) -> None:
    """Force the platform to CUDA and stub the CUTLASS FP8 capability probe."""
    import vllm.model_executor.kernels.linear.scaled_mm.cutlass as cutlass_mod

    def probe(capability: int) -> bool:
        if seen is not None:
            seen.append(capability)
        return _fake_probe(capability)

    monkeypatch.setattr(cutlass_mod.current_platform, "_enum", PlatformEnum.CUDA)
    monkeypatch.setattr(cutlass_mod.ops, "cutlass_scaled_mm_supports_fp8", probe)


def _patch_device(monkeypatch, capability: int) -> None:
    """Make every FP8 kernel gate agree on one synthetic CUDA device.

    `choose_scaled_mm_linear_kernel` passes `compute_capability` to each
    candidate, but several candidates ignore it and query the platform
    directly, so those queries have to be pinned as well for the selection to
    be reproducible off the modelled hardware.
    """
    _patch_probe(monkeypatch)
    platform = current_platform

    monkeypatch.setattr(
        platform,
        "has_device_capability",
        lambda required, device_id=0: (
            capability
            >= (
                required
                if isinstance(required, int)
                else required[0] * 10 + required[1]
            )
        ),
    )
    monkeypatch.setattr(
        platform,
        "is_device_capability_family",
        lambda family, device_id=0: capability // 10 == family // 10,
    )
    monkeypatch.setattr(platform, "supports_fp8", lambda: _fake_probe(capability))


def _fp8_per_tensor_config() -> FP8ScaledMMLinearLayerConfig:
    return FP8ScaledMMLinearLayerConfig(
        activation_quant_key=kFp8StaticTensorSym,
        weight_quant_key=kFp8StaticTensorSym,
        weight_shape=(2048, 2048),
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )


@pytest.mark.parametrize(
    ("compute_capability", "expected"),
    [(75, False), (80, False), (86, False), (89, True), (90, True), (120, True)],
)
def test_cutlass_fp8_is_supported_gates_on_compute_capability(
    monkeypatch, compute_capability: int, expected: bool
) -> None:
    """The kernel must decline capabilities the CUTLASS FP8 probe rejects.

    Without this gate it wins auto-selection on SM75/80/86, where FP8 Marlin is
    the only working FP8 path.
    """
    _patch_probe(monkeypatch)

    is_supported, reason = CutlassFP8ScaledMMLinearKernel.is_supported(
        compute_capability
    )

    assert is_supported is expected
    if expected:
        assert reason is None
    else:
        assert reason is not None
        assert "CUTLASS FP8" in reason
        assert str(compute_capability) in reason


def test_cutlass_fp8_is_supported_queries_device_when_capability_omitted(
    monkeypatch,
) -> None:
    """A caller that omits the capability must not bypass the gate."""
    seen: list[int] = []
    _patch_probe(monkeypatch, seen)

    CutlassFP8ScaledMMLinearKernel.is_supported()

    capability = current_platform.get_device_capability()
    assert seen == [-1 if capability is None else capability.to_int()]


def test_pre_ada_fp8_selection_falls_through_to_marlin(monkeypatch) -> None:
    """The regression this PR fixes, at the selector level."""
    _patch_device(monkeypatch, 80)

    chosen = choose_scaled_mm_linear_kernel(
        _fp8_per_tensor_config(), _POSSIBLE_FP8_KERNELS, compute_capability=80
    )

    assert chosen is MarlinFP8ScaledMMLinearKernel


def test_pre_ada_fp8_selection_picks_cutlass_without_the_gate(monkeypatch) -> None:
    """Control: restoring the CUDA-only check reproduces the regression.

    This is what makes the assertion above meaningful - the fall-through to
    Marlin is caused by the gate, not by some other candidate declining.
    """
    _patch_device(monkeypatch, 80)
    monkeypatch.setattr(
        CutlassFP8ScaledMMLinearKernel,
        "is_supported",
        classmethod(lambda cls, compute_capability=None: (True, None)),
    )

    chosen = choose_scaled_mm_linear_kernel(
        _fp8_per_tensor_config(), _POSSIBLE_FP8_KERNELS, compute_capability=80
    )

    assert chosen is CutlassFP8ScaledMMLinearKernel


def test_supported_capability_still_selects_cutlass(monkeypatch) -> None:
    """The gate must not over-reject: SM90 keeps the CUTLASS kernel."""
    _patch_device(monkeypatch, 90)

    chosen = choose_scaled_mm_linear_kernel(
        _fp8_per_tensor_config(), _POSSIBLE_FP8_KERNELS, compute_capability=90
    )

    assert chosen is CutlassFP8ScaledMMLinearKernel


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
def test_cutlass_fp8_is_supported_matches_runtime_probe() -> None:
    """Unmocked: the gate agrees with the probe the rest of vLLM already uses."""
    is_supported, _ = CutlassFP8ScaledMMLinearKernel.is_supported()

    assert is_supported == cutlass_fp8_supported()
