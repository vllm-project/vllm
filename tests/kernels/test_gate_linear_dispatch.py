# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-free eligibility tests for GateLinear's router GEMMs.

These assert the dispatch flags directly, so they run device-free by mocking
the platform predicates. ``allow_cublas_router_gemm`` selects the
bf16xbf16->fp32 ``torch.mm`` epilogue, while ``allow_fp32_router_gemm`` selects
the CUDA or gfx950 low-M kernel with fp32 weights and output.

The ROCm branch is guarded on ``not bias`` because ``torch.mm`` has no bias
term; a biased gate must fall back so the bias is not silently dropped.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.router.gate_linear as gate_linear_mod
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.platforms.interface import DeviceCapability


def _make_gate(
    monkeypatch,
    *,
    is_rocm: bool,
    is_cuda: bool = False,
    bias: bool = False,
    params_dtype: torch.dtype = torch.bfloat16,
    out_dtype: torch.dtype | None = torch.float32,
    input_size: int = 2048,
    output_size: int = 64,
    on_gfx950: bool = False,
    parallel_world_size: int = 1,
    device_capability: tuple[int, int] | None = None,
    force_fp32_compute: bool = False,
) -> GateLinear:
    """Build a GateLinear with platform predicates mocked, no GPU needed."""
    for target in (
        "vllm.model_executor.layers.linear",
        "vllm.model_executor.parameter",
    ):
        monkeypatch.setattr(
            f"{target}.get_tensor_model_parallel_rank",
            lambda: 0,
        )
        monkeypatch.setattr(
            f"{target}.get_tensor_model_parallel_world_size",
            lambda: parallel_world_size,
        )

    platform = gate_linear_mod.current_platform
    monkeypatch.setattr(platform, "is_cuda", lambda: is_cuda)
    monkeypatch.setattr(platform, "is_rocm", lambda: is_rocm)
    monkeypatch.setattr(
        type(platform),
        "get_device_capability",
        classmethod(
            lambda cls, device_id=0: (
                DeviceCapability(*device_capability) if device_capability else None
            )
        ),
    )
    if is_rocm:
        # The ROCm module queries the GCN architecture at import time.
        with monkeypatch.context() as mp:
            mp.setattr(
                torch.cuda,
                "get_device_properties",
                lambda device: SimpleNamespace(gcnArchName="gfx950"),
            )
            import vllm.platforms.rocm as rocm_platform

        monkeypatch.setattr(rocm_platform, "on_gfx950", lambda: on_gfx950)

    return GateLinear(
        input_size=input_size,
        output_size=output_size,
        bias=bias,
        out_dtype=out_dtype,
        params_dtype=params_dtype,
        force_fp32_compute=force_fp32_compute,
    )


@pytest.mark.parametrize("input_size,output_size", [(3072, 256), (6144, 128)])
@pytest.mark.parametrize(
    "device_capability,enabled",
    [
        ((9, 0), True),
        ((10, 0), True),
        ((10, 3), True),
        ((12, 0), True),
        ((12, 1), False),
        ((8, 6), False),
    ],
)
def test_cuda_fp32_router_architecture_gate(
    monkeypatch, input_size, output_size, device_capability, enabled
):
    """Admit exact SM120 without extending admission to the whole family."""
    gate = _make_gate(
        monkeypatch,
        is_rocm=False,
        is_cuda=True,
        params_dtype=torch.float32,
        input_size=input_size,
        output_size=output_size,
        device_capability=device_capability,
    )
    assert gate.allow_fp32_router_gemm == enabled
    if device_capability == (12, 0):
        assert not gate.allow_specialized_router_gemm
        assert not gate.allow_ll_bf16_gemm
        assert not gate.allow_bf16x3_router_gemm
        assert not gate.allow_cublas_router_gemm


@pytest.mark.parametrize(
    "overrides",
    [
        {"is_cuda": False},
        {"params_dtype": torch.bfloat16},
        {"params_dtype": torch.float16},
        {"input_size": 4096},
        {"output_size": 256},
        {"bias": True},
    ],
)
def test_sm120_fp32_router_rejects_unsupported_configs(monkeypatch, overrides):
    kwargs = dict(
        is_rocm=False,
        is_cuda=True,
        params_dtype=torch.float32,
        input_size=6144,
        output_size=128,
        device_capability=(12, 0),
    )
    gate = _make_gate(monkeypatch, **(kwargs | overrides))
    assert not gate.allow_fp32_router_gemm


def test_sm120_force_fp32_compute_preserves_fp32_weights(monkeypatch):
    gate = _make_gate(
        monkeypatch,
        is_rocm=False,
        is_cuda=True,
        params_dtype=torch.bfloat16,
        input_size=6144,
        output_size=128,
        device_capability=(12, 0),
        force_fp32_compute=True,
    )
    assert gate.weight.dtype == torch.float32
    assert gate.allow_fp32_router_gemm
    assert not gate.allow_specialized_router_gemm


@pytest.mark.parametrize("deferred", [False, True])
def test_sm120_fp32_router_preserves_requested_output_dtype(monkeypatch, deferred):
    """The FP32-compute path must honor explicit and deferred output casts."""
    gate = _make_gate(
        monkeypatch,
        is_rocm=False,
        is_cuda=True,
        params_dtype=torch.float32,
        out_dtype=None if deferred else torch.bfloat16,
        input_size=6144,
        output_size=128,
        device_capability=(12, 0),
    )
    if deferred:
        gate.set_out_dtype(torch.bfloat16)
    # Run the real large-M dispatch on CPU, without CUDA op registration.
    monkeypatch.setattr(
        torch.ops.vllm,
        "fp32_router_gemm_dispatch",
        gate_linear_mod.fp32_router_gemm_dispatch_impl,
    )
    with torch.no_grad():
        gate.weight.fill_(1.0 / 6144)
    x = torch.ones(33, 6144, dtype=torch.bfloat16)
    output, bias = gate(x)
    assert bias is None
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output, torch.ones(33, 128, dtype=torch.bfloat16))


def test_rocm_no_bias_bf16_fp32_enables_fused_gemm(monkeypatch):
    gate = _make_gate(monkeypatch, is_rocm=True, bias=False)
    assert not gate.allow_specialized_router_gemm
    assert gate.allow_cublas_router_gemm


def test_rocm_bias_disables_fused_gemm(monkeypatch):
    # torch.mm cannot add a bias, so a biased gate must not take the fused path.
    gate = _make_gate(monkeypatch, is_rocm=True, bias=True)
    assert not gate.allow_cublas_router_gemm


def test_rocm_fp32_weight_disables_fused_gemm(monkeypatch):
    gate = _make_gate(monkeypatch, is_rocm=True, params_dtype=torch.float32)
    assert not gate.allow_cublas_router_gemm


def test_rocm_non_fp32_out_dtype_disables_fused_gemm(monkeypatch):
    gate = _make_gate(monkeypatch, is_rocm=True, out_dtype=torch.bfloat16)
    assert not gate.allow_cublas_router_gemm


def test_non_rocm_non_cuda_disables_fused_gemm(monkeypatch):
    # Neither the CUDA specialized path nor the ROCm branch applies.
    gate = _make_gate(monkeypatch, is_rocm=False, is_cuda=False)
    assert not gate.allow_cublas_router_gemm


def test_rocm_set_out_dtype_enables_fused_gemm(monkeypatch):
    gate = _make_gate(monkeypatch, is_rocm=True, bias=False, out_dtype=None)
    assert not gate.allow_cublas_router_gemm
    gate.set_out_dtype(torch.float32)
    assert gate.allow_cublas_router_gemm


def test_rocm_set_out_dtype_respects_bias_guard(monkeypatch):
    gate = _make_gate(monkeypatch, is_rocm=True, bias=True, out_dtype=None)
    gate.set_out_dtype(torch.float32)
    assert not gate.allow_cublas_router_gemm


@pytest.mark.parametrize(
    ("input_size", "output_size"),
    [(3072, 256), (4096, 8), (4096, 192), (6144, 128), (6144, 256)],
)
def test_rocm_gfx950_enables_fp32_router_gemm(
    monkeypatch, input_size: int, output_size: int
) -> None:
    gate = _make_gate(
        monkeypatch,
        is_rocm=True,
        params_dtype=torch.float32,
        input_size=input_size,
        output_size=output_size,
        on_gfx950=True,
    )
    assert gate.allow_fp32_router_gemm
    assert not gate.allow_cublas_router_gemm


@pytest.mark.parametrize("ep_size", [2, 4, 8])
def test_rocm_fp32_router_gemm_is_replicated_across_ep_sizes(
    monkeypatch, ep_size: int
) -> None:
    gate = _make_gate(
        monkeypatch,
        is_rocm=True,
        params_dtype=torch.float32,
        input_size=6144,
        output_size=128,
        on_gfx950=True,
        parallel_world_size=ep_size,
    )
    assert gate.weight.shape == (128, 6144)
    assert gate.allow_fp32_router_gemm


@pytest.mark.parametrize(
    ("input_size", "output_size", "params_dtype", "bias", "on_gfx950"),
    [
        pytest.param(6144, 128, torch.float32, False, False, id="gfx942"),
        pytest.param(2048, 64, torch.float32, False, True, id="shape"),
        pytest.param(6144, 128, torch.bfloat16, False, True, id="bf16-weight"),
        pytest.param(6144, 128, torch.float32, True, True, id="bias"),
    ],
)
def test_rocm_fp32_router_gemm_rejects_unsupported_configs(
    monkeypatch,
    input_size: int,
    output_size: int,
    params_dtype: torch.dtype,
    bias: bool,
    on_gfx950: bool,
) -> None:
    gate = _make_gate(
        monkeypatch,
        is_rocm=True,
        params_dtype=params_dtype,
        bias=bias,
        input_size=input_size,
        output_size=output_size,
        on_gfx950=on_gfx950,
    )
    assert not gate.allow_fp32_router_gemm
