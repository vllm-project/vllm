# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the flashinfer moe_ep backend plumbing.

Everything here runs without a GPU or a flashinfer install: the flashinfer
modules the helpers import lazily are replaced with capture fakes.
"""

import sys
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.config.kernel import (
    FLASHINFER_MOE_EP_BACKENDS,
    MEGA_MOE_BACKENDS,
    validate_flashinfer_moe_ep_model,
)
from vllm.utils.flashinfer_moe_ep import (
    _E2M1_LUT,
    FI_MOE_EP_BACKEND_SPECS,
    _dequant_fp4_ue8m0_gran32,
    build_fi_mega_config,
    fi_moe_ep_backend_spec,
    make_fi_moe_ep_bootstrap,
    megakernel_runtime_requirements,
)


@dataclass
class _FakeBootstrapConfig:
    world_size: int
    rank: int
    process_group: Any = None
    auto_bootstrap: bool = True
    device: int | None = field(default=None, kw_only=True)


@dataclass
class _FakeDeepGemmMegaMoeConfig:
    intermediate_size: int
    top_k: int
    activation_clamp: float | None
    fast_math: bool


@dataclass
class _FakeNvfp4CutedslMegaMoeConfig:
    intermediate_size: int
    top_k: int
    activation_clamp: float | None
    fast_math: bool


@dataclass
class _FakeSm90PullFp8CutedslMegaMoeConfig:
    intermediate_size: int
    top_k: int
    kind: str
    fp8_scale_mode: str
    fp8_accum_mode: str
    gate_up_clamp: float | None
    fast_math: bool


@dataclass
class _FakeMegaConfig:
    megakernel: Any
    preprocess_weights: bool
    quantize_input: bool


@pytest.fixture
def fake_flashinfer(monkeypatch):
    """Install a minimal fake flashinfer.moe_ep for the lazy imports."""
    moe_ep = ModuleType("flashinfer.moe_ep")
    core = ModuleType("flashinfer.moe_ep.core")
    runtime = ModuleType("flashinfer.moe_ep.core.runtime")
    flashinfer = ModuleType("flashinfer")
    fake_attrs: dict[ModuleType, dict[str, Any]] = {
        moe_ep: {
            "BootstrapConfig": _FakeBootstrapConfig,
            "DeepGemmMegaMoeConfig": _FakeDeepGemmMegaMoeConfig,
            "Nvfp4CutedslMegaMoeConfig": _FakeNvfp4CutedslMegaMoeConfig,
            "Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig": (
                _FakeSm90PullFp8CutedslMegaMoeConfig
            ),
            "MegaConfig": _FakeMegaConfig,
            "core": core,
        },
        runtime: {"TORCH_DIST": "torch_dist", "NVSHMEM": "nvshmem"},
        flashinfer: {"moe_ep": moe_ep},
        core: {"runtime": runtime},
    }
    for mod, attrs in fake_attrs.items():
        for attr, value in attrs.items():
            setattr(mod, attr, value)

    for name, mod in {
        "flashinfer": flashinfer,
        "flashinfer.moe_ep": moe_ep,
        "flashinfer.moe_ep.core": core,
        "flashinfer.moe_ep.core.runtime": runtime,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return moe_ep


def test_fi_backend_strings_are_registered_mega_moe_backends():
    assert set(FI_MOE_EP_BACKEND_SPECS) == FLASHINFER_MOE_EP_BACKENDS
    assert FLASHINFER_MOE_EP_BACKENDS < MEGA_MOE_BACKENDS


@pytest.mark.parametrize("moe_backend", sorted(FLASHINFER_MOE_EP_BACKENDS))
def test_fi_moe_ep_backend_rejected_for_non_dsv4(moe_backend):
    """An FI moe_ep backend with a non-DSv4 model must fail at config time
    instead of silently falling through to the generic FusedMoE path."""
    with pytest.raises(ValueError, match="only supported for DeepSeek-V4"):
        validate_flashinfer_moe_ep_model(moe_backend, ["MixtralForCausalLM"])


@pytest.mark.parametrize("moe_backend", sorted(FLASHINFER_MOE_EP_BACKENDS))
def test_fi_moe_ep_backend_accepted_for_dsv4(moe_backend):
    validate_flashinfer_moe_ep_model(moe_backend, ["DeepseekV4ForCausalLM"])


@pytest.mark.parametrize(
    "architectures",
    [["KimiK3ForConditionalGeneration"], ["MixtralForCausalLM"]],
)
def test_native_deep_gemm_mega_moe_not_arch_gated(architectures):
    """VLLM's own deep_gemm mega path is not DSv4-only (Kimi K3 uses it);
    models validate their own constraints at construction time."""
    validate_flashinfer_moe_ep_model("deep_gemm_mega_moe", architectures)


def test_non_fi_backend_ignores_architectures():
    validate_flashinfer_moe_ep_model("auto", ["MixtralForCausalLM"])


@pytest.mark.parametrize("moe_backend", sorted(MEGA_MOE_BACKENDS))
def test_all_mega_backends_get_sequence_parallel_moe(moe_backend):
    """Every mega backend must qualify for sequence-parallel MoE at
    TP>1/EP: the predicate once matched only the native backend string,
    which silently ran the fi backends full-batch with an all-reduce on
    every rank — 0.42-0.65x native e2e at TP8."""
    from vllm.models.deepseek_v4.nvidia.model import _use_sequence_parallel

    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            enable_expert_parallel=True,
            tensor_parallel_size=8,
            data_parallel_size=1,
        ),
        kernel_config=SimpleNamespace(moe_backend=moe_backend),
    )
    assert _use_sequence_parallel(vllm_config)


def test_fi_moe_ep_backend_spec_kernel_and_nvshmem_contract():
    dg = fi_moe_ep_backend_spec("flashinfer_moe_ep_mega_deep_gemm")
    assert dg.megakernel == "deep_gemm_mega"
    assert not dg.needs_nvshmem

    cd = fi_moe_ep_backend_spec("flashinfer_moe_ep_mega_cutedsl")
    assert cd.megakernel == "nvfp4_cutedsl"
    assert cd.needs_nvshmem
    assert cd.supported_capabilities == frozenset({(10, 0), (10, 3)})

    sm90 = fi_moe_ep_backend_spec("flashinfer_moe_ep_mega_sm90_fp8")
    assert sm90.megakernel == "sm90_fp8_pull"
    assert sm90.needs_nvshmem
    assert sm90.supported_capabilities == frozenset({(9, 0)})

    with pytest.raises(ValueError, match="not a flashinfer moe_ep backend"):
        fi_moe_ep_backend_spec("deep_gemm_mega_moe")


def test_megakernel_runtime_requirements(fake_flashinfer):
    dg = megakernel_runtime_requirements(
        fi_moe_ep_backend_spec("flashinfer_moe_ep_mega_deep_gemm")
    )
    assert dg == frozenset({"torch_dist"})

    cd = megakernel_runtime_requirements(
        fi_moe_ep_backend_spec("flashinfer_moe_ep_mega_cutedsl")
    )
    assert cd == frozenset({"torch_dist", "nvshmem"})


def test_bootstrap_pins_the_device_vllm_bound(fake_flashinfer, monkeypatch):
    """The runtime must not rederive the device from LOCAL_RANK/rank: under a
    remapped CUDA_VISIBLE_DEVICES that ordinal points at the wrong GPU
    (CUDA_ERROR_ILLEGAL_ADDRESS in the weight transforms). vLLM passes the
    device it already bound via BootstrapConfig.device."""
    import vllm.utils.flashinfer_moe_ep as mod

    pg = object()
    monkeypatch.setattr(
        mod,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=4, rank_in_group=2, device_group=pg),
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 3)

    bootstrap = make_fi_moe_ep_bootstrap()

    assert bootstrap.world_size == 4
    assert bootstrap.rank == 2
    assert bootstrap.process_group is pg
    assert bootstrap.auto_bootstrap is False
    assert bootstrap.device == 3


def test_build_fi_mega_config_selects_kernel_config(fake_flashinfer):
    dg = build_fi_mega_config(
        intermediate_size=2048,
        top_k=8,
        activation_clamp=7.0,
        megakernel="deep_gemm_mega",
    )
    assert isinstance(dg.megakernel, _FakeDeepGemmMegaMoeConfig)
    assert dg.megakernel.intermediate_size == 2048
    assert dg.megakernel.top_k == 8
    assert dg.megakernel.activation_clamp == 7.0
    assert dg.preprocess_weights and dg.quantize_input

    cd = build_fi_mega_config(
        intermediate_size=2048,
        top_k=8,
        activation_clamp=None,
        megakernel="nvfp4_cutedsl",
    )
    assert isinstance(cd.megakernel, _FakeNvfp4CutedslMegaMoeConfig)

    sm90 = build_fi_mega_config(
        intermediate_size=2048,
        top_k=8,
        activation_clamp=10.0,
        megakernel="sm90_fp8_pull",
    )
    assert isinstance(sm90.megakernel, _FakeSm90PullFp8CutedslMegaMoeConfig)
    assert sm90.megakernel.intermediate_size == 2048
    assert sm90.megakernel.kind == "fp8_e4m3"
    assert sm90.megakernel.fp8_scale_mode == "blockwise"
    assert sm90.megakernel.gate_up_clamp == 10.0
    assert sm90.preprocess_weights and sm90.quantize_input

    with pytest.raises(ValueError, match="Unsupported fi_moe_ep megakernel"):
        build_fi_mega_config(
            intermediate_size=2048,
            top_k=8,
            activation_clamp=None,
            megakernel="deep_gemm",
        )


def test_ckpt_uses_nvfp4_experts_reads_moe_quant_algo():
    from vllm.models.deepseek_v4.nvidia.fi_moe import ckpt_uses_nvfp4_experts

    nvfp4 = SimpleNamespace(quant_config=SimpleNamespace(moe_quant_algo="NVFP4"))
    assert ckpt_uses_nvfp4_experts(nvfp4)

    mxfp4 = SimpleNamespace(quant_config=SimpleNamespace(moe_quant_algo=None))
    assert not ckpt_uses_nvfp4_experts(mxfp4)

    no_algo = SimpleNamespace(quant_config=SimpleNamespace())
    assert not ckpt_uses_nvfp4_experts(no_algo)


def test_dequant_fp4_ue8m0_gran32_decodes_lut_and_scales():
    """One 32-element scale group per row: low nibble is the even element,
    high nibble the odd one, ue8m0 scale applies to the whole group."""
    packed = torch.arange(32, dtype=torch.uint8).reshape(2, 16)
    sf = torch.tensor([[127], [128]], dtype=torch.uint8)  # 2**0, 2**1

    out = _dequant_fp4_ue8m0_gran32(packed, sf)

    assert out.shape == (2, 32)
    assert out.dtype == torch.bfloat16
    expected = torch.empty(2, 32)
    for row in range(2):
        for col in range(16):
            byte = int(packed[row, col])
            expected[row, 2 * col] = _E2M1_LUT[byte & 0x0F]
            expected[row, 2 * col + 1] = _E2M1_LUT[byte >> 4]
        expected[row] *= 2.0**row
    assert torch.equal(out, expected.to(torch.bfloat16))


def test_dequant_nvfp4_expert_weights_to_bf16_applies_block_and_global_scales():
    """NVFP4: e2m1 nibbles * per-16 e4m3 block scale * per-tensor scale_2.

    w13 carries two scale_2 columns (gate rows / up rows); w2 carries one.
    """
    from vllm.utils.flashinfer_moe_ep import (
        _dequant_nvfp4_expert_weights_to_bf16,
    )

    # E=1, N=4, K=32 -> K//2=16 packed bytes, K//16=2 block scales.
    packed = torch.arange(4 * 16, dtype=torch.uint8).reshape(1, 4, 16)
    block_scale = torch.ones(1, 4, 2, dtype=torch.float32).to(torch.float8_e4m3fn)
    block_scale[0, :, 1] = 2.0

    # w2-style: one scalar per expert.
    out = _dequant_nvfp4_expert_weights_to_bf16(
        packed, block_scale, torch.tensor([3.0])
    )
    expected = torch.empty(1, 4, 32)
    for row in range(4):
        for col in range(16):
            byte = int(packed[0, row, col])
            group = 1.0 if col < 8 else 2.0
            expected[0, row, 2 * col] = _E2M1_LUT[byte & 0x0F] * group * 3.0
            expected[0, row, 2 * col + 1] = _E2M1_LUT[byte >> 4] * group * 3.0
    assert torch.equal(out, expected.to(torch.bfloat16))

    # w13-style: gate rows use column 0, up rows use column 1.
    out13 = _dequant_nvfp4_expert_weights_to_bf16(
        packed,
        torch.ones(1, 4, 2, dtype=torch.float32).to(torch.float8_e4m3fn),
        torch.tensor([[5.0, 7.0]]),
        gate_rows=2,
    )
    expected13 = torch.empty(1, 4, 32)
    for row in range(4):
        s2 = 5.0 if row < 2 else 7.0
        for col in range(16):
            byte = int(packed[0, row, col])
            expected13[0, row, 2 * col] = _E2M1_LUT[byte & 0x0F] * s2
            expected13[0, row, 2 * col + 1] = _E2M1_LUT[byte >> 4] * s2
    assert torch.equal(out13, expected13.to(torch.bfloat16))
