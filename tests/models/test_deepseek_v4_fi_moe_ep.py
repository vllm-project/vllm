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

from vllm.config.kernel import MEGA_MOE_BACKENDS
from vllm.utils.flashinfer_moe_ep import (
    _E2M1_LUT,
    FI_MOE_EP_BACKENDS,
    _dequant_fp4_ue8m0_gran32,
    build_fi_mega_config,
    fi_moe_ep_backend_spec,
    is_fi_moe_ep_backend,
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
class _FakeMegaConfig:
    megakernel: Any
    preprocess_weights: bool
    quantize_input: bool


@pytest.fixture
def fake_flashinfer(monkeypatch):
    """Install a minimal fake flashinfer.moe_ep for the lazy imports."""
    moe_ep = ModuleType("flashinfer.moe_ep")
    moe_ep.BootstrapConfig = _FakeBootstrapConfig
    moe_ep.DeepGemmMegaMoeConfig = _FakeDeepGemmMegaMoeConfig
    moe_ep.Nvfp4CutedslMegaMoeConfig = _FakeNvfp4CutedslMegaMoeConfig
    moe_ep.MegaConfig = _FakeMegaConfig

    core = ModuleType("flashinfer.moe_ep.core")
    runtime = ModuleType("flashinfer.moe_ep.core.runtime")
    runtime.TORCH_DIST = "torch_dist"
    runtime.NVSHMEM = "nvshmem"

    flashinfer = ModuleType("flashinfer")
    flashinfer.moe_ep = moe_ep
    moe_ep.core = core
    core.runtime = runtime

    for name, mod in {
        "flashinfer": flashinfer,
        "flashinfer.moe_ep": moe_ep,
        "flashinfer.moe_ep.core": core,
        "flashinfer.moe_ep.core.runtime": runtime,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return moe_ep


def test_fi_backend_strings_are_registered_mega_moe_backends():
    assert set(FI_MOE_EP_BACKENDS) <= MEGA_MOE_BACKENDS


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


def test_backend_selection_rejects_flashinfer_without_device_bootstrap(
    fake_flashinfer, monkeypatch
):
    """Pre-0.6.18 flashinfer lacks BootstrapConfig.device and would rebind
    workers to the wrong GPU under a remapped CUDA_VISIBLE_DEVICES; selecting
    a fi backend against such a build must fail at config time."""
    import vllm.utils.flashinfer_moe_ep as mod

    @dataclass
    class _LegacyBootstrapConfig:
        world_size: int
        rank: int
        process_group: Any = None
        auto_bootstrap: bool = True

    fake_flashinfer.bootstrap_moe_ep_runtime = lambda *a, **k: None
    fake_flashinfer.ensure_moe_ep_cuda_device = lambda *a, **k: None
    fake_flashinfer.finalize_moe_ep_runtime = lambda *a, **k: None

    monkeypatch.setattr(mod, "_FI_MOE_EP_RUNTIME_AVAILABLE", None)
    fake_flashinfer.BootstrapConfig = _LegacyBootstrapConfig
    with pytest.raises(ImportError, match="0.6.18"):
        is_fi_moe_ep_backend("flashinfer_moe_ep_mega_cutedsl")

    monkeypatch.setattr(mod, "_FI_MOE_EP_RUNTIME_AVAILABLE", None)
    fake_flashinfer.BootstrapConfig = _FakeBootstrapConfig
    assert is_fi_moe_ep_backend("flashinfer_moe_ep_mega_cutedsl")


def test_fi_moe_ep_backend_spec_kernel_and_nvshmem_contract():
    dg = fi_moe_ep_backend_spec("flashinfer_moe_ep_mega_deep_gemm")
    assert dg.megakernel == "deep_gemm_mega"
    assert not dg.needs_nvshmem

    cd = fi_moe_ep_backend_spec("flashinfer_moe_ep_mega_cutedsl")
    assert cd.megakernel == "nvfp4_cutedsl"
    assert cd.needs_nvshmem

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
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)

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
