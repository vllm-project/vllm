# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
from collections.abc import Iterator
from pathlib import Path

import pytest

import torch

import vllm.utils.flashinfer as fi


def _make_exe(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(mode=0o755)


@pytest.fixture
def default_cuda_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Hide any real toolkit, keep ninja on PATH, redirect /usr/local/cuda."""
    for var in ("CUDA_HOME", "CUDA_PATH", "FLASHINFER_NVCC"):
        monkeypatch.delenv(var, raising=False)
    _make_exe(tmp_path / "venv" / "bin" / "ninja")
    monkeypatch.setenv("PATH", str(tmp_path / "venv" / "bin"))
    home = tmp_path / "usr_local_cuda"
    monkeypatch.setattr(fi, "_DEFAULT_CUDA_HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def flashinfer_without_cubin(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pretend flashinfer-python is installed without flashinfer-cubin."""
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args: (
            object() if name == "flashinfer" else real_find_spec(name, *args)
        ),
    )
    monkeypatch.setattr(fi, "has_flashinfer_cubin", lambda: False)
    fi.has_flashinfer.cache_clear()
    yield
    fi.has_flashinfer.cache_clear()


def test_has_flashinfer_finds_toolkit_off_path(default_cuda_home: Path):
    """Bare-metal installs keep nvcc in /usr/local/cuda, off PATH, which is
    where FlashInfer's JIT finds it."""
    _make_exe(default_cuda_home / "bin" / "nvcc")
    assert fi.has_flashinfer()


def test_has_flashinfer_requires_ninja(
    default_cuda_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """FlashInfer runs `ninja` from PATH, which lacks the venv's bin directory
    when vLLM is launched without activating the venv."""
    _make_exe(tmp_path / "cuda" / "bin" / "nvcc")
    monkeypatch.setenv("PATH", str(tmp_path / "cuda" / "bin"))
    assert not fi.has_flashinfer()


def test_swap_w13_to_w31():
    x = torch.tensor([
        [[1, 2], [3, 4], [5, 6], [7, 8]],
    ])
    swapped = fi.swap_w13_to_w31(x)
    expected = torch.tensor([
        [[5, 6], [7, 8], [1, 2], [3, 4]],
    ])
    assert torch.equal(swapped, expected)


def test_align_moe_weights_for_fi():
    w13 = torch.randn(2, 34, 16)
    w2 = torch.randn(2, 16, 17)
    padded_w13, padded_w2, padded_intermediate = fi.align_moe_weights_for_fi(
        w13, w2, is_act_and_mul=True, min_alignment=16
    )
    assert padded_intermediate == 32
    assert padded_w13.shape == (2, 64, 16)
    assert padded_w2.shape == (2, 16, 32)


def test_align_fp4_moe_weights_for_fi():
    num_experts, hidden_size, intermediate = 2, 32, 18
    w13 = torch.zeros(
        (num_experts, intermediate * 2, hidden_size // 2), dtype=torch.uint8
    )
    w13_scale = torch.zeros(
        (num_experts, intermediate * 2, hidden_size // 16), dtype=torch.uint8
    )
    w2 = torch.zeros((num_experts, hidden_size, intermediate // 2), dtype=torch.uint8)
    w2_scale = torch.zeros(
        (num_experts, hidden_size, intermediate // 16), dtype=torch.uint8
    )

    pw13, pw13_s, pw2, pw2_s, p_inter = fi.align_fp4_moe_weights_for_fi(
        w13, w13_scale, w2, w2_scale, is_act_and_mul=True, min_alignment=16
    )
    assert p_inter == 32
    assert pw13.shape == (num_experts, 64, hidden_size // 2)
    assert pw2.shape == (num_experts, hidden_size, 16)


def test_align_fp4_moe_hidden_dim_for_fi():
    num_experts, gate_up_dim, packed_hidden = 2, 32, 100
    w13 = torch.zeros((num_experts, gate_up_dim, packed_hidden), dtype=torch.uint8)
    w13_scale = torch.zeros(
        (num_experts, gate_up_dim, packed_hidden // 8), dtype=torch.uint8
    )
    w2 = torch.zeros((num_experts, packed_hidden * 2, 16), dtype=torch.uint8)
    w2_scale = torch.zeros((num_experts, packed_hidden * 2, 2), dtype=torch.uint8)

    pw13, pw13_s, pw2, pw2_s, p_hidden = fi.align_fp4_moe_hidden_dim_for_fi(
        w13, w13_scale, w2, w2_scale, min_alignment=256
    )
    assert p_hidden == 256
    assert pw13.shape == (num_experts, gate_up_dim, 128)
    assert pw2.shape == (num_experts, 256, 16)


def test_has_flashinfer_situ_activation_without_flashinfer():
    assert not fi.has_flashinfer_situ_activation()


def test_activation_to_flashinfer_type_mappings(monkeypatch):
    import sys
    from enum import Enum
    from types import ModuleType

    class MockActivationType(Enum):
        Silu = 0
        Gelu = 1
        Swiglu = 2
        Geglu = 3
        Relu2 = 4
        Situ = 5

    mock_mod = ModuleType("flashinfer.fused_moe.core")
    mock_mod.ActivationType = MockActivationType
    monkeypatch.setitem(sys.modules, "flashinfer.fused_moe.core", mock_mod)

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    assert fi.has_flashinfer_situ_activation()
    assert (
        fi.activation_to_flashinfer_type(MoEActivation.SILU_NO_MUL)
        == MockActivationType.Silu
    )
    assert (
        fi.activation_to_flashinfer_type(MoEActivation.GELU_NO_MUL)
        == MockActivationType.Gelu
    )
    assert (
        fi.activation_to_flashinfer_type(MoEActivation.SILU)
        == MockActivationType.Swiglu
    )
    assert (
        fi.activation_to_flashinfer_type(MoEActivation.SWIGLUOAI)
        == MockActivationType.Swiglu
    )
    assert (
        fi.activation_to_flashinfer_type(MoEActivation.SWIGLUOAI_UNINTERLEAVE)
        == MockActivationType.Swiglu
    )
    assert (
        fi.activation_to_flashinfer_type(MoEActivation.GELU)
        == MockActivationType.Geglu
    )
    assert (
        fi.activation_to_flashinfer_type(MoEActivation.GELU_TANH)
        == MockActivationType.Geglu
    )
    assert (
        fi.activation_to_flashinfer_type(MoEActivation.RELU2_NO_MUL)
        == MockActivationType.Relu2
    )
    assert (
        fi.activation_to_flashinfer_type(MoEActivation.SITU)
        == MockActivationType.Situ
    )
    assert (
        fi.activation_to_flashinfer_int(MoEActivation.SILU)
        == MockActivationType.Swiglu.value
    )
