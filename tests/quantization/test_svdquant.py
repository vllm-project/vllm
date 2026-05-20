# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke tests for the SVDQuant quantization plugin.

Real W4A4 numerics live on top of an actual quantized checkpoint and
require a CUDA capability that the kernel backend supports. These
tests cover the boundary that vLLM owns: the registry wiring, the
config / linear method shape, and the hardware-keyed backend
selection.
"""

import pytest
import torch

from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import (
    QUANTIZATION_METHODS,
    get_quantization_config,
)
from vllm.model_executor.layers.quantization.svdquant import (
    SVDQuantConfig,
    SVDQuantLinearMethod,
)
from vllm.model_executor.layers.quantization.utils.svdquant_dispatch import (
    assert_svdquant_supported,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.nunchaku import has_nunchaku_w4a4


def test_svdquant_is_registered() -> None:
    assert "svdquant" in QUANTIZATION_METHODS
    cls = get_quantization_config("svdquant")
    assert cls is SVDQuantConfig
    assert cls.get_name() == "svdquant"


def test_config_from_dict_int4() -> None:
    cfg = SVDQuantConfig.from_config(
        {"rank": 32, "precision": "int4", "act_unsigned": False}
    )
    assert cfg.rank == 32
    assert cfg.precision == "int4"
    assert cfg.group_size == 64
    assert cfg.act_unsigned is False
    assert cfg.modules_to_not_convert == []


def test_config_from_dict_nvfp4() -> None:
    cfg = SVDQuantConfig.from_config(
        {
            "rank": 64,
            "precision": "nvfp4",
            "modules_to_not_convert": ["embedder", "final_layer"],
        }
    )
    assert cfg.precision == "nvfp4"
    assert cfg.group_size == 16  # NVFP4 tcgen05 scale block
    assert cfg.modules_to_not_convert == ["embedder", "final_layer"]


def test_config_rejects_unknown_precision() -> None:
    with pytest.raises(ValueError, match="precision"):
        SVDQuantConfig(precision="fp8")  # type: ignore[arg-type]


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="hardware gate is CUDA-specific"
)
def test_hardware_gate_accepts_consumer_gpus() -> None:
    if not has_nunchaku_w4a4():
        pytest.skip("nunchaku not installed")
    major, _ = current_platform.get_device_capability()
    if major == 9:
        pytest.skip("Hopper is intentionally unsupported")
    if major == 10:
        pytest.skip("Datacenter Blackwell is out of scope (FlashInfer planned)")
    # Turing/Ampere/Ada (SM_75-89) and consumer Blackwell SM_120 are
    # accepted by the gate for int4.
    assert_svdquant_supported("int4")


def test_hardware_gate_rejects_hopper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hopper SM_90 must raise."""
    # Patch the class (not the instance): classmethods in Platform call
    # cls.get_device_capability(), bypassing instance attribute lookup.
    cls = type(current_platform)
    monkeypatch.setattr(cls, "is_cuda", classmethod(lambda c: True))
    monkeypatch.setattr(
        cls,
        "get_device_capability",
        classmethod(lambda c, *a, **k: DeviceCapability(9, 0)),
    )
    with pytest.raises(RuntimeError, match="Hopper"):
        assert_svdquant_supported("int4")


def test_hardware_gate_rejects_datacenter_blackwell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SM_100/103 is out of scope here (FlashInfer-planned); must raise."""
    cls = type(current_platform)
    monkeypatch.setattr(cls, "is_cuda", classmethod(lambda c: True))
    monkeypatch.setattr(
        cls,
        "get_device_capability",
        classmethod(lambda c, *a, **k: DeviceCapability(10, 0)),
    )
    with pytest.raises(RuntimeError, match="FlashInfer"):
        assert_svdquant_supported("nvfp4")


def test_hardware_gate_rejects_nvfp4_on_pre_blackwell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NVFP4 needs SM_100+ tensor units; SM_8x must raise cleanly."""
    if not has_nunchaku_w4a4():
        pytest.skip("nunchaku not installed")
    cls = type(current_platform)
    monkeypatch.setattr(cls, "is_cuda", classmethod(lambda c: True))
    monkeypatch.setattr(
        cls,
        "get_device_capability",
        classmethod(lambda c, *a, **k: DeviceCapability(8, 9)),
    )
    with pytest.raises(ValueError, match="NVFP4"):
        assert_svdquant_supported("nvfp4")


@pytest.mark.skipif(
    not (current_platform.is_cuda() and has_nunchaku_w4a4()),
    reason="requires CUDA + nunchaku for create_weights smoke",
)
def test_linear_method_create_weights_int4() -> None:
    """Validate the parameter layout without invoking the kernel.

    Only checks that `create_weights` populates the layer with
    correctly-shaped, correctly-dtyped tensors.
    """
    cfg = SVDQuantConfig(rank=32, precision="int4")
    method = SVDQuantLinearMethod(cfg)

    # Mimic a 4096-in / 4096-out column-parallel layer with TP=1.
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=4096,
        output_partition_sizes=[4096],
        input_size=4096,
        output_size=4096,
        params_dtype=torch.bfloat16,
    )

    assert layer.qweight.shape == (4096, 4096 // 2)
    assert layer.qweight.dtype == torch.int8
    assert layer.wscales.shape == (4096 // 64, 4096)
    assert layer.wscales.dtype == torch.bfloat16
    assert layer.proj_down.shape == (4096, 32)
    assert layer.proj_up.shape == (4096, 32)
    assert layer.smooth_factor.shape == (4096,)
    assert layer.wcscales is None
    assert layer.wtscale is None


@pytest.mark.skipif(
    not (current_platform.is_cuda() and has_nunchaku_w4a4()),
    reason="requires CUDA + nunchaku for create_weights smoke",
)
def test_linear_method_create_weights_nvfp4_has_per_channel_scales() -> None:
    cfg = SVDQuantConfig(rank=32, precision="nvfp4")
    try:
        assert_svdquant_supported("nvfp4")
    except (RuntimeError, ValueError, ImportError) as exc:
        pytest.skip(f"nvfp4 unsupported on this box: {exc}")
    method = SVDQuantLinearMethod(cfg)
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=2048,
        output_partition_sizes=[2048],
        input_size=2048,
        output_size=2048,
        params_dtype=torch.bfloat16,
    )
    assert layer.wscales.dtype == torch.float8_e4m3fn
    assert layer.wcscales is not None
    assert layer.wcscales.shape == (2048,)
    assert layer.wtscale is not None
    assert layer.wtscale.shape == (1,)


def test_get_quant_method_skips_listed_modules() -> None:
    cfg = SVDQuantConfig(modules_to_not_convert=["embedder"])
    if not has_nunchaku_w4a4():
        # SVDQuantLinearMethod ctor would call assert_svdquant_supported()
        # and raise; in that case we can only check the skip path.
        pytest.skip("nunchaku not installed")
    fake_layer = torch.nn.Linear(8, 8)
    # Subclass to satisfy isinstance(layer, LinearBase).
    fake_layer.__class__ = type(
        "FakeLinear", (torch.nn.Linear, LinearBase), {}
    )

    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    method = cfg.get_quant_method(fake_layer, "model.embedder.proj")
    assert isinstance(method, UnquantizedLinearMethod)
