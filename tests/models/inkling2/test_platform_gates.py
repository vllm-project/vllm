# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform scope of the Inkling2 dispatch predicates.

On CUDA the predicates keep the decisions the CUDA kernels made before the
portable paths. The platform and its device capability are
emulated on the current device's platform object.
"""

import pytest

from vllm.models.inkling2.common.ops import platform as predicates
from vllm.platforms import current_platform

_PREDICATES = (
    predicates.use_sm100_kernels,
    predicates.use_pdl,
    predicates.pdl_launch_kwargs,
    predicates.use_native_conv_mix,
    predicates.use_native_e4m3,
)


@pytest.fixture
def emulate(monkeypatch):
    """`emulate(platform, capability)`: `current_platform` reports `platform`
    ("cuda", "rocm" or "xpu") and device capability `capability` (e.g. 90
    for SM90, 94 for gfx942; XPU reports none). The cached predicates are
    recomputed."""

    def clear():
        for predicate in _PREDICATES:
            predicate.cache_clear()

    def apply(platform: str, capability: int | None = None) -> None:
        assert platform != "xpu" or capability is None
        monkeypatch.setattr(current_platform, "is_cuda", lambda: platform == "cuda")
        monkeypatch.setattr(current_platform, "is_rocm", lambda: platform == "rocm")
        monkeypatch.setattr(current_platform, "is_xpu", lambda: platform == "xpu")
        monkeypatch.setattr(
            current_platform,
            "is_device_capability_family",
            lambda family, device_id=0: (
                capability is not None and capability // 10 == family // 10
            ),
        )
        monkeypatch.setattr(
            current_platform,
            "has_device_capability",
            lambda minimum, device_id=0: (
                capability is not None and capability >= minimum
            ),
        )
        clear()

    clear()
    yield apply
    monkeypatch.undo()
    clear()


@pytest.mark.parametrize(
    "platform,capability,kwargs,pdl",
    [
        # The pin launched every PDL kernel with launch_pdl=True.
        ("cuda", 90, {"launch_pdl": True}, True),
        ("cuda", 100, {"launch_pdl": True}, True),
        ("cuda", 120, {"launch_pdl": True}, True),
        # vllm-project/vllm#69165: launch_pdl=use_pdl() below SM90.
        ("cuda", 80, {"launch_pdl": False}, False),
        ("cuda", 89, {"launch_pdl": False}, False),
        # The AMD (HIP) and Intel Triton backends have no `launch_pdl`
        # option and reject it, even when it is False. use_pdl() stays True
        # on gfx94x / gfx95x (ROCm is not a portable path here: its gdc_*
        # calls, TMA K-group kernel and native conv_mix op would still be
        # selected; ROCm rejects the inkling2 quantization today).
        ("rocm", 94, {}, True),
        ("rocm", 95, {}, True),
        ("xpu", None, {}, False),
    ],
)
def test_pdl_launch_kwargs_per_platform(emulate, platform, capability, kwargs, pdl):
    """`pdl_launch_kwargs()` is `launch_pdl=use_pdl()` on CUDA and no option
    elsewhere; `use_pdl` (and so the griddepcontrol calls of `pdl.py`)
    follows SM90+ (#69165)."""
    emulate(platform, capability)
    assert dict(predicates.pdl_launch_kwargs()) == kwargs
    assert predicates.use_pdl() == pdl


def test_no_literal_launch_pdl_outside_pdl_launch_kwargs():
    """No launch site of the shared code (`vllm/models/inkling2/` outside the
    CUDA-only `nvidia/` package) passes a literal `launch_pdl=` keyword:
    every PDL launch unpacks `**pdl_launch_kwargs()`, the only place that
    sets the option, so the shared kernels launch on Triton backends
    without it (XPU, ROCm)."""
    import ast
    import pathlib

    import vllm.models.inkling2 as package

    root = pathlib.Path(package.__file__).parent
    found = []
    for path in sorted(root.rglob("*.py")):
        if path.relative_to(root).parts[0] == "nvidia":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call):
                found += [
                    f"{path.relative_to(root)}:{node.lineno}"
                    for keyword in node.keywords
                    if keyword.arg == "launch_pdl"
                ]
    assert not found, found


def test_xpu_predicates_take_portable_paths(emulate):
    """XPU: `short_conv` + `attn_res`, stream-ordered launches (no
    `launch_pdl` option at all), software e4m3, no SM10x kernels."""
    emulate("xpu")
    assert not predicates.use_native_conv_mix()
    assert not predicates.use_pdl()
    assert dict(predicates.pdl_launch_kwargs()) == {}
    assert not predicates.use_native_e4m3()
    assert not predicates.use_sm100_kernels()


@pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="needs a GPU",
)
@pytest.mark.parametrize("rows", [1, 8, 32])
def test_linear_small_batches_without_flashinfer(monkeypatch, rows):
    """Off SM10x, `linear` of up to 32 rows never queries FlashInfer (a
    CUDA-only dependency, made unimportable here) and is `F.linear`, or on
    SM12x the `sm12x_bf16_ll_gemm` GEMV where it applies (up to 16 rows).
    Before the operand reorder, the FlashInfer autotuning query ran first and
    raised `ModuleNotFoundError`."""
    import sys

    import torch

    from vllm.models.inkling2.common.ops import linear as linear_ops
    from vllm.models.inkling2.nvidia.ops.ll_gemm_sm120 import (
        sm12x_bf16_ll_gemm,
        use_sm12x_bf16_ll_gemm,
    )

    if predicates.use_sm100_kernels():
        pytest.skip("SM10x runs FlashInfer's mm_bf16 for up to 32 rows")
    monkeypatch.setitem(sys.modules, "flashinfer", None)
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", None)
    device = current_platform.device_type
    torch.manual_seed(0)
    x = torch.randn(rows, 512, device=device, dtype=torch.bfloat16)
    weight = torch.randn(256, 512, device=device, dtype=torch.bfloat16)
    actual = linear_ops.linear(x, weight)
    if use_sm12x_bf16_ll_gemm(x, weight):
        expected = sm12x_bf16_ll_gemm(x, weight)
    else:
        expected = torch.nn.functional.linear(x, weight)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _vllm_config(tmp_path, lora: bool, cache_dtype: str = "auto"):
    import json

    from tests.models.inkling2.utils import TINY_CONFIG
    from vllm.config import CacheConfig, LoRAConfig, ModelConfig, VllmConfig

    (tmp_path / "config.json").write_text(json.dumps(TINY_CONFIG))
    return VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            dtype="bfloat16",
            skip_tokenizer_init=True,
            tokenizer_mode="hf",
            max_model_len=2048,
            limit_mm_per_prompt={"image": 0},
        ),
        cache_config=CacheConfig(cache_dtype=cache_dtype),
        lora_config=LoRAConfig(max_lora_rank=8) if lora else None,
    )


@pytest.mark.parametrize(
    "lora,vision,expected",
    [
        (False, False, []),
        (True, False, ["LoRA"]),
        (False, True, ["image inputs"]),
        (True, True, ["image inputs", "LoRA"]),
    ],
    ids=["text", "lora", "image", "image-and-lora"],
)
def test_xpu_unsupported_features(tmp_path, lora, vision, expected):
    """The features XPU rejects at model init, for explicit requests: image
    inputs and LoRA (DSpark: `test_model.py`); a text-only model has
    none."""
    from vllm.models.inkling2.common.model import xpu_unsupported_features

    features = xpu_unsupported_features(_vllm_config(tmp_path, lora), vision)
    assert [f.split(" (")[0] for f in features] == expected


@pytest.mark.parametrize("lora", [False, True])
def test_model_init_gate_fires_on_xpu_only(tmp_path, lora):
    """On XPU the model-init gate raises for LoRA and passes a text-only
    model; on other platforms it passes LoRA too."""
    from vllm.models.inkling2.common.model import (
        _check_platform_support,
        xpu_unsupported_features,
    )

    config = _vllm_config(tmp_path, lora)
    assert xpu_unsupported_features(config, False) == (["LoRA"] if lora else [])
    if current_platform.is_xpu() and lora:
        with pytest.raises(NotImplementedError, match=r"does not support LoRA$"):
            _check_platform_support(config, vision=False)
    else:
        _check_platform_support(config, vision=False)


def test_flat_experts_per_platform():
    """The flat-row expert backend: the XPU backends on XPU, Humming
    elsewhere."""
    import functools

    from vllm.models.inkling2.common.moe import flat_experts_class

    for quant_format in ("nvfp4", "mxfp8"):
        backend = flat_experts_class(quant_format, 8, 4.0, 25.0)
        if current_platform.is_xpu():
            from vllm.models.inkling2.xpu.experts import (
                XpuMxfp8Experts,
                XpuNvfp4Experts,
            )

            expected = XpuNvfp4Experts if quant_format == "nvfp4" else XpuMxfp8Experts
            assert isinstance(backend, functools.partial)
            assert backend.func is expected
        else:
            from vllm.models.inkling2.nvidia.ops.moe_sm90 import HummingExperts

            assert backend is HummingExperts


@pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="builds the model on the device",
)
def test_fp8_kv_cache_follows_the_attention_layer_support(tmp_path, dist_init):
    """An FP8 KV cache builds where the attention layers list it
    (`supported_kv_cache_dtypes`) and is rejected at model init elsewhere
    (e.g. on XPU)."""
    import torch

    from vllm.config import set_current_vllm_config
    from vllm.models.inkling2.common.attention import supported_kv_cache_dtypes
    from vllm.models.inkling2.common.model import Inkling2ForConditionalGeneration

    config = _vllm_config(tmp_path, lora=False, cache_dtype="fp8_e4m3")
    with set_current_vllm_config(config), torch.device(current_platform.device_type):
        if "fp8_e4m3" in supported_kv_cache_dtypes():
            Inkling2ForConditionalGeneration(vllm_config=config)
        else:
            with pytest.raises(ValueError, match="KV cache dtype must be one of"):
                Inkling2ForConditionalGeneration(vllm_config=config)
    if current_platform.is_xpu():
        assert "fp8_e4m3" not in supported_kv_cache_dtypes()
