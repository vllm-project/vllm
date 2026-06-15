# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Quark quantization config handling, model loading, and accuracy.

This is the canonical home for Quark tests that are not ROCm-only. Platform-
specific MXFP4 utility coverage lives in
``tests/kernels/quantization/rocm/test_quark.py``.
"""

import importlib.metadata
from dataclasses import dataclass
from functools import lru_cache
from importlib.util import find_spec
from unittest.mock import MagicMock

import huggingface_hub
import lm_eval
import pytest
import torch
from packaging import version

from tests.utils import multi_gpu_marks, multi_gpu_only
from vllm.model_executor.layers.quantization.quark.quark import (
    QuarkConfig,
    QuarkKVCacheMethod,
    QuarkLinearMethod,
    QuarkW8A8Fp8,
    QuarkW8A8Int8,
)
from vllm.model_executor.layers.quantization.quark.quark_moe import (
    QuarkW8A8Int8MoEMethod,
)
from vllm.model_executor.layers.quantization.quark.schemes import QuarkOCP_MX
from vllm.platforms import current_platform

from .reference_mxfp4 import dq_mxfp4_torch, qdq_mxfp4_torch

QUARK_MXFP4_MIN_VERSION = "0.8.99"
QUARK_MXFP4_AVAILABLE = find_spec("quark") is not None and version.parse(
    importlib.metadata.version("amd-quark")
) >= version.parse(QUARK_MXFP4_MIN_VERSION)

DEVICE_TYPE = current_platform.device_type

_PARIS_PROMPTS = ["Answer with one word only. The capital of France is"]
_SMOKE_PROMPTS = ["Tell me a short fact."]

_FP8_PER_TENSOR_WEIGHT = {
    "dtype": "fp8_e4m3",
    "qscheme": "per_tensor",
    "is_dynamic": False,
}
_FP8_PER_CHANNEL_WEIGHT = {
    "dtype": "fp8_e4m3",
    "qscheme": "per_channel",
    "is_dynamic": False,
}
_FP8_DYNAMIC_PER_TENSOR_INPUT = {
    "dtype": "fp8_e4m3",
    "qscheme": "per_tensor",
    "is_dynamic": True,
}
_FP8_DYNAMIC_PER_TOKEN_INPUT = {
    "dtype": "fp8_e4m3",
    "qscheme": "per_token",
    "is_dynamic": True,
}
_INT8_STATIC_PER_TENSOR = {
    "dtype": "int8",
    "qscheme": "per_tensor",
    "is_dynamic": False,
    "symmetric": True,
}
_INT8_STATIC_PER_CHANNEL_WEIGHT = {
    "dtype": "int8",
    "qscheme": "per_channel",
    "is_dynamic": False,
    "symmetric": True,
}
_MXFP4_PER_GROUP_WEIGHT = {
    "dtype": "fp4",
    "qscheme": "per_group",
    "group_size": 32,
    "scale_format": "e8m0",
    "is_dynamic": False,
}
_MXFP4_DYNAMIC_INPUT = {
    "dtype": "fp4",
    "qscheme": "per_group",
    "group_size": 32,
    "scale_format": "e8m0",
    "is_dynamic": True,
}


@pytest.fixture(scope="function", autouse=True)
def enable_pickle(monkeypatch):
    """`LLM.apply_model` requires pickling a function."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def _make_quark_config(
    *,
    quant_config: dict | None = None,
    kv_cache_group: list[str] | None = None,
    kv_cache_config: dict | None = None,
    pack_method: str = "reorder",
) -> QuarkConfig:
    return QuarkConfig(
        quant_config={} if quant_config is None else quant_config,
        kv_cache_group=[] if kv_cache_group is None else kv_cache_group,
        kv_cache_config=kv_cache_config,
        pack_method=pack_method,
    )


def _assert_generation_succeeds(
    outputs: list[tuple[list[int], str]],
    *,
    required_word: str | None = None,
) -> None:
    assert len(outputs) == 1
    token_ids, text = outputs[0]
    print(f"[quark] generated text: {text!r}")
    assert token_ids, "expected at least one generated token"
    assert text.strip(), "expected non-empty generated text"
    if required_word is not None:
        assert required_word in text.lower(), (
            f"expected generated text to contain {required_word!r}, got {text!r}"
        )


def _get_first_qkv_proj(model):
    return model.model.layers[0].self_attn.qkv_proj


def _assert_metric_close(
    *,
    actual: float,
    expected: float,
    tolerance: float,
    label: str,
) -> None:
    print(
        f"[quark] {label}: expected={expected:.4f} "
        f"measured={actual:.4f} tolerance={tolerance:.4f}"
    )
    assert abs(actual - expected) <= tolerance, (
        f"{label} drifted beyond tolerance: "
        f"expected {expected:.4f}, measured {actual:.4f}, tolerance {tolerance:.4f}"
    )


@lru_cache(maxsize=1)
def _has_hf_amd_org_access() -> bool:
    try:
        huggingface_hub.list_repo_refs(
            "amd/Llama-3.3-70B-Instruct-WMXFP4-AMXFP4-KVFP8-Scale-UINT8-SQ"
        )
        return True
    except Exception:
        return False


def _import_quark_mxfp4_modules():
    from quark.torch.export.nn.modules.realquantizer import StaticScaledRealQuantizer
    from quark.torch.kernel import mx as mx_kernel
    from quark.torch.quantization.config.config import FP4PerGroupSpec

    return StaticScaledRealQuantizer, mx_kernel, FP4PerGroupSpec


def test_quark_config_stores_constructor_args():
    quant_config = {"quant_type": "a8w8_fp8_dynamic"}
    kv_cache_group = ["attn.k_proj", "attn.v_proj"]

    cfg = _make_quark_config(
        quant_config=quant_config,
        kv_cache_group=kv_cache_group,
        pack_method="order",
    )

    assert cfg.quant_config == quant_config
    assert cfg.kv_cache_group == kv_cache_group
    assert cfg.pack_method == "order"
    assert _make_quark_config().pack_method == "reorder"


@pytest.mark.parametrize(
    ("weight_config", "input_config", "expected"),
    [
        (_FP8_PER_TENSOR_WEIGHT, _FP8_DYNAMIC_PER_TENSOR_INPUT, True),
        (_FP8_PER_CHANNEL_WEIGHT, _FP8_DYNAMIC_PER_TOKEN_INPUT, True),
        (_INT8_STATIC_PER_TENSOR, _INT8_STATIC_PER_TENSOR, False),
        (_FP8_PER_TENSOR_WEIGHT, None, False),
        (None, _FP8_DYNAMIC_PER_TENSOR_INPUT, False),
    ],
)
def test_quark_is_fp8_w8a8(weight_config, input_config, expected):
    cfg = _make_quark_config()
    assert cfg._is_fp8_w8a8(weight_config, input_config) is expected


@pytest.mark.parametrize(
    ("weight_config", "input_config", "expected"),
    [
        (_INT8_STATIC_PER_TENSOR, _INT8_STATIC_PER_TENSOR, True),
        (_INT8_STATIC_PER_CHANNEL_WEIGHT, _INT8_STATIC_PER_TENSOR, True),
        (
            {**_INT8_STATIC_PER_TENSOR, "symmetric": False},
            _INT8_STATIC_PER_TENSOR,
            False,
        ),
        (
            {**_INT8_STATIC_PER_TENSOR, "is_dynamic": True},
            _INT8_STATIC_PER_TENSOR,
            False,
        ),
    ],
)
def test_quark_is_static_tensor_w8a8(weight_config, input_config, expected):
    cfg = _make_quark_config()
    assert cfg._is_static_tensor_w8a8(weight_config, input_config) is expected


@pytest.mark.parametrize(
    ("weight_config", "expected"),
    [
        (_MXFP4_PER_GROUP_WEIGHT, True),
        (
            {
                **_MXFP4_PER_GROUP_WEIGHT,
                "dtype": "fp6_e3m2",
            },
            True,
        ),
        (
            {
                **_MXFP4_PER_GROUP_WEIGHT,
                "group_size": 64,
            },
            False,
        ),
        (
            {
                **_MXFP4_PER_GROUP_WEIGHT,
                "scale_format": "e5m2",
            },
            False,
        ),
        (
            {
                **_MXFP4_PER_GROUP_WEIGHT,
                "qscheme": "per_tensor",
            },
            False,
        ),
        ([{"dtype": "fp8_e4m3"}, {"dtype": "int4"}], False),
    ],
)
def test_quark_is_w_ocp_mx_a_x(weight_config, expected):
    cfg = _make_quark_config()
    assert cfg._is_w_ocp_mx_a_x(weight_config, None) is expected


@pytest.mark.parametrize(
    ("config", "expected_cls", "expected_attrs"),
    [
        (
            {
                "weight": _FP8_PER_TENSOR_WEIGHT,
                "input_tensors": _FP8_DYNAMIC_PER_TENSOR_INPUT,
            },
            QuarkW8A8Fp8,
            {"is_static_input_scheme": False},
        ),
        (
            {
                "weight": _INT8_STATIC_PER_TENSOR,
                "input_tensors": _INT8_STATIC_PER_TENSOR,
            },
            QuarkW8A8Int8,
            {"qscheme": "per_tensor", "is_static_input_scheme": True},
        ),
        (
            {
                "weight": _MXFP4_PER_GROUP_WEIGHT,
                "input_tensors": _MXFP4_DYNAMIC_INPUT,
            },
            QuarkOCP_MX,
            {"weight_dtype": "mxfp4", "input_dtype": "mxfp4"},
        ),
    ],
)
def test_quark_get_scheme_from_config_dispatches_supported_schemes(
    default_vllm_config, config, expected_cls, expected_attrs
):
    default_vllm_config.model_config = MagicMock(dtype=torch.bfloat16)
    scheme = _make_quark_config()._get_scheme_from_config(config)

    assert isinstance(scheme, expected_cls)
    for attr_name, expected_value in expected_attrs.items():
        assert getattr(scheme, attr_name) == expected_value


def test_quark_get_scheme_from_config_rejects_unknown_scheme():
    config = {
        "weight": {"dtype": "int2", "qscheme": "per_tensor"},
        "input_tensors": {"dtype": "int2", "qscheme": "per_tensor"},
    }

    with pytest.raises(NotImplementedError, match="No quark compatible scheme"):
        _make_quark_config()._get_scheme_from_config(config)


def test_quark_get_scheme_from_config_rejects_output_tensor_quantization():
    with pytest.raises(NotImplementedError, match="output_tensors"):
        _make_quark_config()._get_scheme_from_config(
            {"output_tensors": {"dtype": "fp8_e4m3"}}
        )


def test_quark_find_matched_config_exact_match():
    cfg = _make_quark_config(
        quant_config={
            "layer_quant_config": {
                "model.layers.0.self_attn.q_proj": {
                    "weight": {"dtype": "fp8_e4m3"},
                },
            },
            "layer_type_quant_config": {},
            "global_quant_config": {},
        }
    )

    matched = cfg._find_matched_config(
        "model.layers.0.self_attn.q_proj",
        MagicMock(),
    )

    assert matched["weight"]["dtype"] == "fp8_e4m3"


def test_quark_find_matched_config_wildcard_match():
    cfg = _make_quark_config(
        quant_config={
            "layer_quant_config": {"*.q_proj": {"weight": {"dtype": "int8"}}},
            "layer_type_quant_config": {},
            "global_quant_config": {},
        }
    )

    matched = cfg._find_matched_config(
        "model.layers.5.self_attn.q_proj",
        MagicMock(),
    )

    assert matched["weight"]["dtype"] == "int8"


def test_quark_find_matched_config_falls_back_to_global_config():
    cfg = _make_quark_config(
        quant_config={
            "layer_quant_config": {},
            "layer_type_quant_config": {},
            "global_quant_config": {"weight": {"dtype": "fp8_e4m3"}},
        }
    )

    matched = cfg._find_matched_config(
        "model.layers.0.mlp.gate_proj",
        MagicMock(),
    )

    assert matched["weight"]["dtype"] == "fp8_e4m3"


def test_quark_find_matched_config_handles_fused_qkv_projection():
    fp8_config = {
        "weight": _FP8_PER_TENSOR_WEIGHT,
        "input_tensors": _FP8_DYNAMIC_PER_TENSOR_INPUT,
    }
    cfg = _make_quark_config(
        quant_config={
            "layer_quant_config": {
                "model.layers.0.self_attn.q_proj": fp8_config,
                "model.layers.0.self_attn.k_proj": fp8_config,
                "model.layers.0.self_attn.v_proj": fp8_config,
            },
            "layer_type_quant_config": {},
            "global_quant_config": {},
        }
    )
    cfg.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    matched = cfg._find_matched_config(
        "model.layers.0.self_attn.qkv_proj",
        MagicMock(),
    )

    assert matched["weight"]["dtype"] == "fp8_e4m3"


def test_quark_get_cache_scale_remaps_attention_names():
    cfg = _make_quark_config()

    assert (
        cfg.get_cache_scale("model.layers.0.self_attn.k_proj.output_scale")
        == "model.layers.0.self_attn.attn.k_scale"
    )
    assert (
        cfg.get_cache_scale("model.layers.0.self_attn.v_proj.output_scale")
        == "model.layers.0.self_attn.attn.v_scale"
    )
    assert cfg.get_cache_scale("model.layers.0.mlp.weight") is None


def test_quark_kv_cache_method_accepts_supported_config():
    QuarkKVCacheMethod.validate_kv_cache_config(
        {"dtype": "fp8_e4m3", "qscheme": "per_tensor"}
    )
    QuarkKVCacheMethod.validate_kv_cache_config(None)


@pytest.mark.parametrize(
    ("config", "error_match"),
    [
        ({"dtype": "int8", "qscheme": "per_tensor"}, "fp8_e4m3"),
        ({"dtype": "fp8_e4m3", "qscheme": "per_channel"}, "per_tensor"),
    ],
)
def test_quark_kv_cache_method_rejects_unsupported_config(config, error_match):
    with pytest.raises(NotImplementedError, match=error_match):
        QuarkKVCacheMethod.validate_kv_cache_config(config)


@pytest.mark.parametrize(
    ("weight_dtype", "input_dtype", "expected_weight_dtype", "expected_input_dtype"),
    [
        ("fp4", "fp4", "mxfp4", "mxfp4"),
        ("fp8", "fp8", "mxfp8", "mxfp8"),
        ("fp6_e3m2", "fp6_e3m2", "mxfp6_e3m2", "mxfp6_e3m2"),
    ],
)
def test_quark_ocp_mx_constructor_maps_quant_dtypes(
    weight_dtype,
    input_dtype,
    expected_weight_dtype,
    expected_input_dtype,
):
    scheme = QuarkOCP_MX(
        weight_quant_spec={
            **_MXFP4_PER_GROUP_WEIGHT,
            "dtype": weight_dtype,
        },
        input_quant_spec={
            **_MXFP4_DYNAMIC_INPUT,
            "dtype": input_dtype,
        },
    )

    assert scheme.weight_dtype == expected_weight_dtype
    assert scheme.input_dtype == expected_input_dtype


def test_quark_ocp_mx_dynamic_mxfp4_quant_flag():
    scheme = QuarkOCP_MX(
        weight_quant_spec=_MXFP4_PER_GROUP_WEIGHT,
        input_quant_spec=_MXFP4_DYNAMIC_INPUT,
        dynamic_mxfp4_quant=True,
    )

    assert scheme.dynamic_mxfp4_quant is True


@pytest.mark.parametrize(
    (
        "weight_config",
        "input_config",
        "expected_static_input",
        "expected_input_qscheme",
    ),
    [
        (
            {"dtype": "fp8", "qscheme": "per_tensor", "is_dynamic": False},
            {"dtype": "fp8", "qscheme": "per_tensor", "is_dynamic": True},
            False,
            "per_tensor",
        ),
        (
            {"dtype": "fp8", "qscheme": "per_tensor", "is_dynamic": False},
            {"dtype": "fp8", "qscheme": "per_tensor", "is_dynamic": False},
            True,
            "per_tensor",
        ),
        (
            {"dtype": "fp8", "qscheme": "per_channel", "is_dynamic": False},
            {"dtype": "fp8", "qscheme": "per_channel", "is_dynamic": False},
            True,
            "per_channel",
        ),
        (
            {"dtype": "fp8", "qscheme": "per_tensor", "is_dynamic": False},
            None,
            False,
            None,
        ),
    ],
)
def test_quark_w8a8_fp8_constructor_variants(
    default_vllm_config,
    weight_config,
    input_config,
    expected_static_input,
    expected_input_qscheme,
):
    from vllm.model_executor.layers.quantization.quark.schemes.quark_w8a8_fp8 import (
        QuarkW8A8Fp8,
    )

    default_vllm_config.model_config = MagicMock(dtype=torch.bfloat16)
    scheme = QuarkW8A8Fp8(weight_config=weight_config, input_config=input_config)

    assert scheme.is_static_input_scheme is expected_static_input
    assert scheme.input_qscheme == expected_input_qscheme
    assert scheme.input_dtype == torch.bfloat16


@pytest.mark.parametrize(
    ("qscheme", "is_static_input_scheme", "input_symmetric"),
    [
        ("per_tensor", False, True),
        ("per_channel", True, False),
    ],
)
def test_quark_w8a8_int8_constructor_variants(
    qscheme,
    is_static_input_scheme,
    input_symmetric,
):
    from vllm.model_executor.layers.quantization.quark.schemes.quark_w8a8_int8 import (
        QuarkW8A8Int8,
    )

    scheme = QuarkW8A8Int8(
        qscheme=qscheme,
        is_static_input_scheme=is_static_input_scheme,
        input_symmetric=input_symmetric,
    )

    assert scheme.qscheme == qscheme
    assert scheme.is_static_input_scheme is is_static_input_scheme
    assert scheme.input_symmetric is input_symmetric


def test_quark_moe_method_classes_are_importable():
    from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
    from vllm.model_executor.layers.quantization.quark.quark_moe import (
        QuarkMoEMethod,
        QuarkOCP_MX_MoEMethod,
        QuarkOCP_MX_MoEMethod_OSS,
        QuarkW4A8Fp8MoEMethod,
        QuarkW8A8Fp8MoEMethod,
    )

    assert issubclass(QuarkMoEMethod, FusedMoEMethodBase)
    assert issubclass(QuarkW8A8Fp8MoEMethod, QuarkMoEMethod)
    assert QuarkW4A8Fp8MoEMethod is not None
    assert QuarkOCP_MX_MoEMethod is not None
    assert issubclass(QuarkOCP_MX_MoEMethod_OSS, QuarkOCP_MX_MoEMethod)


def test_quark_scheme_classes_are_importable():
    from vllm.model_executor.layers.quantization.quark.schemes import (
        QuarkOCP_MX,
        QuarkScheme,
        QuarkW8A8Fp8,
        QuarkW8A8Int8,
    )

    for cls in [QuarkScheme, QuarkW8A8Fp8, QuarkW8A8Int8, QuarkOCP_MX]:
        assert isinstance(cls, type)


@pytest.mark.skipif(
    not current_platform.supports_fp8(),
    reason="FP8 is not supported on this GPU type.",
)
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_quark_fp8_per_tensor_model_loads_and_generates(vllm_runner, kv_cache_dtype):
    model_path = "amd/Llama-3.1-8B-Instruct-FP8-KV-Quark-test"

    with vllm_runner(
        model_path,
        enforce_eager=True,
        kv_cache_dtype=kv_cache_dtype,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.3,
    ) as llm:

        def check_model(model):
            qkv_proj = _get_first_qkv_proj(model)
            assert isinstance(qkv_proj.quant_method, QuarkLinearMethod)
            assert isinstance(qkv_proj.scheme, QuarkW8A8Fp8)
            assert qkv_proj.input_scale.ndim == 0
            assert qkv_proj.weight.dtype == current_platform.fp8_dtype()
            assert qkv_proj.weight_scale.ndim == 0

        llm.apply_model(check_model)
        outputs = llm.generate_greedy(_PARIS_PROMPTS, max_tokens=4)

    _assert_generation_succeeds(outputs, required_word="paris")


@pytest.mark.skipif(
    not current_platform.supports_fp8(),
    reason="FP8 is not supported on this GPU type.",
)
def test_quark_fp8_per_channel_model_loads_and_generates(vllm_runner):
    model_path = "amd/Qwen2.5-1.5B-Instruct-ptpc-Quark-ts"

    with vllm_runner(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.3,
    ) as llm:

        def check_model(model):
            qkv_proj = _get_first_qkv_proj(model)
            assert isinstance(qkv_proj.quant_method, QuarkLinearMethod)
            assert isinstance(qkv_proj.scheme, QuarkW8A8Fp8)
            assert qkv_proj.weight.dtype == current_platform.fp8_dtype()
            assert qkv_proj.weight_scale.shape == (qkv_proj.weight.shape[1], 1)

        llm.apply_model(check_model)
        outputs = llm.generate_greedy(_PARIS_PROMPTS, max_tokens=4)

    _assert_generation_succeeds(outputs, required_word="paris")


def test_quark_int8_model_loads_and_generates(vllm_runner):
    model_path = "amd/Llama-3.1-8B-Instruct-w-int8-a-int8-sym-test"

    with vllm_runner(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.3,
    ) as llm:

        def check_model(model):
            qkv_proj = _get_first_qkv_proj(model)
            assert isinstance(qkv_proj.quant_method, QuarkLinearMethod)
            assert isinstance(qkv_proj.scheme, QuarkW8A8Int8)

        llm.apply_model(check_model)
        outputs = llm.generate_greedy(_PARIS_PROMPTS, max_tokens=4)

    _assert_generation_succeeds(outputs, required_word="paris")


def test_quark_int8_moe_model_loads_and_generates(vllm_runner):
    model_path = "nameistoken/tiny-qwen3-moe-w8a8-int8-quark"

    with vllm_runner(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.1,
    ) as llm:

        def check_model(model):
            layer = model.model.layers[0]
            # MoE experts should use QuarkW8A8Int8MoEMethod
            moe = layer.mlp.experts
            assert isinstance(moe._quant_method, QuarkW8A8Int8MoEMethod), (
                f"Expected QuarkW8A8Int8MoEMethod, got {type(moe._quant_method)}"
            )
            # Non-MoE linear layers should use QuarkW8A8Int8
            qkv_proj = layer.self_attn.qkv_proj
            assert isinstance(qkv_proj.scheme, QuarkW8A8Int8)

        llm.apply_model(check_model)
        outputs = llm.generate_greedy(_SMOKE_PROMPTS, max_tokens=4)

    _assert_generation_succeeds(outputs)


@pytest.mark.skipif(
    not current_platform.supports_fp8(),
    reason="FP8 is not supported on this GPU type.",
)
def test_quark_fp8_parity(vllm_runner):
    quark_model_id = "amd-quark/llama-tiny-fp8-quark-quant-method"
    fp8_model_id = "amd-quark/llama-tiny-fp8-quant-method"

    llm_kwargs = {
        "tensor_parallel_size": 1,
        "enforce_eager": True,
        "gpu_memory_utilization": 0.1,
    }
    with (
        vllm_runner(quark_model_id, **llm_kwargs) as quark_handle,
        vllm_runner(fp8_model_id, **llm_kwargs) as fp8_handle,
    ):

        def get_state_dict(model):
            return {name: tensor.cpu() for name, tensor in model.state_dict().items()}

        (quark_state_dict,) = quark_handle.apply_model(get_state_dict)
        (fp8_state_dict,) = fp8_handle.apply_model(get_state_dict)

    assert fp8_state_dict.keys() == quark_state_dict.keys()

    for key in fp8_state_dict:
        assert torch.equal(fp8_state_dict[key], quark_state_dict[key])


@dataclass
class AccuracyTestConfig:
    model_name: str
    expected_value: float

    def get_model_args(
        self,
        tp_size: int,
        model_max_len: int | None = None,
        kwargs: dict | None = None,
    ) -> dict:
        model_args = {
            "pretrained": self.model_name,
            "dtype": "auto",
            "add_bos_token": True,
            "tensor_parallel_size": tp_size,
            "gpu_memory_utilization": 0.7,
        }
        if kwargs is not None:
            model_args.update(kwargs)
        if model_max_len is not None:
            model_args["max_model_len"] = model_max_len

        return model_args


GSM8K_ACCURACY_CONFIGS = [
    # Private model.
    AccuracyTestConfig(
        model_name="amd/DeepSeek-R1-WMXFP4-AMXFP4-Scale-UINT8-MoE-Quant",
        expected_value=0.96,
    ),
]

WIKITEXT_ACCURACY_CONFIGS = [
    AccuracyTestConfig(
        model_name="fxmarty/qwen1.5_moe_a2.7b_chat_w_fp4_a_fp6_e2m3",
        expected_value=11.3,
    ),
    AccuracyTestConfig(
        model_name="fxmarty/qwen1.5_moe_a2.7b_chat_w_fp6_e3m2_a_fp6_e3m2",
        expected_value=10.6,
    ),
    AccuracyTestConfig(
        model_name="fxmarty/qwen_1.5-moe-a2.7b-mxfp4",
        expected_value=12.4,
    ),
]


@pytest.mark.skipif(
    not QUARK_MXFP4_AVAILABLE,
    reason=f"amd-quark>={QUARK_MXFP4_MIN_VERSION} is not available",
)
@pytest.mark.parametrize("config", WIKITEXT_ACCURACY_CONFIGS)
@pytest.mark.parametrize(
    "tp_size",
    [
        pytest.param(1, id="tp1"),
        pytest.param(2, marks=multi_gpu_marks(num_gpus=2), id="tp2"),
    ],
)
def test_ocp_mx_wikitext_correctness(config: AccuracyTestConfig, tp_size: int):
    # Smaller cudagraph_capture_sizes to speed up the test.
    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=config.get_model_args(
            tp_size=tp_size,
            kwargs={"cudagraph_capture_sizes": [16]},
        ),
        tasks="wikitext",
        batch_size=64,
    )

    actual = results["results"]["wikitext"]["word_perplexity,none"]
    _assert_metric_close(
        actual=actual,
        expected=config.expected_value,
        tolerance=0.1,
        label=f"wikitext perplexity for {config.model_name}",
    )


@pytest.mark.skipif(
    not QUARK_MXFP4_AVAILABLE,
    reason=f"amd-quark>={QUARK_MXFP4_MIN_VERSION} is not available",
)
@pytest.mark.parametrize("config", GSM8K_ACCURACY_CONFIGS)
@multi_gpu_only(num_gpus=8)
def test_mxfp4_gsm8k_correctness(config: AccuracyTestConfig):
    if not _has_hf_amd_org_access():
        pytest.skip("Read access to huggingface.co/amd is required for this test.")

    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=config.get_model_args(tp_size=8, model_max_len=38768),
        tasks="gsm8k",
        batch_size=64,
        num_fewshot=8,
    )

    actual = results["results"]["gsm8k"]["exact_match,strict-match"]
    _assert_metric_close(
        actual=actual,
        expected=config.expected_value,
        tolerance=0.03,
        label=f"GSM8K exact-match for {config.model_name}",
    )


@pytest.mark.skipif(
    not QUARK_MXFP4_AVAILABLE,
    reason=f"amd-quark>={QUARK_MXFP4_MIN_VERSION} is not available",
)
@pytest.mark.parametrize("float_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scalings", [[2.3, 0.03, 7.3, 0.1, 0.004, 17.3, 1e4, 1e-4]])
def test_mxfp4_fused_qdq_matches_quark(float_dtype: torch.dtype, scalings: list[int]):
    _, mx_kernel, _ = _import_quark_mxfp4_modules()
    torch.manual_seed(0)

    hidden_size = 64 * 32
    inp = (torch.rand(1, hidden_size, dtype=float_dtype, device=DEVICE_TYPE) - 0.5) * 2
    for i in range(hidden_size // 32):
        sl = slice(i * 32, (i + 1) * 32)
        inp[:, sl] = inp[:, sl] * scalings[i % len(scalings)]

    res_hip = mx_kernel.qdq_mxfp4_hip(inp.clone(), "even")
    res_torch = qdq_mxfp4_torch(inp, "even")

    for i in range(hidden_size // 32):
        sl = slice(i * 32, (i + 1) * 32)
        assert torch.all(torch.isfinite(res_hip[:, sl]))
        assert torch.all(torch.isfinite(res_torch[:, sl]))
    torch.testing.assert_close(res_hip, res_torch, atol=0.0, rtol=0.0)


@pytest.mark.skipif(
    not QUARK_MXFP4_AVAILABLE,
    reason=f"amd-quark>={QUARK_MXFP4_MIN_VERSION} is not available",
)
@pytest.mark.parametrize("float_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scalings", [[2.3, 0.03, 7.3, 0.1, 0.004, 17.3, 1e4, 1e-4]])
def test_mxfp4_dequant_kernel_matches_quark(
    float_dtype: torch.dtype,
    scalings: list[int],
):
    StaticScaledRealQuantizer, mx_kernel, FP4PerGroupSpec = (
        _import_quark_mxfp4_modules()
    )
    qspec = FP4PerGroupSpec(
        ch_axis=-1,
        group_size=32,
        scale_format="e8m0",
        scale_calculation_mode="even",
        is_dynamic=False,
    ).to_quantization_spec()

    weight_quantizer = StaticScaledRealQuantizer(
        qspec=qspec,
        quantizer=None,
        reorder=False,
        real_quantized=True,
        float_dtype=float_dtype,
        device=DEVICE_TYPE,
    )

    observer = qspec.observer_cls(qspec, device=DEVICE_TYPE)

    hidden_size = 512
    shape = (11008, hidden_size)

    weights = (torch.rand(shape, device=DEVICE_TYPE, dtype=float_dtype) - 0.5) * 2

    # Make it so that different groups have different scales.
    for i in range(hidden_size // 32):
        sl = slice(i * 32, (i + 1) * 32)
        weights[:, sl] = weights[:, sl] * scalings[i % len(scalings)]

    observer(weights)
    scale, _ = observer._calculate_qparams()
    weight_quantizer.scale = scale

    weights_mxfp4 = weight_quantizer.to_real_quantize_params(weights).to(DEVICE_TYPE)
    weight_quantizer.maybe_convert_and_transpose_scale()

    scale = weight_quantizer.scale
    out_hip = mx_kernel.dq_mxfp4_hip(weights_mxfp4, scale, float_dtype)
    out_torch = dq_mxfp4_torch(weights_mxfp4, scale, float_dtype)

    torch.testing.assert_close(out_hip, out_torch, atol=0.0, rtol=0.0)
