# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test model set-up and weight loading for quark-quantized models.

Run `pytest tests/quantization/test_quark.py`.

See also `tests/kernels/moe/test_ocp_mx_moe.py`.
"""

import importlib.metadata
from dataclasses import dataclass
from importlib.util import find_spec

import huggingface_hub
import lm_eval
import pytest
import torch
from packaging import version

from vllm.model_executor.layers.quantization.quark.quark import (  # noqa: E501
    QuarkConfig,
    QuarkLinearMethod,
    QuarkW8A8Fp8,
    QuarkW8A8Int8,
)
from vllm.model_executor.layers.quantization.quark.quark_moe import (  # noqa: E501
    QuarkW4A16Int4MoEMethod,
    QuarkW8A8Int8MoEMethod,
)
from vllm.model_executor.layers.quantization.quark.schemes import QuarkW4A16Int4
from vllm.model_executor.layers.quantization.quark.utils import (
    canonicalize_quark_packed_int4,
    should_ignore_layer,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.models.utils import WeightsMapper
from vllm.platforms import current_platform

if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx950
else:

    def on_gfx950() -> bool:
        return False


from .reference_mxfp4 import dq_mxfp4_torch, qdq_mxfp4_torch

# Minimum amd-quark version for MXFP4/OCP_MX tests (single source of truth).
QUARK_MXFP4_MIN_VERSION = "0.12"

QUARK_MXFP4_AVAILABLE = find_spec("quark") is not None and version.parse(
    importlib.metadata.version("amd-quark")
) >= version.parse(QUARK_MXFP4_MIN_VERSION)

DEVICE_TYPE = current_platform.device_type

if QUARK_MXFP4_AVAILABLE:
    from quark.torch.export.nn.modules.realquantizer import StaticScaledRealQuantizer
    from quark.torch.kernel import mx as mx_kernel
    from quark.torch.quantization.config.config import FP4PerGroupSpec

try:
    huggingface_hub.list_repo_refs(
        "amd/Llama-3.3-70B-Instruct-WMXFP4-AMXFP4-KVFP8-Scale-UINT8-SQ"
    )
    HF_HUB_AMD_ORG_ACCESS = True
except huggingface_hub.errors.RepositoryNotFoundError:
    HF_HUB_AMD_ORG_ACCESS = False


@pytest.fixture(scope="function", autouse=True)
def enable_pickle(monkeypatch):
    """`LLM.apply_model` requires pickling a function."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("tp", [1])
def test_quark_fp8_w_per_tensor_a_per_tensor(vllm_runner, kv_cache_dtype, tp):
    model_path = "amd/Llama-3.1-8B-Instruct-FP8-KV-Quark-test"
    with vllm_runner(
        model_path,
        enforce_eager=True,
        kv_cache_dtype=kv_cache_dtype,
        tensor_parallel_size=tp,
    ) as llm:

        def check_model(model):
            layer = model.model.layers[0]

            qkv_proj = layer.self_attn.qkv_proj

            assert isinstance(qkv_proj.quant_method, QuarkLinearMethod)
            assert isinstance(qkv_proj.scheme, QuarkW8A8Fp8)

            if isinstance(qkv_proj.scheme, QuarkW8A8Fp8):
                assert len(qkv_proj.input_scale.shape) == 0
                assert qkv_proj.weight.dtype is current_platform.fp8_dtype()
                assert len(qkv_proj.weight_scale.shape) == 0

        llm.apply_model(check_model)

        output = llm.generate_greedy("Hello my name is", max_tokens=4)
        assert output


@pytest.mark.parametrize("tp", [1])
def test_quark_fp8_w_per_channel_a_per_token(vllm_runner, tp):
    model_path = "amd/Qwen2.5-1.5B-Instruct-ptpc-Quark-ts"
    with vllm_runner(model_path, enforce_eager=True, tensor_parallel_size=tp) as llm:

        def check_model(model):
            layer = model.model.layers[0]

            qkv_proj = layer.self_attn.qkv_proj

            assert isinstance(qkv_proj.quant_method, QuarkLinearMethod)
            assert isinstance(qkv_proj.scheme, QuarkW8A8Fp8)

            if isinstance(qkv_proj.scheme, QuarkW8A8Fp8):
                assert qkv_proj.weight.dtype is current_platform.fp8_dtype()
                assert qkv_proj.weight_scale.shape[0] == qkv_proj.weight.shape[1]
                assert qkv_proj.weight_scale.shape[1] == 1

        llm.apply_model(check_model)

        output = llm.generate_greedy("Hello my name is", max_tokens=4)
        assert output


@pytest.mark.parametrize("tp", [1])
def test_quark_int8_w_per_tensor_a_per_tensor(vllm_runner, tp):
    model_path = "amd/Llama-3.1-8B-Instruct-w-int8-a-int8-sym-test"
    with vllm_runner(model_path, enforce_eager=True, tensor_parallel_size=tp) as llm:

        def check_model(model):
            layer = model.model.layers[0]

            qkv_proj = layer.self_attn.qkv_proj

            assert isinstance(qkv_proj.quant_method, QuarkLinearMethod)
            assert isinstance(qkv_proj.scheme, QuarkW8A8Int8)

        llm.apply_model(check_model)

        output = llm.generate_greedy("Hello my name is", max_tokens=4)
        assert output


@pytest.mark.parametrize("tp", [1])
def test_quark_int8_w8a8_moe(vllm_runner, tp):
    """Test W8A8 INT8 MoE quantization with a tiny Qwen3 MoE model."""
    model_path = "nameistoken/tiny-qwen3-moe-w8a8-int8-quark"
    with vllm_runner(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=tp,
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

        output = llm.generate_greedy("Hello", max_tokens=4)
        assert output


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
            return {k: v.cpu() for k, v in model.state_dict().items()}

        (quark_state_dict,) = quark_handle.apply_model(get_state_dict)
        (fp8_state_dict,) = fp8_handle.apply_model(get_state_dict)

    assert fp8_state_dict.keys() == quark_state_dict.keys()

    for key in fp8_state_dict:
        assert torch.equal(fp8_state_dict[key], quark_state_dict[key])


@dataclass
class AccuracyTestConfig:
    model_name: str
    excepted_value: float

    def get_model_args(
        self,
        tp_size: int,
        model_max_len: int | None = None,
        kwargs: dict | None = None,
    ) -> dict:
        if kwargs is None:
            kwargs = {}

        model_args = {
            "pretrained": self.model_name,
            "dtype": "auto",
            "add_bos_token": True,
            "tensor_parallel_size": tp_size,
            "gpu_memory_utilization": 0.7,
            **kwargs,
        }
        if model_max_len is not None:
            model_args["max_model_len"] = model_max_len

        return model_args


GSM8K_ACCURACY_CONFIGS = [
    # Private model.
    AccuracyTestConfig(
        model_name="amd/DeepSeek-R1-WMXFP4-AMXFP4-Scale-UINT8-MoE-Quant",
        excepted_value=0.96,
    ),
]

WIKITEXT_ACCURACY_CONFIGS = [
    AccuracyTestConfig(
        model_name="fxmarty/qwen1.5_moe_a2.7b_chat_w_fp4_a_fp6_e2m3",
        excepted_value=11.3,
    ),
    AccuracyTestConfig(
        model_name="fxmarty/qwen1.5_moe_a2.7b_chat_w_fp6_e3m2_a_fp6_e3m2",
        excepted_value=10.6,
    ),
    AccuracyTestConfig(
        model_name="fxmarty/qwen_1.5-moe-a2.7b-mxfp4", excepted_value=12.4
    ),
]


@pytest.mark.skipif(
    not QUARK_MXFP4_AVAILABLE,
    reason=f"amd-quark>={QUARK_MXFP4_MIN_VERSION} is not available",
)
@pytest.mark.parametrize(
    "config",
    [pytest.param(val, id=f"config:{val}") for val in WIKITEXT_ACCURACY_CONFIGS],
)
@pytest.mark.parametrize(
    "tp_size", [pytest.param(val, id=f"tp_size:{val}") for val in [1, 2]]
)
def test_ocp_mx_wikitext_correctness(config: AccuracyTestConfig, tp_size: int):
    device_count = torch.accelerator.device_count()
    if device_count < tp_size:
        pytest.skip(f"This test requires >={tp_size} gpus, got only {device_count}")

    task = "wikitext"
    rtol = 0.1

    # Smaller cudagraph_capture_sizes to speed up the test.
    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=config.get_model_args(
            tp_size=tp_size, kwargs={"cudagraph_capture_sizes": [16]}
        ),
        tasks=task,
        batch_size=64,
    )

    EXPECTED_VALUE = config.excepted_value
    measured_value = results["results"][task]["word_perplexity,none"]
    assert (
        measured_value < EXPECTED_VALUE + rtol
        and measured_value > EXPECTED_VALUE - rtol
    ), f"Expected: {EXPECTED_VALUE} |  Measured: {measured_value}"


@pytest.mark.skipif(
    not QUARK_MXFP4_AVAILABLE,
    reason=f"amd-quark>={QUARK_MXFP4_MIN_VERSION} is not available",
)
@pytest.mark.parametrize("tp_size", [1, 2])
def test_nvfp4_wikitext_correctness(tp_size: int):
    device_count = torch.accelerator.device_count()
    if device_count < tp_size:
        pytest.skip(f"This test requires >={tp_size} gpus, got only {device_count}")

    # NOTE: expected_value from nvidia/Qwen3-30B-A3B-NVFP4
    expected_value = 11.2391

    model_name = "amd-quark/Qwen3-30B-A3B-nvfp4-quark"
    task = "wikitext"

    rtol = 0.25

    config = AccuracyTestConfig(
        model_name=model_name,
        excepted_value=expected_value,
    )

    model_args = config.get_model_args(
        tp_size=tp_size,
        kwargs={
            "cudagraph_capture_sizes": [16],
        },
    )
    model_args.pop("add_bos_token")

    # Smaller cudagraph_capture_sizes to speed up the test.
    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=task,
        batch_size=64,
    )

    EXPECTED_VALUE = config.excepted_value
    measured_value = results["results"][task]["word_perplexity,none"]
    assert (
        measured_value < EXPECTED_VALUE + rtol
        and measured_value > EXPECTED_VALUE - rtol
    ), f"Expected: {EXPECTED_VALUE} |  Measured: {measured_value}"


@pytest.mark.parametrize("config", GSM8K_ACCURACY_CONFIGS)
@pytest.mark.skipif(
    not QUARK_MXFP4_AVAILABLE,
    reason=f"amd-quark>={QUARK_MXFP4_MIN_VERSION} is not available",
)
@pytest.mark.skipif(
    not HF_HUB_AMD_ORG_ACCESS,
    reason="Read access to huggingface.co/amd is required for this test.",
)
def test_mxfp4_gsm8k_correctness(config: AccuracyTestConfig):
    device_count = torch.accelerator.device_count()
    if device_count < 8:
        pytest.skip(f"This test requires >=8 gpus, got only {device_count}")

    task = "gsm8k"
    rtol = 0.03

    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=config.get_model_args(tp_size=8, model_max_len=38768),
        tasks=task,
        batch_size=64,
        num_fewshot=8,
    )

    EXPECTED_VALUE = config.excepted_value
    measured_value = results["results"][task]["exact_match,strict-match"]
    assert (
        measured_value - rtol < EXPECTED_VALUE
        and measured_value + rtol > EXPECTED_VALUE
    ), f"Expected: {EXPECTED_VALUE} |  Measured: {measured_value}"


@pytest.mark.skipif(
    not QUARK_MXFP4_AVAILABLE,
    reason=f"amd-quark>={QUARK_MXFP4_MIN_VERSION} is not available",
)
@pytest.mark.parametrize("float_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scalings", [[2.3, 0.03, 7.3, 0.1, 0.004, 17.3, 1e4, 1e-4]])
def test_mxfp4_fused_qdq_match_quark(float_dtype: torch.dtype, scalings: list[int]):
    torch.manual_seed(0)

    hidden_size = 64 * 32
    inp = (torch.rand(1, hidden_size, dtype=float_dtype, device=DEVICE_TYPE) - 0.5) * 2
    for i in range(hidden_size // 32):
        inp[:, i * 32 : (i + 1) * 32] = (
            inp[:, i * 32 : (i + 1) * 32] * scalings[i % len(scalings)]
        )

    inp_kernel = inp.clone()
    inp_kernel_clone = inp_kernel.clone()

    res_hip = mx_kernel.qdq_mxfp4_hip(inp_kernel_clone, "even")
    res_torch = qdq_mxfp4_torch(inp_kernel, "even")

    for i in range(hidden_size // 32):
        assert torch.all(torch.isfinite(res_hip[:, i * 32 : (i + 1) * 32]))
        assert torch.all(torch.isfinite(res_torch[:, i * 32 : (i + 1) * 32]))

        torch.testing.assert_close(
            res_hip[:, i * 32 : (i + 1) * 32], res_torch[:, i * 32 : (i + 1) * 32]
        )


@pytest.mark.skipif(
    not QUARK_MXFP4_AVAILABLE,
    reason=f"amd-quark>={QUARK_MXFP4_MIN_VERSION} is not available",
)
@pytest.mark.parametrize("float_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scalings", [[2.3, 0.03, 7.3, 0.1, 0.004, 17.3, 1e4, 1e-4]])
def test_mxfp4_dequant_kernel_match_quark(
    float_dtype: torch.dtype, scalings: list[int]
):
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

    w = (torch.rand(shape, device=DEVICE_TYPE, dtype=float_dtype) - 0.5) * 2

    # Make it so that different groups have different scales.
    for i in range(hidden_size // 32):
        w[:, i * 32 : (i + 1) * 32] = (
            w[:, i * 32 : (i + 1) * 32] * scalings[i % len(scalings)]
        )

    observer(w)
    scale, _ = observer._calculate_qparams()
    weight_quantizer.scale = scale

    w_mxfp4 = weight_quantizer.to_real_quantize_params(w).to(DEVICE_TYPE)
    weight_quantizer.maybe_convert_and_transpose_scale()

    scale = weight_quantizer.scale

    out_hip = mx_kernel.dq_mxfp4_hip(w_mxfp4, scale, float_dtype)

    out_torch = dq_mxfp4_torch(w_mxfp4, scale, float_dtype)

    assert torch.equal(out_hip, out_torch)


# Unit tests for ``is_layer_skipped`` fused-name handling.

FUSED_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def test_fused_name_listed_directly_is_skipped():
    # Regression for Step-3.5-Flash-FP8: the checkpoint lists the fused
    # name (``qkv_proj``) directly in ``modules_to_not_convert``. When a
    # ``packed_modules_mapping`` is registered on the model, the fused
    # match must still win over per-shard expansion.
    ignored = ["model.layers.0.self_attn.qkv_proj"]
    assert is_layer_skipped(
        prefix="model.layers.0.self_attn.qkv_proj",
        ignored_layers=ignored,
        fused_mapping=FUSED_MAPPING,
    )
    assert is_layer_skipped(
        prefix="model.layers.0.mlp.gate_up_proj",
        ignored_layers=["model.layers.0.mlp.gate_up_proj"],
        fused_mapping=FUSED_MAPPING,
    )


def test_unfused_shards_listed_is_skipped():
    # Quark INT8 style: per-shard names listed; all shards present means
    # the fused layer is skipped via expansion.
    ignored = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
    ]
    assert is_layer_skipped(
        prefix="model.layers.0.self_attn.qkv_proj",
        ignored_layers=ignored,
        fused_mapping=FUSED_MAPPING,
    )


def test_partial_shards_raises():
    # Only some shards listed -> ambiguous, must raise. Fused name is
    # not in ignored_layers, so we fall through to per-shard expansion.
    ignored = ["model.layers.0.self_attn.q_proj"]
    with pytest.raises(ValueError):
        is_layer_skipped(
            prefix="model.layers.0.self_attn.qkv_proj",
            ignored_layers=ignored,
            fused_mapping=FUSED_MAPPING,
        )


def test_not_skipped_when_nothing_listed():
    assert not is_layer_skipped(
        prefix="model.layers.0.self_attn.qkv_proj",
        ignored_layers=["model.layers.0.mlp.gate_up_proj"],
        fused_mapping=FUSED_MAPPING,
    )


def test_non_fused_layer_unaffected():
    assert is_layer_skipped(
        prefix="model.layers.0.self_attn.o_proj",
        ignored_layers=["model.layers.0.self_attn.o_proj"],
        fused_mapping=FUSED_MAPPING,
    )
    assert not is_layer_skipped(
        prefix="model.layers.0.self_attn.o_proj",
        ignored_layers=["model.layers.1.self_attn.o_proj"],
        fused_mapping=FUSED_MAPPING,
    )


def test_substr_match_on_fused_name():
    # skip_with_substr=True path: fused-name substring match should also
    # short-circuit before shard expansion.
    assert is_layer_skipped(
        prefix="model.layers.0.self_attn.qkv_proj",
        ignored_layers=["self_attn.qkv_proj"],
        fused_mapping=FUSED_MAPPING,
        skip_with_substr=True,
    )


_REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def _quark_int4_config(
    *,
    pack_method: str = "reorder",
    symmetric: bool = True,
    exclude: list[str] | None = None,
) -> dict:
    return {
        "quant_method": "quark",
        "export": {"pack_method": pack_method, "kv_cache_group": []},
        "global_quant_config": {
            "weight": {
                "dtype": "int4",
                "group_size": 128,
                "symmetric": symmetric,
            }
        },
        "exclude": exclude or [],
    }


def _sign_extend_int4_nibbles(t: torch.Tensor) -> torch.Tensor:
    mask = (t & 0x8).bool()
    t = t.clone()
    t[mask] = t[mask] | 0xF0
    return t


def _dequantize_quark_signed_awq_torch(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
    *,
    pack_reorder: bool = True,
) -> torch.Tensor:
    bits = 4
    shifts = torch.arange(0, 32, bits, device=qweight.device)
    iweights = ((qweight[:, :, None] >> shifts[None, None, :]) & 0xF).to(torch.int8)
    iweights = iweights.view(qweight.shape[0], -1)
    zeros = ((qzeros[:, :, None] >> shifts[None, None, :]) & 0xF).to(torch.int8)
    zeros = zeros.view(qzeros.shape[0], -1)

    if pack_reorder:
        order = torch.tensor(_REVERSE_AWQ_PACK_ORDER, device=qweight.device)
    else:
        order = torch.arange(8, device=qweight.device)
    iweights = iweights.view(qweight.shape[0], -1, 8)[:, :, order].reshape(
        qweight.shape[0], -1
    )
    zeros = zeros.view(qzeros.shape[0], -1, 8)[:, :, order].reshape(
        qzeros.shape[0], -1
    )
    iweights = _sign_extend_int4_nibbles(iweights & 0xF)
    zeros = _sign_extend_int4_nibbles(zeros & 0xF)

    scales = scales.repeat_interleave(group_size, dim=0)
    zeros = zeros.repeat_interleave(group_size, dim=0)
    return (iweights - zeros) * scales


def _dequantize_awq_unsigned_torch(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
    *,
    pack_reorder: bool = True,
) -> torch.Tensor:
    bits = 4
    shifts = torch.arange(0, 32, bits, device=qweight.device)
    iweights = ((qweight[:, :, None] >> shifts[None, None, :]) & 0xF).to(torch.int8)
    iweights = iweights.view(qweight.shape[0], -1)
    zeros = ((qzeros[:, :, None] >> shifts[None, None, :]) & 0xF).to(torch.int8)
    zeros = zeros.view(qzeros.shape[0], -1)

    if pack_reorder:
        order = torch.tensor(_REVERSE_AWQ_PACK_ORDER, device=qweight.device)
    else:
        order = torch.arange(8, device=qweight.device)
    iweights = iweights.view(qweight.shape[0], -1, 8)[:, :, order].reshape(
        qweight.shape[0], -1
    )
    zeros = zeros.view(qzeros.shape[0], -1, 8)[:, :, order].reshape(
        qzeros.shape[0], -1
    )

    scales = scales.repeat_interleave(group_size, dim=0)
    zeros = zeros.repeat_interleave(group_size, dim=0)
    return (iweights - zeros) * scales


def _pack_int4_nibbles(nibbles: torch.Tensor, *, pack_reorder: bool) -> torch.Tensor:
    pack_order = (
        torch.tensor(_REVERSE_AWQ_PACK_ORDER, device=nibbles.device)
        if pack_reorder
        else torch.arange(8, device=nibbles.device)
    )
    shifts = pack_order * 4
    return ((nibbles.to(torch.int64) & 0xF) << shifts).sum(dim=-1).to(torch.int32)


class TestQuarkInt4Format:
    """Tests for Quark INT4 export format compatibility."""

    def test_quark_order_pack_method_uses_native_int4_scheme(self):
        quant_config = QuarkConfig.from_config(_quark_int4_config(pack_method="order"))
        scheme = quant_config._get_scheme_from_config(
            quant_config.quant_config["global_quant_config"]
        )

        assert isinstance(scheme, QuarkW4A16Int4)
        assert not scheme.pack_reorder

    def test_quark_int4_scheme_supports_asymmetric_weights(self):
        quant_config = QuarkConfig.from_config(_quark_int4_config(symmetric=False))
        scheme = quant_config._get_scheme_from_config(
            quant_config.quant_config["global_quant_config"]
        )

        assert isinstance(scheme, QuarkW4A16Int4)
        assert not scheme.is_symmetric

    @pytest.mark.parametrize("missing_field", ["group_size", "symmetric"])
    def test_quark_int4_scheme_requires_weight_config_fields(self, missing_field):
        weight_config = {
            "dtype": "int4",
            "group_size": 128,
            "symmetric": True,
        }
        weight_config.pop(missing_field)
        quant_config = QuarkConfig.from_config(
            {
                "quant_method": "quark",
                "export": {"pack_method": "reorder", "kv_cache_group": []},
                "global_quant_config": {"weight": weight_config},
                "exclude": [],
            }
        )

        with pytest.raises(ValueError, match=missing_field):
            quant_config._get_scheme_from_config(
                quant_config.quant_config["global_quant_config"]
            )

    def test_quark_int4_moe_uses_native_moe_method(self):
        quant_config = QuarkConfig.from_config(_quark_int4_config())
        moe_config = type("MoeConfig", (), {})()
        moe_config.has_bias = False

        method = QuarkW4A16Int4MoEMethod(
            quant_config.quant_config["global_quant_config"]["weight"],
            quant_config.pack_method,
            moe_config,
        )

        assert method.group_size == 128
        assert method.pack_reorder

    def test_quark_shared_expert_gate_keeps_quantized_tensors(self):
        quant_config = QuarkConfig.from_config(_quark_int4_config())
        mapper = quant_config.get_cache_scale_mapper()

        output_names = {
            name
            for name, _ in mapper.apply(
                [
                    (
                        "model.language_model.layers.0.mlp.shared_expert_gate.weight",
                        torch.zeros(1),
                    ),
                    (
                        "model.language_model.layers.0.mlp.shared_expert_gate"
                        ".weight_scale",
                        torch.zeros(1),
                    ),
                    (
                        "model.language_model.layers.0.mlp.shared_expert_gate"
                        ".weight_zero_point",
                        torch.zeros(1),
                    ),
                ]
            )
        }

        assert (
            "model.language_model.layers.0.mlp.shared_expert_gate.weight"
            in output_names
        )
        assert (
            "model.language_model.layers.0.mlp.shared_expert_gate.weight_scale"
            in output_names
        )
        assert (
            "model.language_model.layers.0.mlp.shared_expert_gate.weight_zero_point"
            in output_names
        )

    def test_quark_mapper_adds_suffix_remappings(self):
        quant_config = QuarkConfig.from_config(_quark_int4_config(symmetric=False))
        mapper = quant_config.get_cache_scale_mapper()

        assert ".qscales" in mapper.orig_to_new_suffix
        assert mapper.orig_to_new_suffix[".qscales"] == ".weight_scale"
        assert ".qqzeros" in mapper.orig_to_new_suffix
        assert mapper.orig_to_new_suffix[".qqzeros"] == ".weight_zero_point"

    def test_quark_apply_mapper_updates_exclude_and_layer_quant_config(self):
        quant_config = QuarkConfig.from_config(
            {
                **_quark_int4_config(),
                "exclude": ["lm_head"],
                "layer_quant_config": {
                    "model.language_model.layers.0.mlp.gate": {
                        "weight": {"dtype": "float16"},
                    },
                },
            }
        )
        quant_config.apply_vllm_mapper(
            WeightsMapper(
                orig_to_new_prefix={
                    "lm_head": "language_model.lm_head",
                    "model.language_model.": "language_model.model.",
                },
            )
        )

        layer_quant_config = quant_config.quant_config["layer_quant_config"]
        assert "language_model.model.layers.0.mlp.gate" in layer_quant_config
        assert "model.language_model.layers.0.mlp.gate" not in layer_quant_config
        assert quant_config.quant_config["exclude"] == [
            "language_model.lm_head",
        ]

    def test_quark_bare_exclude_matches_nested_module_prefix(self):
        quant_config = QuarkConfig.from_config(_quark_int4_config(exclude=["lm_head"]))
        quant_config.apply_vllm_mapper(WeightsMapper())

        exclude_layers = quant_config.quant_config["exclude"]
        assert should_ignore_layer("language_model.lm_head", ignore=exclude_layers)
        assert not should_ignore_layer(
            "language_model.model.layers.0.mlp.gate", ignore=exclude_layers
        )

    def test_quark_mapper_renames_tensor_names(self):
        quant_config = QuarkConfig.from_config(_quark_int4_config(symmetric=False))
        mapper = quant_config.get_cache_scale_mapper()

        input_weights = [
            ("model.layers.0.mlp.down_proj.weight", torch.zeros(1)),
            ("model.layers.0.mlp.down_proj.qscales", torch.zeros(1)),
            ("model.layers.0.mlp.down_proj.qqzeros", torch.zeros(1)),
        ]
        output_names = {name for name, _ in mapper.apply(input_weights)}

        assert "model.layers.0.mlp.down_proj.weight" in output_names
        assert "model.layers.0.mlp.down_proj.weight_scale" in output_names
        assert "model.layers.0.mlp.down_proj.weight_zero_point" in output_names
        assert "model.layers.0.mlp.down_proj.qscales" not in output_names
        assert "model.layers.0.mlp.down_proj.qqzeros" not in output_names

@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("pack_method", ["order", "reorder"])
def test_quark_int4_canonicalizes_pack_for_kernel_layout(pack_method, symmetric):
    """Quark pack order is normalized to the layout expected by awq_* ops.

    This reuses the existing AWQ dequant/gemm kernels for compute only; loading
    still goes through the native Quark quantization path, not AutoAWQ.
    """
    pack_reorder = pack_method == "reorder"
    group_size = 2
    packed_values = torch.tensor(
        [
            [0, 1, 7, 8, 9, 15, 2, 14],
            [15, 8, 0, 3, 12, 7, 1, 9],
        ],
        dtype=torch.int32,
    )
    packed_zeros = torch.zeros((1, 8), dtype=torch.int32)
    qweight = _pack_int4_nibbles(packed_values, pack_reorder=pack_reorder).view(2, 1)
    qzeros = _pack_int4_nibbles(packed_zeros, pack_reorder=pack_reorder).view(1, 1)
    scales = torch.ones((1, 8), dtype=torch.float16)

    dequantize = (
        _dequantize_quark_signed_awq_torch
        if symmetric
        else _dequantize_awq_unsigned_torch
    )
    expected = dequantize(
        qweight,
        scales,
        qzeros,
        group_size,
        pack_reorder=pack_reorder,
    )
    scheme = QuarkW4A16Int4(
        group_size=group_size, pack_method=pack_method, is_symmetric=symmetric
    )
    canonical_weight = canonicalize_quark_packed_int4(
        qweight,
        pack_reorder=pack_reorder,
        is_symmetric=symmetric,
    )
    canonical_zero = canonicalize_quark_packed_int4(
        qzeros,
        pack_reorder=pack_reorder,
        is_symmetric=symmetric,
    )
    actual = _dequantize_awq_unsigned_torch(
        canonical_weight, scales, canonical_zero, group_size
    )

    torch.testing.assert_close(actual, expected)
