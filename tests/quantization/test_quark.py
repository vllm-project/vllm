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
    QuarkLinearMethod,
    QuarkW8A8Fp8,
    QuarkW8A8Int8,
)
from vllm.platforms import current_platform

from .reference_mxfp4 import dq_mxfp4_torch, qdq_mxfp4_torch

QUARK_MXFP4_AVAILABLE = find_spec("quark") is not None and version.parse(
    importlib.metadata.version("amd-quark")
) >= version.parse("0.8.99")

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


@pytest.mark.skipif(not QUARK_MXFP4_AVAILABLE, reason="amd-quark>=0.9 is not available")
@pytest.mark.parametrize("config", WIKITEXT_ACCURACY_CONFIGS)
@pytest.mark.parametrize("tp_size", [1, 2])
def test_ocp_mx_wikitext_correctness(config: AccuracyTestConfig, tp_size: int):
    if torch.cuda.device_count() < tp_size:
        pytest.skip(
            f"This test requires >={tp_size} gpus, got only {torch.cuda.device_count()}"
        )

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


@pytest.mark.parametrize("config", GSM8K_ACCURACY_CONFIGS)
@pytest.mark.skipif(not QUARK_MXFP4_AVAILABLE, reason="amd-quark>=0.9 is not available")
@pytest.mark.skipif(
    not HF_HUB_AMD_ORG_ACCESS,
    reason="Read access to huggingface.co/amd is required for this test.",
)
def test_mxfp4_gsm8k_correctness(config: AccuracyTestConfig):
    if torch.cuda.device_count() < 8:
        pytest.skip(
            f"This test requires >=8 gpus, got only {torch.cuda.device_count()}"
        )

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


@pytest.mark.skipif(not QUARK_MXFP4_AVAILABLE, reason="amd-quark>=0.9 is not available")
@pytest.mark.parametrize("float_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scalings", [[2.3, 0.03, 7.3, 0.1, 0.004, 17.3, 1e4, 1e-4]])
def test_mxfp4_fused_qdq_match_quark(float_dtype: torch.dtype, scalings: list[int]):
    torch.manual_seed(0)

    hidden_size = 64 * 32
    inp = (torch.rand(1, hidden_size, dtype=float_dtype, device="cuda") - 0.5) * 2
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


@pytest.mark.skipif(not QUARK_MXFP4_AVAILABLE, reason="amd-quark>=0.9 is not available")
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
        device="cuda",
    )

    observer = qspec.observer_cls(qspec, device="cuda")

    hidden_size = 512
    shape = (11008, hidden_size)

    w = (torch.rand(shape, device="cuda", dtype=float_dtype) - 0.5) * 2

    # Make it so that different groups have different scales.
    for i in range(hidden_size // 32):
        w[:, i * 32 : (i + 1) * 32] = (
            w[:, i * 32 : (i + 1) * 32] * scalings[i % len(scalings)]
        )

    observer(w)
    scale, _ = observer._calculate_qparams()
    weight_quantizer.scale = scale

    w_mxfp4 = weight_quantizer.to_real_quantize_params(w).to("cuda")
    weight_quantizer.maybe_convert_and_transpose_scale()

    scale = weight_quantizer.scale

    out_hip = mx_kernel.dq_mxfp4_hip(w_mxfp4, scale, float_dtype)

    out_torch = dq_mxfp4_torch(w_mxfp4, scale, float_dtype)

    assert torch.equal(out_hip, out_torch)
