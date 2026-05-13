# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util

import pytest
import torch

from tests.quantization._zentorch_helpers import zentorch_ops_mock  # noqa: F401
from vllm.model_executor.layers.quantization.torchao import torchao_version_at_least
from vllm.model_executor.model_loader import get_model_loader
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type
DTYPE = ["bfloat16"]

TORCHAO_AVAILABLE = importlib.util.find_spec("torchao") is not None
TORCHAO_VERSION_AT_LEAST_0_17_0 = torchao_version_at_least("0.17.0")


@pytest.mark.skipif(
    current_platform.is_rocm() and current_platform.is_fp8_fnuz(),
    reason="Only fp8_fnuz supported on CDNA3 architecture",
)
@pytest.mark.skipif(not TORCHAO_AVAILABLE, reason="torchao is not available")
def test_pre_quantized_model(vllm_runner):
    with vllm_runner(
        "torchao-testing/opt-125m-Float8WeightOnlyConfig-v2-0.15.0",
        quantization="torchao",
        dtype="bfloat16",
        enforce_eager=True,
    ) as llm:
        output = llm.generate_greedy(["The capital of France is"], max_tokens=4)
    assert output


@pytest.mark.skipif(not TORCHAO_AVAILABLE, reason="torchao is not available")
@pytest.mark.parametrize(
    "pt_load_map_location",
    [
        f"{DEVICE_TYPE}:0",
        # {"": "cuda"},
    ],
)
def test_opt_125m_int8wo_model_loading_with_params(vllm_runner, pt_load_map_location):
    torch._dynamo.reset()
    model_name = "jerryzh168/opt-125m-int8wo-partial-quant"
    with vllm_runner(
        model_name=model_name,
        quantization="torchao",
        dtype="bfloat16",
        pt_load_map_location=pt_load_map_location,
        enforce_eager=True,
    ) as llm:
        output = llm.generate_greedy(["The capital of France is"], max_tokens=4)

        assert output


@pytest.mark.skipif(not TORCHAO_AVAILABLE, reason="torchao is not available")
def test_qwenvl_int8wo_model_loading_with_params(vllm_runner):
    torch._dynamo.reset()
    model_name = "mobicham/Qwen2.5-VL-3B-Instruct_int8wo_ao"
    with vllm_runner(
        model_name=model_name,
        quantization="torchao",
        dtype="bfloat16",
        pt_load_map_location=f"{DEVICE_TYPE}:0",
        enforce_eager=True,
    ) as llm:
        output = llm.generate_greedy(["The capital of France is"], max_tokens=4)

        assert output


@pytest.mark.skipif(not TORCHAO_AVAILABLE, reason="torchao is not available")
@pytest.mark.skip(
    reason="since torchao nightly is only compatible with torch nightly"
    "currently https://github.com/pytorch/ao/issues/2919, we'll have to skip "
    "torchao tests that requires newer versions (0.14.0.dev+) for now"
)
def test_opt_125m_awq_int4wo_model_loading_with_params(vllm_runner):
    torch._dynamo.reset()
    model_name = "torchao-testing/opt-125m-AWQConfig-Int4WeightOnlyConfig-v2-0.14.0.dev"
    with vllm_runner(
        model_name=model_name,
        quantization="torchao",
        dtype="bfloat16",
        pt_load_map_location=f"{DEVICE_TYPE}:0",
    ) as llm:
        output = llm.generate_greedy(["The capital of France is"], max_tokens=4)

        assert output


@pytest.mark.skipif(not TORCHAO_AVAILABLE, reason="torchao is not available")
def test_online_quant_config_dict_json(vllm_runner, enable_pickle):
    """Testing online quantization, load_weights integration point,
    with config dict serialized to json string
    """
    torch._dynamo.reset()
    model_name = "facebook/opt-125m"

    import json

    from torchao.core.config import config_to_dict
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow

    torchao_quant_config = Float8DynamicActivationFloat8WeightConfig(
        granularity=PerRow()
    )
    hf_overrides = {
        "quantization_config_dict_json": json.dumps(
            config_to_dict(torchao_quant_config)
        )
    }
    with vllm_runner(
        model_name=model_name,
        dtype="bfloat16",
        pt_load_map_location=f"{DEVICE_TYPE}:0",
        quantization="torchao",
        hf_overrides=hf_overrides,
        enforce_eager=True,
    ) as llm:
        output = llm.generate_greedy(["The capital of France is"], max_tokens=4)

        load_config = llm.llm.llm_engine.vllm_config.load_config
        model_config = llm.llm.llm_engine.vllm_config.model_config

        def load_weights(model):
            model_loader = get_model_loader(load_config)
            weights_iterator = model_loader.get_all_weights(model_config, model)
            model.load_weights(weights_iterator)

        llm.apply_model(load_weights)

        reload_output = llm.generate_greedy(["The capital of France is"], max_tokens=4)
        assert output[0][0] == reload_output[0][0]


@pytest.mark.skipif(not TORCHAO_AVAILABLE, reason="torchao is not available")
def test_online_quant_config_file(vllm_runner):
    """Testing on the fly quantization, load_weights integration point,
    with config file
    """
    torch._dynamo.reset()
    model_name = "facebook/opt-125m"
    import json
    from tempfile import NamedTemporaryFile

    from torchao.core.config import config_to_dict
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow

    config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())

    with NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(json.dumps(config_to_dict(config)))
        # close the file to save it
        f.close()
        config_file_name = str(f.name)

        hf_overrides = {"quantization_config_file": config_file_name}
        with vllm_runner(
            model_name=model_name,
            dtype="bfloat16",
            pt_load_map_location=f"{DEVICE_TYPE}:0",
            quantization="torchao",
            hf_overrides=hf_overrides,
            enforce_eager=True,
        ) as llm:
            output = llm.generate_greedy(["The capital of France is"], max_tokens=4)

            assert output


@pytest.mark.skipif(not TORCHAO_AVAILABLE, reason="torchao is not available")
def test_reload_weights():
    import json

    from torchao.core.config import config_to_dict
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow

    from vllm import LLM, SamplingParams

    torchao_quant_config = Float8DynamicActivationFloat8WeightConfig(
        granularity=PerRow()
    )

    hf_overrides = {
        "quantization_config_dict_json": json.dumps(
            config_to_dict(torchao_quant_config)
        )
    }

    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        dtype="bfloat16",
        load_format="dummy",
        enforce_eager=True,
        quantization="torchao",
        hf_overrides=hf_overrides,
    )
    # Update load format from `dummy` to `auto`
    llm.collective_rpc(
        "update_config", args=({"load_config": {"load_format": "auto"}},)
    )
    # Now reload real weights inplace
    llm.collective_rpc("reload_weights")
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0, top_p=0.95)
    outputs = llm.generate(prompts, sampling_params)
    # make sure it runs
    for output in outputs:
        generated_text = output.outputs[0].text
        assert generated_text
        # can also uncomment locally to make sure the generated
        # output makes sense
        # prompt = output.prompt
        # print(f"Prompt:    {prompt!r}")
        # print(f"Output:    {generated_text!r}")
        # print("-" * 60)


@pytest.mark.skipif(not TORCHAO_AVAILABLE, reason="torchao is not available")
@pytest.mark.skip(
    reason="since torchao nightly is only compatible with torch nightly"
    "currently https://github.com/pytorch/ao/issues/2919, we'll have to skip "
    "torchao tests that requires newer versions (0.15.0.dev+) for now"
)
def test_safetensors_model_loading_with_params(vllm_runner):
    torch._dynamo.reset()
    # using this model to test safetensors loading with file sharding
    model_name = "torchao-testing/Qwen3-8B-INT4-0.15.0dev-safetensors"
    with vllm_runner(model_name=model_name, dtype="bfloat16") as llm:
        output = llm.generate_greedy(["The capital of France is"], max_tokens=4)

        assert output


@pytest.mark.skipif(not TORCHAO_AVAILABLE, reason="torchao is not available")
@pytest.mark.skip(
    reason="since torchao nightly is only compatible with torch nightly"
    "currently https://github.com/pytorch/ao/issues/2919, we'll have to skip "
    "torchao tests that requires newer versions (0.14.0.dev+) for now"
)
def test_opt_125m_module_fqn_to_config_regex_model(vllm_runner):
    torch._dynamo.reset()
    model_name = "torchao-testing/opt-125m-ModuleFqnToConfig-v1-regex-0.14.0.dev"
    with vllm_runner(
        model_name=model_name, dtype="bfloat16", pt_load_map_location=f"{DEVICE_TYPE}:0"
    ) as llm:
        output = llm.generate_greedy(["The capital of France is"], max_tokens=4)

    assert output


@pytest.mark.skipif(not TORCHAO_AVAILABLE, reason="torchao is not available")
@pytest.mark.skip(
    reason="since torchao nightly is only compatible with torch nightly"
    "currently https://github.com/pytorch/ao/issues/2919, we'll have to skip "
    "torchao tests that requires newer versions (0.14.0.dev+) for now"
)
def test_opt_125m_int4wo_model_running_preshuffled_kernel(vllm_runner, monkeypatch):
    """We load a model with Int4Tensor (plain format) linear weights
    and verify that the weight is updated to Int4PreshuffledTensor
    after loading in vllm
    """
    from torchao.quantization import Int4PreshuffledTensor
    from torchao.utils import _is_fbgemm_gpu_genai_available, is_sm_at_least_90

    torch._dynamo.reset()
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    model_name = "torchao-testing/opt-125m-Int4WeightOnlyConfig-v2-0.14.0.dev"
    # Note: using enforce_eager=True because the `bf16i4bf16_shuffled` doesn't
    # have meta kernel implemented yet, can remove this flag after that is implemented
    with vllm_runner(
        model_name=model_name,
        quantization="torchao",
        dtype="bfloat16",
        pt_load_map_location=f"{DEVICE_TYPE}:0",
        enforce_eager=True,
    ) as llm:

        def has_int4_preshuffled_tensor_weight(model):
            return isinstance(
                model.model.decoder.layers[0].self_attn.qkv_proj.weight,
                Int4PreshuffledTensor,
            )

        def get_weight_attrs(model):
            weight = model.model.decoder.layers[0].self_attn.qkv_proj.weight
            return [
                weight.requires_grad,
                weight.input_dim,
                weight.output_dim,
                hasattr(weight, "weight_loader"),
            ]

        llm_engine = llm.get_llm().llm_engine
        has_int4_preshuffled_tensor = any(
            llm_engine.apply_model(has_int4_preshuffled_tensor_weight)
        )
        weight_attrs = llm_engine.apply_model(get_weight_attrs)[0]

        # making sure we are using Int4PreshuffledTensor on H100 GPU, when
        # fbgemm_gpu_genai
        # library is installed, otherwise it should be using Int4Tensor
        if _is_fbgemm_gpu_genai_available() and is_sm_at_least_90():
            assert has_int4_preshuffled_tensor
        else:
            assert not has_int4_preshuffled_tensor

        assert weight_attrs == [False, 1, 0, True]
        output = llm.generate_greedy(["The capital of France is"], max_tokens=32)

        assert output


@pytest.mark.skipif(not TORCHAO_AVAILABLE, reason="torchao is not available")
@pytest.mark.skip(
    reason="since torchao nightly is only compatible with torch nightly"
    "currently https://github.com/pytorch/ao/issues/2919, we'll have to skip "
    "torchao tests that requires newer versions (0.14.0.dev+) for now"
)
def test_opt_125m_int4wo_model_running_preshuffled_kernel_online_quant(
    vllm_runner, monkeypatch
):
    """We load a bf16 model and online quantize the model to int4, then verify that
    the weights are updated to Int4PreshuffledTensor after online quantization
    """
    from torchao.quantization import Int4PreshuffledTensor
    from torchao.utils import _is_fbgemm_gpu_genai_available, is_sm_at_least_90

    torch._dynamo.reset()
    model_name = "facebook/opt-125m"

    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    import json

    from torchao.core.config import config_to_dict
    from torchao.quantization import Int4WeightOnlyConfig

    torchao_quant_config = Int4WeightOnlyConfig(
        group_size=128, int4_packing_format="plain"
    )
    hf_overrides = {
        "quantization_config_dict_json": json.dumps(
            config_to_dict(torchao_quant_config)
        )
    }

    # Note: using enforce_eager=True because the `bf16i4bf16_shuffled` doesn't
    # have meta kernel implemented yet, can remove this flag after that is implemented
    with vllm_runner(
        model_name=model_name,
        quantization="torchao",
        dtype="bfloat16",
        pt_load_map_location=f"{DEVICE_TYPE}:0",
        hf_overrides=hf_overrides,
        enforce_eager=True,
    ) as llm:

        def has_int4_preshuffled_tensor_weight(model):
            return isinstance(
                model.model.decoder.layers[0].self_attn.qkv_proj.weight,
                Int4PreshuffledTensor,
            )

        def get_weight_attrs(model):
            weight = model.model.decoder.layers[0].self_attn.qkv_proj.weight
            return [
                weight.requires_grad,
                weight.input_dim,
                weight.output_dim,
                hasattr(weight, "weight_loader"),
            ]

        llm_engine = llm.get_llm().llm_engine
        has_int4_preshuffled_tensor = any(
            llm_engine.apply_model(has_int4_preshuffled_tensor_weight)
        )
        weight_attrs = llm_engine.apply_model(get_weight_attrs)[0]

        # making sure we are using Int4PreshuffledTensor on H100 GPU, when
        # fbgemm_gpu_genai
        # library is installed, otherwise it should be using Int4Tensor
        if _is_fbgemm_gpu_genai_available() and is_sm_at_least_90():
            assert has_int4_preshuffled_tensor
        else:
            assert not has_int4_preshuffled_tensor

        assert weight_attrs == [False, 1, 0, True]
        output = llm.generate_greedy(["The capital of France is"], max_tokens=4)

        assert output


# Zen CPU dispatch unit tests.


def _make_linear_method():
    """Build a ``TorchAOLinearMethod`` without invoking the full config pipeline."""
    from vllm.model_executor.layers.quantization.torchao import (
        TorchAOConfig,
        TorchAOLinearMethod,
    )

    # ``torchao_config`` is only consumed on the online-quantize path, which
    # these tests do not exercise. A sentinel keeps the constructor happy.
    config = TorchAOConfig.__new__(TorchAOConfig)
    config.torchao_config = object()
    config.skip_modules = []
    config.is_checkpoint_torchao_serialized = True
    return TorchAOLinearMethod(config)


def _make_int8(n: int = 8, k: int = 16, *, with_act_quant: bool = False):
    """Build a real ``Int8Tensor``; when ``with_act_quant`` is True the
    tensor carries activation-quant kwargs so it routes through the dynamic
    qlinear path."""
    from torchao.quantization import PerRow
    from torchao.quantization.quantize_.workflows import Int8Tensor
    from torchao.quantization.quantize_.workflows.int8.int8_tensor import (
        QuantizeTensorToInt8Kwargs,
    )

    qdata = torch.zeros((n, k), dtype=torch.int8)
    # (N, 1) scale exercises the squeeze branch in process_weights_after_loading.
    scale = torch.zeros((n, 1), dtype=torch.bfloat16)
    act_quant_kwargs = (
        QuantizeTensorToInt8Kwargs(granularity=PerRow()) if with_act_quant else None
    )
    return Int8Tensor(
        qdata=qdata,
        scale=scale,
        block_size=[1, k],
        dtype=torch.bfloat16,
        act_quant_kwargs=act_quant_kwargs,
    )


def _wrap_as_layer(weight_tensor: torch.Tensor):
    """Wrap a tensor in a minimal ``nn.Module`` with a registered weight."""

    layer = torch.nn.Module()
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(weight_tensor, requires_grad=False),
    )
    return layer


# ----- process_weights_after_loading: success paths ------------------------


@pytest.mark.skipif(
    not TORCHAO_VERSION_AT_LEAST_0_17_0, reason="torchao is not available"
)
def test_process_weights_after_loading_int8_dynamic_caches_dynamic_attrs(
    monkeypatch,
    zentorch_ops_mock,  # noqa: F811
):
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)

    n, k = 8, 16
    layer = _wrap_as_layer(_make_int8(n=n, k=k, with_act_quant=True))

    _make_linear_method().process_weights_after_loading(layer)

    assert hasattr(layer, "_zentorch_dynamic_qlinear_weight")
    assert hasattr(layer, "_zentorch_dynamic_qlinear_scales")
    # (N, 1) scale was squeezed to (N,).
    assert layer._zentorch_dynamic_qlinear_scales.shape == (n,)
    assert layer.weight.numel() == 0


# ----- process_weights_after_loading: failure paths preserve weight --------


def test_process_weights_after_loading_not_zen_cpu_preserves_weight(monkeypatch):
    """On non-zen platforms ``apply`` should fall back to ``F.linear``."""
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: False)

    n, k = 8, 16
    original = torch.randn(n, k)
    layer = _wrap_as_layer(original.clone())

    _make_linear_method().process_weights_after_loading(layer)

    assert layer.weight.shape == (n, k)
    assert layer.weight.numel() == n * k
    torch.testing.assert_close(layer.weight.data, original)
    assert not hasattr(layer, "_zentorch_dynamic_qlinear_weight")


def test_process_weights_after_loading_non_matching_weight_preserves_weight(
    monkeypatch,
    zentorch_ops_mock,  # noqa: F811
):
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)

    n, k = 8, 16
    original = torch.randn(n, k)
    layer = _wrap_as_layer(original.clone())

    _make_linear_method().process_weights_after_loading(layer)

    assert layer.weight.numel() == n * k
    torch.testing.assert_close(layer.weight.data, original)
    assert not hasattr(layer, "_zentorch_dynamic_qlinear_weight")


@pytest.mark.skipif(
    not TORCHAO_VERSION_AT_LEAST_0_17_0, reason="torchao is not available"
)
def test_process_weights_after_loading_int8_without_act_kwargs_preserves_weight(
    monkeypatch,
    zentorch_ops_mock,  # noqa: F811
):
    """Weight-only ``Int8Tensor`` (no ``act_quant_kwargs``)."""
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)

    layer = _wrap_as_layer(_make_int8(with_act_quant=False))

    _make_linear_method().process_weights_after_loading(layer)

    assert layer.weight.numel() > 0
    assert not hasattr(layer, "_zentorch_dynamic_qlinear_weight")


# ----- apply(): zentorch dispatch ------------------------------------------


def test_apply_dispatches_to_zentorch_dynamic_qlinear(
    monkeypatch,
    zentorch_ops_mock,  # noqa: F811
):
    from types import SimpleNamespace

    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)

    batch, k, n = 4, 16, 8
    layer = SimpleNamespace(weight=torch.nn.Parameter(torch.empty(0)))
    layer._zentorch_dynamic_qlinear_weight = torch.zeros((n, k), dtype=torch.int8)
    layer._zentorch_dynamic_qlinear_scales = torch.randn(n)

    captured: dict = {}

    def spy(
        inp,
        weight,
        weight_scales,
        bias=None,
        zentorch_op_name="zentorch::zentorch_dynamic_qlinear",
    ):
        captured["called"] = True
        captured["op_name"] = zentorch_op_name
        return torch.zeros(inp.shape[:-1] + (weight.shape[0],), dtype=inp.dtype)

    monkeypatch.setattr(torch.ops.zentorch, "zentorch_dynamic_qlinear", spy)

    x = torch.randn(batch, k)
    out = _make_linear_method().apply(layer, x)

    assert captured.get("called") is True
    assert captured["op_name"] == "zentorch::zentorch_dynamic_qlinear"
    assert out.shape == (batch, n)


def test_apply_falls_back_when_no_cached_zen_attrs(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)

    k, n = 16, 8
    weight = torch.randn(n, k)
    layer = SimpleNamespace(weight=torch.nn.Parameter(weight))

    x = torch.randn(4, k)
    out = _make_linear_method().apply(layer, x)

    torch.testing.assert_close(out, torch.nn.functional.linear(x, weight))


if __name__ == "__main__":
    pytest.main([__file__])
