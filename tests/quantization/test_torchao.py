# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.metadata
import importlib.util

import pytest
import torch

DTYPE = ["bfloat16"]

TORCHAO_AVAILABLE = importlib.util.find_spec("torchao") is not None


@pytest.mark.skipif(not TORCHAO_AVAILABLE, reason="torchao is not available")
def test_pre_quantized_model(vllm_runner):
    with vllm_runner(
        "drisspg/fp8-opt-125m",
        quantization="torchao",
        dtype="bfloat16",
        enforce_eager=True,
    ) as llm:
        output = llm.generate_greedy(["The capital of France is"], max_tokens=32)
    assert output


@pytest.mark.skipif(not TORCHAO_AVAILABLE, reason="torchao is not available")
@pytest.mark.parametrize(
    "pt_load_map_location",
    [
        "cuda:0",
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
    ) as llm:
        output = llm.generate_greedy(["The capital of France is"], max_tokens=32)

        assert output


@pytest.mark.skipif(not TORCHAO_AVAILABLE, reason="torchao is not available")
def test_opt_125m_int4wo_model_per_module_quant(vllm_runner):
    torch._dynamo.reset()
    model_name = "jerryzh168/opt-125m-int4wo-per-module"
    with vllm_runner(
        model_name=model_name,
        quantization="torchao",
        dtype="bfloat16",
        pt_load_map_location="cuda:0",
    ) as llm:
        output = llm.generate_greedy(["The capital of France is"], max_tokens=32)

        assert output


@pytest.mark.skipif(not TORCHAO_AVAILABLE, reason="torchao is not available")
def test_qwenvl_int8wo_model_loading_with_params(vllm_runner):
    torch._dynamo.reset()
    model_name = "mobicham/Qwen2.5-VL-3B-Instruct_int8wo_ao"
    with vllm_runner(
        model_name=model_name,
        quantization="torchao",
        dtype="bfloat16",
        pt_load_map_location="cuda:0",
    ) as llm:
        output = llm.generate_greedy(["The capital of France is"], max_tokens=32)

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
        pt_load_map_location="cuda:0",
    ) as llm:
        output = llm.generate_greedy(["The capital of France is"], max_tokens=32)

        assert output


@pytest.mark.skipif(not TORCHAO_AVAILABLE, reason="torchao is not available")
def test_on_the_fly_quant_config_dict_json(vllm_runner):
    """Testing on the fly quantization, load_weights integration point,
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
        pt_load_map_location="cuda:0",
        quantization="torchao",
        hf_overrides=hf_overrides,
    ) as llm:
        output = llm.generate_greedy(["The capital of France is"], max_tokens=32)

        assert output


@pytest.mark.skipif(not TORCHAO_AVAILABLE, reason="torchao is not available")
def test_on_the_fly_quant_config_file(vllm_runner):
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
            pt_load_map_location="cuda:0",
            quantization="torchao",
            hf_overrides=hf_overrides,
        ) as llm:
            output = llm.generate_greedy(["The capital of France is"], max_tokens=32)

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
    "torchao tests that requires newer versions (0.14.0.dev+) for now"
)
def test_opt_125m_float8_weight_only_safetensors_model_loading_with_params(vllm_runner):
    torch._dynamo.reset()
    model_name = (
        "torchao-testing/opt-125m-Float8WeightOnlyConfig-v2-0.14.0.dev-safetensors"
    )
    with vllm_runner(model_name=model_name, dtype="bfloat16") as llm:
        output = llm.generate_greedy(["The capital of France is"], max_tokens=32)

        assert output


if __name__ == "__main__":
    pytest.main([__file__])
