# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.metadata
import importlib.util

import pytest
import torch
from packaging import version

from tests.quantization.utils import load_model_without_vllm_runner
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.platforms import current_platform

if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx950
else:

    def on_gfx950() -> bool:
        return False


DEVICE_TYPE = current_platform.device_type
DTYPE = ["bfloat16"]

TORCHAO_AVAILABLE = importlib.util.find_spec("torchao") is not None
TORCHAO_VERSION_0_18_AVAILABLE = TORCHAO_AVAILABLE and version.parse(
    importlib.metadata.version("torchao")
) >= version.parse("0.18.0")


# Fix for leaking `torch.set_float32_matmul_precision("high")`
# https://github.com/pytorch/ao/blob/v0.17.0/torchao/quantization/utils.py#L696
@pytest.fixture(autouse=True)
def restore_float32_matmul_precision():
    original_precision = torch.get_float32_matmul_precision()
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(original_precision)


@pytest.mark.skipif(
    current_platform.is_rocm() and current_platform.is_fp8_fnuz(),
    reason="Only fp8_fnuz supported on CDNA3 architecture",
)
@pytest.mark.skipif(not TORCHAO_AVAILABLE, reason="torchao is not available")
def test_pre_quantized_model(monkeypatch, dist_init, workspace_init):
    model, vllm_config = load_model_without_vllm_runner(
        "torchao-testing/opt-125m-Float8WeightOnlyConfig-v2-0.15.0",
        quantization="torchao",
        dtype="bfloat16",
    )

    monkeypatch.setattr(Attention, "forward", lambda _, q, k, v: q.contiguous())
    input_ids = torch.tensor([1, 2, 3, 4], device=DEVICE_TYPE)
    positions = torch.arange(input_ids.numel(), device=DEVICE_TYPE)
    with (
        set_current_vllm_config(vllm_config),
        set_forward_context(None, vllm_config, num_tokens=input_ids.numel()),
    ):
        hidden_states = model(input_ids, positions, None)
        logits = model.compute_logits(hidden_states)
    assert torch.isfinite(logits).all()


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
@pytest.mark.skipif(
    on_gfx950() and not TORCHAO_VERSION_0_18_AVAILABLE,
    reason="requires torchao>=0.18.0 on gfx950",
)
def test_online_quant_config_dict_json(vllm_runner, enable_pickle):
    """Testing online quantization and reload_weights, with config dict
    serialized to json string
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

        llm.collective_rpc("reload_weights")

        reload_output = llm.generate_greedy(["The capital of France is"], max_tokens=4)
        assert output[0][0] == reload_output[0][0]


@pytest.mark.skipif(not TORCHAO_AVAILABLE, reason="torchao is not available")
@pytest.mark.skipif(
    on_gfx950() and not TORCHAO_VERSION_0_18_AVAILABLE,
    reason="requires torchao>=0.18.0 on gfx950",
)
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
@pytest.mark.skipif(
    on_gfx950() and not TORCHAO_VERSION_0_18_AVAILABLE,
    reason="requires torchao>=0.18.0 on gfx950",
)
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


def _save_awq_checkpoint(model_name: str, out_dir) -> str:
    """Quantize `model_name` with torchao AWQ on the tinygemm int4 layout and save
    it the way torchao safetensors checkpoints are stored."""
    from safetensors.torch import save_file
    from torchao.prototype.awq import AWQConfig
    from torchao.prototype.safetensors.safetensors_support import (
        flatten_tensor_state_dict,
    )
    from torchao.quantization import Int4WeightOnlyConfig, quantize_
    from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map=DEVICE_TYPE
    )
    base_config = Int4WeightOnlyConfig(
        group_size=128, int4_packing_format="tile_packed_to_4d"
    )

    def is_decoder_linear(module, fqn):
        return isinstance(module, torch.nn.Linear) and "lm_head" not in fqn

    quantize_(model, AWQConfig(base_config, step="prepare"), is_decoder_linear)
    calibration = "The capital of France is Paris. def add(a, b): return a + b. " * 8
    with torch.no_grad():
        model(**tokenizer(calibration, return_tensors="pt").to(DEVICE_TYPE))
    quantize_(model, AWQConfig(base_config, step="convert"), is_decoder_linear)

    state_dict = model.state_dict()
    if model.config.tie_word_embeddings:
        state_dict.pop("lm_head.weight")
    tensors, metadata = flatten_tensor_state_dict(state_dict)
    tensors = {name: t.detach().clone().contiguous() for name, t in tensors.items()}
    save_file(tensors, f"{out_dir}/model.safetensors", metadata=metadata)
    model.config.quantization_config = TorchAoConfig(
        AWQConfig(base_config, step="prepare_for_loading")
    ).to_dict()
    model.config.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return str(out_dir)


@pytest.mark.skipif(not TORCHAO_AVAILABLE, reason="torchao is not available")
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="tinygemm int4 kernel is CUDA only"
)
def test_awq_fused_layers_keep_each_shard_act_pre_scale(
    tmp_path, hf_runner, vllm_runner, example_prompts, enable_pickle
):
    """AWQ gives q/k/v_proj and gate/up_proj their own act_pre_scale. vLLM fuses
    them into qkv_proj and gate_up_proj, and used to apply the last shard's
    scale to every shard."""
    from tests.models.utils import check_logprobs_close

    model_dir = _save_awq_checkpoint("Qwen/Qwen3-0.6B", tmp_path)
    max_tokens, num_logprobs = 16, 5
    with hf_runner(
        model_dir, dtype="bfloat16", model_kwargs={"device_map": DEVICE_TYPE}
    ) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs
        )
    torch._dynamo.reset()
    with vllm_runner(
        model_dir, quantization="torchao", dtype="bfloat16", max_model_len=1024
    ) as llm:

        def max_shards(model):
            return max(len(getattr(m, "torchao_shards", ())) for m in model.modules())

        assert llm.apply_model(max_shards)[0] > 1
        vllm_outputs = llm.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs
        )
    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


if __name__ == "__main__":
    pytest.main([__file__])
