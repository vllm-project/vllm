'''Tests whether bitsandbytes computation is enabled correctly.

Run `pytest tests/quantization/test_bitsandbytes.py`.
'''
import pytest
import torch

from tests.conftest import VllmRunner
from tests.quantization.utils import is_quant_method_supported
from vllm import SamplingParams

models_4bit_to_test = [
    ('huggyllama/llama-7b', 'quantize model inflight'),
    ('lllyasviel/omost-llama-3-8b-4bits',
     'read pre-quantized 4-bit NF4 model'),
    ('PrunaAI/Einstein-v6.1-Llama3-8B-bnb-4bit-smashed',
     'read pre-quantized 4-bit FP4 model'),
]

models_8bit_to_test = [
    ('meta-llama/Llama-Guard-3-8B-INT8', 'read pre-quantized 8-bit model'),
]


@pytest.mark.skipif(not is_quant_method_supported("bitsandbytes"),
                    reason='bitsandbytes is not supported on this GPU type.')
@pytest.mark.parametrize("model_name, description", models_4bit_to_test)
def test_load_4bit_bnb_model(vllm_runner, model_name, description) -> None:
    with vllm_runner(model_name,
                     quantization='bitsandbytes',
                     load_format='bitsandbytes',
                     enforce_eager=True) as llm:
        model = llm.model.llm_engine.model_executor.driver_worker.model_runner.model  # noqa: E501

        # check the weights in MLP & SelfAttention are quantized to torch.uint8
        validate_model_weight_type(model, torch.uint8)

        validate_model_output(llm)


@pytest.mark.skipif(not is_quant_method_supported("bitsandbytes"),
                    reason='bitsandbytes is not supported on this GPU type.')
@pytest.mark.parametrize("model_name, description", models_8bit_to_test)
def test_load_8bit_bnb_model(vllm_runner, model_name, description) -> None:
    with vllm_runner(model_name,
                     quantization='bitsandbytes',
                     load_format='bitsandbytes',
                     enforce_eager=True) as llm:
        model = llm.model.llm_engine.model_executor.driver_worker.model_runner.model  # noqa: E501

        # check the weights in MLP & SelfAttention are quantized to torch.int8
        validate_model_weight_type(model, torch.int8)

        validate_model_output(llm)


def validate_model_weight_type(model, quantized_dtype=torch.uint8):
    # Check quantized weights
    quantized_layers = [('mlp.gate_up_proj.qweight',
                         model.model.layers[0].mlp.gate_up_proj.qweight),
                        ('mlp.down_proj.qweight',
                         model.model.layers[0].mlp.down_proj.qweight),
                        ('self_attn.o_proj.qweight',
                         model.model.layers[0].self_attn.o_proj.qweight),
                        ('self_attn.qkv_proj.qweight',
                         model.model.layers[0].self_attn.qkv_proj.qweight)]

    for name, qweight in quantized_layers:
        assert qweight.dtype == quantized_dtype, (
            f'Expected {name} dtype {quantized_dtype} but got {qweight.dtype}')

    # Check non-quantized weights
    non_quantized_layers = [
        ('lm_head.weight', model.lm_head.weight),
        ('embed_tokens.weight', model.model.embed_tokens.weight),
        ('input_layernorm.weight',
         model.model.layers[0].input_layernorm.weight),
        ('post_attention_layernorm.weight',
         model.model.layers[0].post_attention_layernorm.weight)
    ]

    for name, weight in non_quantized_layers:
        assert weight.dtype != quantized_dtype, (
            f'{name} dtype should not be {quantized_dtype}')


def validate_model_output(llm: VllmRunner):
    sampling_params = SamplingParams(temperature=0.0,
                                     logprobs=1,
                                     prompt_logprobs=1,
                                     max_tokens=8)

    prompts = ['That which does not kill us', 'To be or not to be,']
    expected_outputs = [
        'That which does not kill us makes us stronger.',
        'To be or not to be, that is the question.'
    ]
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    assert len(outputs) == len(prompts)

    for index in range(len(outputs)):
        # compare the first line of the output
        actual_output = outputs[index][1][0].split('\n', 1)[0]
        expected_output = expected_outputs[index].split('\n', 1)[0]

        assert len(actual_output) >= len(expected_output), (
            f'Actual {actual_output} should be larger than or equal to '
            f'expected {expected_output}')
        actual_output = actual_output[:len(expected_output)]

        assert actual_output == expected_output, (
            f'Expected: {expected_output}, but got: {actual_output}')
