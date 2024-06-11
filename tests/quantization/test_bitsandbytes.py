'''Tests whether bitsandbytes computation is enabled correctly.

Run `pytest tests/quantization/test_bitsandbytes.py`.
'''
import pytest
import torch

from vllm import SamplingParams
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

capability = torch.cuda.get_device_capability()
capability = capability[0] * 10 + capability[1]


@pytest.mark.skipif(
    capability < QUANTIZATION_METHODS['bitsandbytes'].get_min_capability(),
    reason='bitsandbytes is not supported on this GPU type.')
def test_load_bnb_model(vllm_runner) -> None:
    with vllm_runner('huggyllama/llama-7b',
                     quantization='bitsandbytes',
                     load_format='bitsandbytes',
                     enforce_eager=True) as llm:

        model = llm.model.llm_engine.model_executor.driver_worker.model_runner.model  # noqa: E501

        # check the weights in MLP & SelfAttention are quantized to torch.uint8
        qweight = model.model.layers[0].mlp.gate_up_proj.qweight
        assert qweight.dtype == torch.uint8, (
            f'Expected gate_up_proj dtype torch.uint8 but got {qweight.dtype}')

        qweight = model.model.layers[0].mlp.down_proj.qweight
        assert qweight.dtype == torch.uint8, (
            f'Expected down_proj dtype torch.uint8 but got {qweight.dtype}')

        qweight = model.model.layers[0].self_attn.o_proj.qweight
        assert qweight.dtype == torch.uint8, (
            f'Expected o_proj dtype torch.uint8 but got {qweight.dtype}')

        qweight = model.model.layers[0].self_attn.qkv_proj.qweight
        assert qweight.dtype == torch.uint8, (
            f'Expected qkv_proj dtype torch.uint8 but got {qweight.dtype}')

        # some weights should not be quantized
        weight = model.lm_head.weight
        assert weight.dtype != torch.uint8, (
            'lm_head weight dtype should not be torch.uint8')

        weight = model.model.embed_tokens.weight
        assert weight.dtype != torch.uint8, (
            'embed_tokens weight dtype should not be torch.uint8')

        weight = model.model.layers[0].input_layernorm.weight
        assert weight.dtype != torch.uint8, (
            'input_layernorm weight dtype should not be torch.uint8')

        weight = model.model.layers[0].post_attention_layernorm.weight
        assert weight.dtype != torch.uint8, (
            'input_layernorm weight dtype should not be torch.uint8')

        # check the output of the model is expected
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
            assert actual_output == expected_output, (
                f'Expected: {expected_output}, but got: {actual_output}')
