'''Tests whether bitsandbytes computation is enabled correctly.

Run `pytest tests/quantization/test_bitsandbytes.py`.
'''

import gc

import pytest
import torch
import time

from tests.quantization.utils import is_quant_method_supported

models_4bit_to_test = [
    ('huggyllama/llama-7b', 'quantize model inflight'),
]

models_pre_qaunt_4bit_to_test = [
    ('lllyasviel/omost-llama-3-8b-4bits',
     'read pre-quantized 4-bit NF4 model'),
    ('PrunaAI/Einstein-v6.1-Llama3-8B-bnb-4bit-smashed',
     'read pre-quantized 4-bit FP4 model'),
]

models_pre_quant_8bit_to_test = [
    ('meta-llama/Llama-Guard-3-8B-INT8', 'read pre-quantized 8-bit model'),
]


@pytest.mark.skipif(not is_quant_method_supported("bitsandbytes"),
                    reason='bitsandbytes is not supported on this GPU type.')
@pytest.mark.parametrize("model_name, description", models_4bit_to_test)
def test_load_4bit_bnb_model(hf_runner, vllm_runner, example_prompts,
                             model_name, description) -> None:
    
    log_gpu_memory("4bit - start")

    hf_model_kwargs = {"load_in_4bit": True}

    validate_generated_texts(hf_runner, vllm_runner, example_prompts[:1],
                             model_name, hf_model_kwargs)
    log_gpu_memory("4bit - middle 1 ")

    validate_model_weight_type(vllm_runner, model_name, torch.uint8)

    log_gpu_memory("4bit - middle 2 ")

    # Forcefully delete objects
    del hf_runner, vllm_runner, example_prompts, model_name, description
    gc.collect()
    torch.cuda.empty_cache()

    time.sleep(5)

    log_gpu_memory("4bit - finish")

    assert 1 == 0


@pytest.mark.skipif(not is_quant_method_supported("bitsandbytes"),
                    reason='bitsandbytes is not supported on this GPU type.')
@pytest.mark.parametrize("model_name, description",
                         models_pre_qaunt_4bit_to_test)
def test_load_pre_quant_4bit_bnb_model(hf_runner, vllm_runner, example_prompts,
                                       model_name, description) -> None:
    log_gpu_memory("4bit 2 - start")

    validate_generated_texts(hf_runner, vllm_runner, example_prompts[:1],
                             model_name)

    validate_model_weight_type(vllm_runner, model_name, torch.uint8)

    log_gpu_memory("4bit 2 - finish")


@pytest.mark.skipif(not is_quant_method_supported("bitsandbytes"),
                    reason='bitsandbytes is not supported on this GPU type.')
@pytest.mark.parametrize("model_name, description",
                         models_pre_quant_8bit_to_test)
def test_load_8bit_bnb_model(hf_runner, vllm_runner, example_prompts,
                             model_name, description) -> None:

    log_gpu_memory("8bit - start")
    validate_generated_texts(hf_runner, vllm_runner, example_prompts[:1],
                             model_name)

    validate_model_weight_type(vllm_runner, model_name, torch.int8)
    log_gpu_memory("8bit - end")

def log_generated_texts(prompts, outputs, runner_name):
    logged_texts = []
    for i, (_, generated_text) in enumerate(outputs):
        log_entry = {
            "prompt": prompts[i],
            "runner_name": runner_name,
            "generated_text": generated_text,
        }
        logged_texts.append(log_entry)
    return logged_texts


def validate_generated_texts(hf_runner,
                             vllm_runner,
                             prompts,
                             model_name,
                             hf_model_kwargs=None):

    if hf_model_kwargs is None:
        hf_model_kwargs = {}

    # Run with HF runner
    with hf_runner(model_name, model_kwargs=hf_model_kwargs) as llm:
        hf_outputs = llm.generate_greedy(prompts, 8)
        hf_logs = log_generated_texts(prompts, hf_outputs, "HfRunner")

    # Clean up the GPU memory for the next test
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    #Run with vLLM runner
    with vllm_runner(model_name,
                     quantization='bitsandbytes',
                     load_format='bitsandbytes',
                     enforce_eager=True,
                     gpu_memory_utilization=0.8) as llm:
        vllm_outputs = llm.generate_greedy(prompts, 8)
        vllm_logs = log_generated_texts(prompts, vllm_outputs, "VllmRunner")

    # Clean up the GPU memory for the next test
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    # Compare the generated strings
    for hf_log, vllm_log in zip(hf_logs, vllm_logs):
        hf_str = hf_log["generated_text"]
        vllm_str = vllm_log["generated_text"]
        prompt = hf_log["prompt"]
        assert hf_str == vllm_str, (f"Model: {model_name}"
                                    f"Mismatch between HF and vLLM outputs:\n"
                                    f"Prompt: {prompt}\n"
                                    f"HF Output: '{hf_str}'\n"
                                    f"vLLM Output: '{vllm_str}'")


def validate_model_weight_type(vllm_runner,
                               model_name,
                               quantized_dtype=torch.uint8):
    with vllm_runner(
            model_name,
            quantization='bitsandbytes',
            load_format='bitsandbytes',
            enforce_eager=True,
    ) as llm:
        model = llm.model.llm_engine.model_executor.driver_worker.model_runner.model  # noqa: E501

        # check the weights in MLP & SelfAttention are quantized to torch.uint8
        qweight = model.model.layers[0].mlp.gate_up_proj.qweight
        assert qweight.dtype == quantized_dtype, (
            f'Expected gate_up_proj dtype {quantized_dtype} but got {qweight.dtype}')

        qweight = model.model.layers[0].mlp.down_proj.qweight
        assert qweight.dtype == quantized_dtype, (
            f'Expected down_proj dtype {quantized_dtype} but got {qweight.dtype}')

        qweight = model.model.layers[0].self_attn.o_proj.qweight
        assert qweight.dtype == quantized_dtype, (
            f'Expected o_proj dtype {quantized_dtype} but got {qweight.dtype}')

        qweight = model.model.layers[0].self_attn.qkv_proj.qweight
        assert qweight.dtype == quantized_dtype, (
            f'Expected qkv_proj dtype {quantized_dtype} but got {qweight.dtype}')

        # some weights should not be quantized
        weight = model.lm_head.weight
        assert weight.dtype != quantized_dtype, (
            f'lm_head weight dtype should not be {quantized_dtype}')

        weight = model.model.embed_tokens.weight
        assert weight.dtype != quantized_dtype, (
            f'embed_tokens weight dtype should not be {quantized_dtype}')

        weight = model.model.layers[0].input_layernorm.weight
        assert weight.dtype != quantized_dtype, (
            f'input_layernorm weight dtype should not be {quantized_dtype}')

        weight = model.model.layers[0].post_attention_layernorm.weight
        assert weight.dtype != quantized_dtype, (
            f'input_layernorm weight dtype should not be {quantized_dtype}')

    torch.cuda.synchronize()
    del model, llm
    gc.collect()
    torch.cuda.empty_cache()


def log_gpu_memory(prefix=""):
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    
    print(f"{prefix} - Memory Allocated: {allocated} bytes")
    print(f"{prefix} - Memory Reserved: {reserved} bytes")