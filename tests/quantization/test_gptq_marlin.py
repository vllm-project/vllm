#import os

import pytest

from tests.quantization.utils import is_quant_method_supported
from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT
from vllm.model_executor.parameter import PackedvLLMParameter

#os.environ["TOKENIZERS_PARALLELISM"] = "true"

MAX_MODEL_LEN = 1024

MODELS = [
    # act_order==False, group_size=channelwise
    ("robertgshaw2/zephyr-7b-beta-channelwise-gptq", "main"),
    # act_order==False, group_size=128
    ("TheBloke/Llama-2-7B-GPTQ", "main"),

    # act_order==True, group_size=128
    ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", "main"),
    # act_order==True, group_size=64
    ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", "gptq-4bit-64g-actorder_True"),
    # act_order==True, group_size=32
    ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", "gptq-4bit-32g-actorder_True"),

    # 8-bit, act_order==True, group_size=channelwise
    ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", "gptq-8bit--1g-actorder_True"),
    # 8-bit, act_order==True, group_size=128
    ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", "gptq-8bit-128g-actorder_True"),
    # 8-bit, act_order==True, group_size=32
    ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", "gptq-8bit-32g-actorder_True"),

    # 4-bit, act_order==True, group_size=128
    ("TechxGenus/gemma-1.1-2b-it-GPTQ", "main")
]


@pytest.mark.skipif(not is_quant_method_supported("gptq_marlin"),
                    reason="gptq_marlin is not supported on this GPU type.")
@pytest.mark.parametrize("model", MODELS)
def test_model_weight_loading(
    vllm_runner,
    model,
    dtype: str,
) -> None:
    model_name, revision = model

    with vllm_runner(model_name=model_name,
                     revision=revision,
                     dtype="half",
                     quantization="gptq_marlin",
                     max_model_len=MAX_MODEL_LEN,
                     tensor_parallel_size=2) as gptq_marlin_model:

        model = gptq_marlin_model.model.llm_engine.model_executor.driver_worker.model_runner.model  # noqa: E501
        layer = model.model.layers[0]
        assert isinstance(layer.qweight, PackedvLLMParameter)
        output = gptq_marlin_model.generate_greedy("Hello world!",
                                                   max_tokens=20)
        assert output

    _ROPE_DICT.clear()  # clear rope cache to avoid rope dtype error
