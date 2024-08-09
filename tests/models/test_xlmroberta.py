from typing import List, Optional, Tuple, Type, Union

import pytest
import torch
from transformers import AutoTokenizer

from ..conftest import HfRunner, VllmRunner

models = ["BAAI/bge-reranker-base"]


def run_test(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    model: str,
    *,
    dtype: str,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    """Inference result should be the same between hf and vllm."""

    prompt = "this is a useless prompt."
    sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]] = \
        [("hello world", "nice to meet you"), ("head north", "head south")]
    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=None)
    inputs = tokenizer(
        sentence_pairs,
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=512,
    ).to("cuda")

    with vllm_runner(model,
                     dtype=dtype,
                     max_model_len=512,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True) as vllm_model:
        vllm_outputs = vllm_model.process([{
            "prompt": prompt,
            "multi_modal_data": {
                "xlmroberta": inputs,
            }
        }])

    with hf_runner(model, dtype=dtype, is_simple_model=True) as hf_model:
        hf_outputs = hf_model.process(**inputs)

    print(vllm_outputs[0].outputs.result, hf_outputs.logits.view(-1, ))
    assert torch.allclose(vllm_outputs[0].outputs.result,
                          hf_outputs.logits.view(-1, ))


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("dtype", ["float"])
def test_models(hf_runner, vllm_runner, model, dtype: str) -> None:
    run_test(
        hf_runner,
        vllm_runner,
        model,
        dtype=dtype,
        tensor_parallel_size=1,
    )
