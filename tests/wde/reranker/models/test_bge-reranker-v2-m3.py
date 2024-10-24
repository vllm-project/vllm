import random
from typing import List, TypeVar

import numpy as np
import pytest
import torch
import torch.nn as nn
from transformers import (AutoModelForSequenceClassification, BatchEncoding,
                          BatchFeature)

from tests.wde.utils import HfRunner, VllmRunner, cleanup
from vllm.wde.reranker.schema.engine_io import RerankerInputs

_T = TypeVar("_T", nn.Module, torch.Tensor, BatchEncoding, BatchFeature)


class VllmRerankerRunner(VllmRunner):

    def reranker(self, inputs: RerankerInputs) -> List[float]:
        req_outputs = self.model.reranker(inputs)
        outputs = []
        for req_output in req_outputs:
            score = req_output.score
            outputs.append(score)
        return outputs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        cleanup()


class HfRerankerRunner(HfRunner):

    def reranker(self, inputs: RerankerInputs) -> List[float]:
        encoded_input = self.tokenizer(inputs,
                                       padding=True,
                                       truncation=True,
                                       return_tensors="pt").to("cuda")

        scores = self.model(**encoded_input).logits.view(-1, )
        return scores.cpu().numpy().tolist()


@pytest.fixture(scope="session")
def hf_runner():
    return HfRerankerRunner


@pytest.fixture(scope="session")
def vllm_runner():
    return VllmRerankerRunner


@pytest.fixture(scope="session")
def example_prompts():
    pairs = [
        ["query", "passage"],
        ["what is panda?", "hi"],
        [
            "what is panda?",
            "The giant panda (Ailuropoda melanoleuca), "
            "sometimes called a panda bear or simply panda, "
            "is a bear species endemic to China.",
        ],
    ] * 11
    random.shuffle(pairs)
    return pairs


MODELS = ["BAAI/bge-reranker-v2-m3"]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_num_seqs", [2, 3, 5, 7])
@pytest.mark.parametrize("scheduling", ["sync", "async", "double_buffer"])
@torch.inference_mode
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_num_seqs: int,
    scheduling: str,
) -> None:
    with hf_runner(model,
                   dtype=dtype,
                   auto_cls=AutoModelForSequenceClassification) as hf_model:
        hf_outputs = hf_model.reranker(example_prompts)

    with vllm_runner(model, dtype=dtype,
                     max_num_seqs=max_num_seqs) as vllm_model:
        vllm_outputs = vllm_model.reranker(example_prompts)

    # Without using sigmoid,
    # the difference may be greater than 1e-2, resulting in flakey test
    hf_outputs = [sigmoid(x) for x in hf_outputs]
    vllm_outputs = [sigmoid(x) for x in vllm_outputs]

    all_similarities = np.array(hf_outputs) - np.array(vllm_outputs)

    tolerance = 1e-2
    assert np.all((all_similarities <= tolerance)
                  & (all_similarities >= -tolerance)
                  ), f"Not all values are within {tolerance} of 1.0"
