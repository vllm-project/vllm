# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import SamplingParams
from vllm.config import LoadConfig, LoadFormat
from vllm.model_executor.model_loader import get_model_loader

test_model = "openai-community/gpt2"

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, seed=0)


def get_runai_model_loader():
    load_config = LoadConfig(load_format=LoadFormat.RUNAI_STREAMER)
    return get_model_loader(load_config)


def test_get_model_loader_with_runai_flag():
    model_loader = get_runai_model_loader()
    assert model_loader.__class__.__name__ == "RunaiModelStreamerLoader"


def test_runai_model_loader_download_files(vllm_runner):
    with vllm_runner(test_model, load_format=LoadFormat.RUNAI_STREAMER) as llm:
        deserialized_outputs = llm.generate(prompts, sampling_params)
        assert deserialized_outputs
