# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref

import pytest
# downloading lora to test lora requests
from huggingface_hub import snapshot_download

from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.lora.request import LoRARequest

MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

LORA_NAME = "typeof/zephyr-7b-beta-lora"


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()


@pytest.fixture(scope="module", params=[False, True])
def llm(request, monkeypatch_module):

    use_v1 = request.param
    monkeypatch_module.setenv('VLLM_USE_V1', '1' if use_v1 else '0')

    # pytest caches the fixture so we use weakref.proxy to
    # enable garbage collection
    llm = LLM(model=MODEL_NAME,
              tensor_parallel_size=1,
              max_model_len=8192,
              enable_lora=True,
              max_loras=4,
              max_lora_rank=64,
              max_num_seqs=128,
              enforce_eager=True)

    with llm.deprecate_legacy_api():
        yield weakref.proxy(llm)

        del llm

    cleanup_dist_env_and_memory()


@pytest.fixture(scope="module")
def zephyr_lora_files():
    return snapshot_download(repo_id=LORA_NAME)


@pytest.mark.skip_global_cleanup
def test_multiple_lora_requests(llm: LLM, zephyr_lora_files):
    lora_request = [
        LoRARequest(LORA_NAME + str(idx), idx + 1, zephyr_lora_files)
        for idx in range(len(PROMPTS))
    ]
    # Multiple SamplingParams should be matched with each prompt
    outputs = llm.generate(PROMPTS, lora_request=lora_request)
    assert len(PROMPTS) == len(outputs)

    # Exception raised, if the size of params does not match the size of prompts
    with pytest.raises(ValueError):
        outputs = llm.generate(PROMPTS, lora_request=lora_request[:1])

    # Single LoRARequest should be applied to every prompt
    single_lora_request = lora_request[0]
    outputs = llm.generate(PROMPTS, lora_request=single_lora_request)
    assert len(PROMPTS) == len(outputs)
