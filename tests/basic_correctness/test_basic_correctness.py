# SPDX-License-Identifier: Apache-2.0
"""Compare the short outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/basic_correctness/test_basic_correctness.py`.
"""
import os
import weakref

import pytest

from vllm import LLM
from vllm.platforms import current_platform

from ..conftest import VllmRunner
from ..models.utils import check_outputs_equal
from ..utils import multi_gpu_test

MODELS = [
    "google/gemma-2-2b-it",
    "meta-llama/Llama-3.2-1B-Instruct",
]

TARGET_TEST_SUITE = os.environ.get("TARGET_TEST_SUITE", "L4")


@pytest.fixture(autouse=True)
def v1(run_with_both_engines):
    # Simple autouse wrapper to run both engines for each test
    # This can be promoted up to conftest.py to run for every
    # test in a package
    pass


def test_vllm_gc_ed():
    """Verify vllm instance is GC'ed when it is deleted"""
    llm = LLM("distilbert/distilgpt2")
    weak_llm = weakref.ref(llm)
    del llm
    # If there's any circular reference to vllm, this fails
    # because llm instance is not GC'ed.
    assert weak_llm() is None


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("backend", ["FLASH_ATTN"])
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [5])
@pytest.mark.parametrize("enforce_eager", [False])
def test_models(
    hf_runner,
    model: str,
    backend: str,
    dtype: str,
    max_tokens: int,
    enforce_eager: bool,
) -> None:

    if backend == "FLASHINFER" and current_platform.is_rocm():
        pytest.skip("Flashinfer does not support ROCm/HIP.")

    if backend in ("XFORMERS",
                   "FLASHINFER") and model == "google/gemma-2-2b-it":
        pytest.skip(
            f"{backend} does not support gemma2 with full context length.")

    os.environ["VLLM_ATTENTION_BACKEND"] = backend

    # 5042 tokens for gemma2
    # gemma2 has alternating sliding window size of 4096
    # we need a prompt with more than 4096 tokens to test the sliding window
    prompt = "The following numbers of the sequence " + ", ".join(
        str(i) for i in range(1024)) + " are:"
    example_prompts = [prompt]

    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

    with VllmRunner(model,
                    max_model_len=8192,
                    dtype=dtype,
                    enforce_eager=enforce_eager,
                    gpu_memory_utilization=0.7) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "model, distributed_executor_backend, attention_backend, "
    "test_suite", [
        ("distilbert/distilgpt2", "ray", "", "L4"),
        ("distilbert/distilgpt2", "mp", "", "L4"),
        ("meta-llama/Llama-3.2-1B-Instruct", "ray", "", "L4"),
        ("meta-llama/Llama-3.2-1B-Instruct", "mp", "", "L4"),
        ("distilbert/distilgpt2", "ray", "", "A100"),
        ("distilbert/distilgpt2", "mp", "", "A100"),
        ("distilbert/distilgpt2", "mp", "FLASHINFER", "A100"),
        ("meta-llama/Meta-Llama-3-8B", "ray", "FLASHINFER", "A100"),
    ])
def test_models_distributed(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    distributed_executor_backend: str,
    attention_backend: str,
    test_suite: str,
) -> None:

    if test_suite != TARGET_TEST_SUITE:
        pytest.skip(f"Skip test for {test_suite}")

    if model == "meta-llama/Llama-3.2-1B-Instruct" and distributed_executor_backend == "ray" and attention_backend == "" and test_suite == "L4":  # noqa
        # test Ray Compiled Graph
        os.environ['VLLM_USE_RAY_SPMD_WORKER'] = "1"
        os.environ['VLLM_USE_RAY_COMPILED_DAG'] = "1"

    if attention_backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = attention_backend

    dtype = "half"
    max_tokens = 5

    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).
    with vllm_runner(model,
                     dtype=dtype,
                     tensor_parallel_size=2,
                     distributed_executor_backend=distributed_executor_backend
                     ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)

    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
