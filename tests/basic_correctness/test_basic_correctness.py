"""Compare the short outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/basic_correctness/test_basic_correctness.py`.
"""
import os
import pickle
import re
import weakref
from unittest.mock import patch

import pytest

from vllm import LLM
from vllm.utils import is_hip
from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

from ..models.utils import check_outputs_equal
from ..utils import multi_gpu_test

MODELS = [
    "facebook/opt-125m",
    "meta-llama/Llama-2-7b-hf",
]

TARGET_TEST_SUITE = os.environ.get("TARGET_TEST_SUITE", "L4")


def test_vllm_gc_ed():
    """Verify vllm instance is GC'ed when it is deleted"""
    llm = LLM("facebook/opt-125m")
    weak_llm = weakref.ref(llm)
    del llm
    # If there's any circular reference to vllm, this fails
    # because llm instance is not GC'ed.
    assert weak_llm() is None


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("backend", ["FLASH_ATTN", "XFORMERS", "FLASHINFER"])
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [5])
@pytest.mark.parametrize("enforce_eager", [False, True])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    backend: str,
    dtype: str,
    max_tokens: int,
    enforce_eager: bool,
) -> None:

    if backend == "FLASHINFER" and is_hip():
        pytest.skip("Flashinfer does not support ROCm/HIP.")

    os.environ["VLLM_ATTENTION_BACKEND"] = backend

    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

    with vllm_runner(model,
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
        ("facebook/opt-125m", "ray", "", "L4"),
        ("facebook/opt-125m", "mp", "", "L4"),
        ("meta-llama/Llama-2-7b-hf", "ray", "", "L4"),
        ("meta-llama/Llama-2-7b-hf", "mp", "", "L4"),
        ("facebook/opt-125m", "ray", "", "A100"),
        ("facebook/opt-125m", "mp", "", "A100"),
        ("facebook/opt-125m", "mp", "FLASHINFER", "A100"),
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

    if model == "meta-llama/Llama-2-7b-hf" and distributed_executor_backend == "ray" and attention_backend == "" and test_suite == "L4":  # noqa
        # test ray adag
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


def test_model_with_failure(vllm_runner) -> None:
    try:
        with patch("vllm.model_executor.models.opt.OPTForCausalLM.forward",
                   side_effect=ValueError()):
            with pytest.raises(ValueError) as exc_info:
                vllm_runner("facebook/opt-125m",
                            dtype="half",
                            enforce_eager=False,
                            gpu_memory_utilization=0.7)
            matches = re.search(r"input dumped to (.+).pkl",
                                str(exc_info.value))
            assert matches is not None
            filename = f"{matches.group(1)}.pkl"

        with open(filename, "rb") as filep:
            inputs = pickle.load(filep)

        if any(key not in inputs for key in ("arg_1", "arg_2", "arg_3")):
            raise AssertionError("Missing keys in dumped inputs. Dumped keys: "
                                 f"{list(inputs.keys())}")
        assert isinstance(inputs["arg_1"],
                          ModelInputForGPUWithSamplingMetadata)
    finally:
        os.remove(filename)
