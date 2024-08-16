"""Compare the outputs of HF and distributed vLLM when using greedy sampling.

Run:
```sh
pytest test_chunked_prefill_distributed.py
```
"""

import os

import pytest

from vllm.utils import cuda_device_count_stateless

from ..models.utils import check_outputs_equal
from ..utils import fork_new_process_for_each_test


@pytest.mark.skipif(cuda_device_count_stateless() < 2,
                    reason="Need at least 2 GPUs to run the test.")
@pytest.mark.parametrize("model, distributed_executor_backend, enable_spmd", [
    ("facebook/opt-125m", "ray", False),
    ("facebook/opt-125m", "ray", True),
    ("meta-llama/Llama-2-7b-hf", "ray", False),
    ("facebook/opt-125m", "mp", False),
    ("meta-llama/Llama-2-7b-hf", "mp", False),
])
@fork_new_process_for_each_test
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    distributed_executor_backend: str,
    enable_spmd: bool,
) -> None:
    if enable_spmd:
        assert distributed_executor_backend == "ray"
        # test ray adag
        os.environ['VLLM_USE_RAY_SPMD_WORKER'] = "1"
        os.environ['VLLM_USE_RAY_COMPILED_DAG'] = "1"

    dtype = "half"
    max_tokens = 5
    chunked_prefill_token_size = 16

    # Add a chunked prefill config.
    max_num_seqs = min(chunked_prefill_token_size, 256)
    assert chunked_prefill_token_size != -1
    enable_chunked_prefill = True
    max_num_batched_tokens = chunked_prefill_token_size

    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).

    with vllm_runner(
            model,
            dtype=dtype,
            tensor_parallel_size=2,
            max_num_seqs=max_num_seqs,
            enable_chunked_prefill=enable_chunked_prefill,
            max_num_batched_tokens=max_num_batched_tokens,
            distributed_executor_backend=distributed_executor_backend,
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
