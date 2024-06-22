"""Compare the outputs of HF and vLLM when using greedy sampling.

Because of numerical precision, we will have "close" but not
necessarily bitwise correctness vs HF implementation. So, we 
compare the logprobs generated from each and check that they
are generally "close" to one another.

With vLLM, we cannot re-initialize an LLM engine in the same 
process when using TP due to cleanup issues. As a result, 
we cannot use pytest. Launch with:

TEST_DIST_MODEL=meta-llama/Meta-Llama-3-8B-Instruct DISTRIBUTED_EXECUTOR_BACKEND="ray" pytest -s tests/distributed/models_core/test_llm_logprobs.py
TEST_DIST_MODEL=meta-llama/Meta-Llama-3-8B-Instruct DISTRIBUTED_EXECUTOR_BACKEND="mp" pytest -s tests/distributed/models_core/test_llm_logprobs.py
"""
import os
import pytest
import torch

from tests.models.utils import check_logprobs_close
from tests.nm_utils.utils_skip import should_skip_test_group

if should_skip_test_group(group_name="TEST_DISTRIBUTED"):
    pytest.skip("TEST_DISTRIBUTED=DISABLE, skipping distributed model test group",
                allow_module_level=True)

MAX_TOKENS = 32
NUM_LOGPROBS = 5
MODEL_MAX_LEN = 1024
NUM_DEVICES = torch.cuda.device_count()

TEST_DIST_MODEL = "TEST_DIST_MODEL"
DISTRIBUTED_EXECUTOR_BACKEND = "DISTRIBUTED_EXECUTOR_BACKEND"
VLLM_ATTENTION_BACKEND = "VLLM_ATTENTION_BACKEND"

@pytest.mark.skipif(NUM_DEVICES < 2,
                    reason="Need at least 2 GPUs to run the test.")
def test_models(
    vllm_runner_nm,
    hf_runner_nm,
    example_prompts,
) -> None:
    model = os.getenv(TEST_DIST_MODEL)
    distributed_executor_backend = os.getenv(DISTRIBUTED_EXECUTOR_BACKEND)
    backend_by_env_var = os.getenv(VLLM_ATTENTION_BACKEND)

    hf_model = hf_runner_nm(model)
    hf_outputs = hf_model.generate_greedy_logprobs_nm(example_prompts,
                                                      MAX_TOKENS,
                                                      NUM_LOGPROBS)

    del hf_model

    enforce_eager = backend_by_env_var == "FLASHINFER"

    vllm_model = vllm_runner_nm(model, 
        max_model_len=MODEL_MAX_LEN,
        enforce_eager=enforce_eager,
        distributed_executor_backend=distributed_executor_backend,
        tensor_parallel_size=NUM_DEVICES,
    )
    vllm_outputs = vllm_model.generate_greedy_logprobs(example_prompts,
                                                       MAX_TOKENS,
                                                       NUM_LOGPROBS)

    del vllm_model

    # loop through the prompts
    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf_model",
        name_1="vllm_model",
    )
