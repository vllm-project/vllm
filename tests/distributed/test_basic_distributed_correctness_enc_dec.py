"""For encoder/decoder models only:
Compare the outputs of HF and distributed vLLM when using greedy sampling.
vLLM will allocate all the available memory, so we need to run the tests one
by one. The solution is to pass arguments (model name) by environment
variables.
Run:
```sh
cd $VLLM_PATH/tests

TEST_DIST_MODEL=facebook/bart-large-cnn pytest \
    distributed/test_basic_distributed_correctness_enc_dec.py
TEST_DIST_MODEL=facebook/bart-large-cnn \
    distributed/test_basic_distributed_correctness_enc_dec.py
```
"""
import os

import pytest

from tests.models.utils import DecoderPromptType

from vllm.utils import cuda_device_count_stateless

from ..models.utils import check_logprobs_close

MODELS = [
    os.environ["TEST_DIST_MODEL"],
]
DISTRIBUTED_EXECUTOR_BACKEND = "DISTRIBUTED_EXECUTOR_BACKEND"


@pytest.mark.skipif(cuda_device_count_stateless() < 2,
                    reason="Need at least 2 GPUs to run the test.")
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(
    hf_runner,
    vllm_runner,
    example_encoder_decoder_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    distributed_executor_backend = os.getenv(DISTRIBUTED_EXECUTOR_BACKEND)

    test_prompts = example_encoder_decoder_prompts[DecoderPromptType.CUSTOM]

    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).
    with vllm_runner(model, 
                     dtype=dtype,
                     tensor_parallel_size=2,
                     distributed_executor_backend=distributed_executor_backend, 
                     enforce_eager=True,
                     ) as vllm_model:
        vllm_outputs = vllm_model.generate_encoder_decoder_greedy_logprobs(
            test_prompts, max_tokens, num_logprobs)

    # Configuration settings for HF baseline
    hf_kwargs = {
        "top_k": None,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "top_p": 1.0,
        "length_penalty": 1.0,
        "early_stopping": False,
        "no_repeat_ngram_size": None,
        "min_length": 0
    }

    with hf_runner(model, dtype=dtype,
                    is_encoder_decoder_model=True) as hf_model:
        hf_outputs = (
            hf_model.generate_encoder_decoder_greedy_logprobs_limit(
                test_prompts,
                max_tokens,
                num_logprobs,
                **hf_kwargs,
            ))

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
