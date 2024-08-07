"""For encoder/decoder models only:
Compare the outputs of HF and distributed vLLM when using greedy sampling.

Run:
```sh
cd $VLLM_PATH/tests

pytest distributed/test_basic_distributed_correctness_enc_dec.py
```
"""

import pytest

from tests.models.utils import DecoderPromptType
from vllm.utils import cuda_device_count_stateless

from ..models.utils import check_logprobs_close
from ..utils import fork_new_process_for_each_test


@pytest.mark.skipif(cuda_device_count_stateless() < 2,
                    reason="Need at least 2 GPUs to run the test.")
@pytest.mark.parametrize("model, distributed_executor_backend", [
    ("facebook/bart-large-cnn", "ray"),
    ("facebook/bart-large-cnn", "mp"),
])
@fork_new_process_for_each_test
def test_models(
    model: str,
    distributed_executor_backend: str,
    hf_runner,
    vllm_runner,
    example_encoder_decoder_prompts,
) -> None:
    '''
    Test vLLM BART inference on more than one GPU, comparing
    outputs against HF as a baseline.

    Fork a new process for each test, to prevent CUDA from
    being re-initialized by successive tests within the same
    process.

    Arguments:

    * model: the HF ID of the specific BART variant under test
    * distributed_executor_backend
    * hf_runner: HuggingFace (HF) test model runner
    * vllm_runner: vLLM test model runner
    * example_encoder_decoder_prompts: test fixture which provides a 
                                        dictionary of dummy prompts
    '''

    dtype = "float"
    max_tokens = 64
    num_logprobs = 5

    # Example inputs with non-trivial (i.e. not None/empty) encoder &
    # decoder prompts.
    test_prompts = example_encoder_decoder_prompts[DecoderPromptType.CUSTOM]

    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).
    with vllm_runner(
            model,
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
        hf_outputs = (hf_model.generate_encoder_decoder_greedy_logprobs_limit(
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
