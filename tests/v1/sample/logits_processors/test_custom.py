# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random

import pytest

from vllm import LLM, SamplingParams

MODEL = "facebook/opt-125m"
DUMMY_LOGITPROC_ENTRYPOINT = "dummy_logitproc"
DUMMY_LOGITPROC_FQN = (
    "vllm.v1.sample.logits_processor.impls:DummyLogitsProcessor")
DUMMY_LOGITPROC_ARG = "target_token"
LOGITPROC_SOURCE_ENTRYPOINT = "entrypoint"
LOGITPROC_SOURCE_FQN = "fqn"
TEMP_GREEDY = 0.0
MAX_TOKENS = 20

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a mixture of requests which do and don't utilize the dummy logitproc
sampling_params_list = [
    SamplingParams(temperature=TEMP_GREEDY,
                   max_tokens=MAX_TOKENS,
                   extra_args={DUMMY_LOGITPROC_ARG: 128}),
    SamplingParams(temperature=TEMP_GREEDY, max_tokens=MAX_TOKENS),
    SamplingParams(temperature=TEMP_GREEDY,
                   max_tokens=MAX_TOKENS,
                   extra_args={DUMMY_LOGITPROC_ARG: 67}),
    SamplingParams(temperature=TEMP_GREEDY, max_tokens=MAX_TOKENS),
]


@pytest.mark.parametrize("logitproc_source",
                         [LOGITPROC_SOURCE_FQN, LOGITPROC_SOURCE_ENTRYPOINT])
def test_custom_logitsprocs(logitproc_source: str):
    """Test Python interface for passing custom logitsprocs
    
    Construct an `LLM` instance which loads a custom logitproc that has a
    well-defined behavior (mask out all tokens except one `target_token`)

    Construct a reference `LLM` instance with no custom logitproc

    Pass in a batch of requests, 50% of which pass a `target_token` value
    in through `SamplingParams.extra_args`, 50% of which do not.

    Validate that
    * Requests which do not activate the custom logitproc, yield the same
      results for both `LLM` instances
    * Requests which activate the custom logitproc, only output `target_token`

    Args:
      logitproc_source: what source (entrypoint or fully-qualified name) the 
                        user pulls the logitproc from
    """
    random.seed(40)

    # Choose LLM args based on logitproc source
    kwargs = ({
        "logits_processors_entrypoints": [DUMMY_LOGITPROC_ENTRYPOINT]
    } if logitproc_source == LOGITPROC_SOURCE_ENTRYPOINT else {
        "logits_processors_fqns": [DUMMY_LOGITPROC_FQN]
    })

    # Create a vLLM instance and load custom logitproc
    llm_logitproc = LLM(
        model=MODEL,
        gpu_memory_utilization=0.1,
        **kwargs,
    )

    # Create a reference vLLM instance without custom logitproc
    llm_ref = LLM(model=MODEL, gpu_memory_utilization=0.1)

    # Run inference with logitproc loaded
    outputs_logitproc = llm_logitproc.generate(prompts, sampling_params_list)

    # Reference run
    outputs_ref = llm_ref.generate(prompts, sampling_params_list)

    # Validate outputs
    for bdx, (out_lp, out_ref, params) in enumerate(
            zip(outputs_logitproc, outputs_ref, sampling_params_list)):
        lp_toks = out_lp.outputs[0].token_ids
        if params.extra_args:
            # This request exercises custom logitproc; validate that logitproc
            # forces `target_token` to be decoded in each step
            target_token = params.extra_args[DUMMY_LOGITPROC_ARG]
            if not all(x == target_token for x in lp_toks):
                raise AssertionError(
                    f"Request {bdx} generated {lp_toks}, shoud all be "
                    f"{target_token}")
        else:
            # This request does not exercise custom logitproc; validate
            # against reference result
            ref_toks = out_ref.outputs[0].token_ids
            if lp_toks != ref_toks:
                raise AssertionError(
                    f"Request {bdx} generated {lp_toks}, should match "
                    f"{ref_toks}")
