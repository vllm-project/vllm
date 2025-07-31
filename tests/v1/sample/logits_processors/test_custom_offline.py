# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random

import pytest

from tests.v1.sample.logits_processors.utils import (DUMMY_LOGITPROC_ARG,
                                                     MAX_TOKENS, MODEL_NAME,
                                                     TEMP_GREEDY,
                                                     CustomLogitprocSource,
                                                     prompts)
from vllm import LLM, SamplingParams
from vllm.test_utils import DUMMY_LOGITPROC_FQCN, DummyLogitsProcessor

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


def _run_test(kwargs: dict, logitproc_loaded: bool):
    # Create a vLLM instance and load custom logitproc
    llm_logitproc = LLM(
        model=MODEL_NAME,
        gpu_memory_utilization=0.1,
        **kwargs,
    )

    # Create a reference vLLM instance without custom logitproc
    llm_ref = LLM(model=MODEL_NAME, gpu_memory_utilization=0.1)

    # Run inference with logitproc loaded
    outputs_logitproc = llm_logitproc.generate(prompts, sampling_params_list)

    # Reference run
    outputs_ref = llm_ref.generate(prompts, sampling_params_list)

    # Validate outputs
    for bdx, (out_lp, out_ref, params) in enumerate(
            zip(outputs_logitproc, outputs_ref, sampling_params_list)):
        lp_toks = out_lp.outputs[0].token_ids
        if logitproc_loaded and params.extra_args:
            # This request exercises custom logitproc; validate that logitproc
            # forces `target_token` to be decoded in each step
            target_token = params.extra_args[DUMMY_LOGITPROC_ARG]
            if not all(x == target_token for x in lp_toks):
                raise AssertionError(
                    f"Request {bdx} generated {lp_toks}, shoud all be "
                    f"{target_token}")
        else:
            # This request does not exercise custom logitproc (or custom
            # logitproc is not enabled on this server); validate against
            # reference result
            ref_toks = out_ref.outputs[0].token_ids
            if lp_toks != ref_toks:
                raise AssertionError(
                    f"Request {bdx} generated {lp_toks}, should match "
                    f"{ref_toks}")


@pytest.mark.parametrize("logitproc_source", list(CustomLogitprocSource))
def test_custom_logitsprocs_py(monkeypatch,
                               logitproc_source: CustomLogitprocSource):
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
      logitproc_source: what source (entrypoint, fully-qualified class name
                        (FQCN), or class object) the user pulls the
                        logitproc from
    """
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    random.seed(40)

    # Choose LLM args based on logitproc source
    kwargs = {}
    if logitproc_source == CustomLogitprocSource.LOGITPROC_SOURCE_NONE:
        # Scenario: the server does not load any custom logitproc
        # Every other scenario is a different way of loading a custom logitproc
        _run_test(kwargs, logitproc_loaded=False)
        return
    if logitproc_source == CustomLogitprocSource.LOGITPROC_SOURCE_ENTRYPOINT:
        # Scenario: vLLM loads a logitproc from a preconfigured entrypoint
        # To that end, mock a dummy logitproc entrypoint
        monkeypatch.setenv("VLLM_MOCK_LP_ENTRYPOINT", "1")
    elif logitproc_source == CustomLogitprocSource.LOGITPROC_SOURCE_FQCN:
        # Scenario: load logitproc based on fully-qualified class name (FQCN)
        kwargs["logits_processors"] = [DUMMY_LOGITPROC_FQCN]
    elif logitproc_source == CustomLogitprocSource.LOGITPROC_SOURCE_CLASS:
        # Scenario: load logitproc from provided class object
        kwargs["logits_processors"] = [DummyLogitsProcessor]

    # Test one of the above scenarios where the server loads a custom logitproc
    _run_test(kwargs, logitproc_loaded=True)
