# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random
import sys
from typing import Any

import pytest

from tests.utils import create_new_process_for_each_test
from tests.v1.logits_processors.utils import (
    DUMMY_LOGITPROC_ARG,
    DUMMY_LOGITPROC_FQCN,
    DUMMY_LOGITPROC_MODULE,
    MAX_TOKENS,
    MODEL_NAME,
    POOLING_MODEL_NAME,
    TEMP_GREEDY,
    CustomLogitprocSource,
    DummyLogitsProcessor,
    WrappedPerReqLogitsProcessor,
    dummy_module,
    prompts,
)
from tests.v1.logits_processors.utils import entry_points as fake_entry_points
from vllm import LLM, SamplingParams
from vllm.v1.sample.logits_processor import (
    STR_POOLING_REJECTS_LOGITSPROCS,
    STR_SPEC_DEC_REJECTS_LOGITSPROCS,
    LogitsProcessor,
)

# Create a mixture of requests which do and don't utilize the dummy logitproc
sampling_params_list = [
    SamplingParams(
        temperature=TEMP_GREEDY,
        max_tokens=MAX_TOKENS,
        extra_args={DUMMY_LOGITPROC_ARG: 128},
    ),
    SamplingParams(temperature=TEMP_GREEDY, max_tokens=MAX_TOKENS),
    SamplingParams(
        temperature=TEMP_GREEDY,
        max_tokens=MAX_TOKENS,
        extra_args={DUMMY_LOGITPROC_ARG: 67},
    ),
    SamplingParams(temperature=TEMP_GREEDY, max_tokens=MAX_TOKENS),
]


def _run_test(kwargs: dict, logitproc_loaded: bool) -> None:
    """Compare `LLM` instance initialized with specified `kwargs` against
    reference `LLM` instance.

    Two scenarios:
    1. Server has loaded dummy logitproc; test that requests which specify
       dummy logitproc arg value behave as if logitproc is operating (output
       token value should repeat), while requests that don't specify dummy
       logitproc arg value should match reference `LLM` output.
    2. Server has *not* loaded dummy logitproc; test that all requests
       behave as if logitproc is *not* operating (output matches reference
       `LLM` output.)

    Args:
      kwargs: `LLM` constructor kwargs
      logitproc_loaded: server has loaded dummy logitproc if True
    """

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
        zip(outputs_logitproc, outputs_ref, sampling_params_list)
    ):
        lp_toks = out_lp.outputs[0].token_ids
        if logitproc_loaded and params.extra_args:
            # This request exercises custom logitproc; validate that logitproc
            # forces `target_token` to be decoded in each step
            target_token = params.extra_args[DUMMY_LOGITPROC_ARG]
            if not all(x == target_token for x in lp_toks):
                raise AssertionError(
                    f"Request {bdx} generated {lp_toks}, should all be {target_token}"
                )
        else:
            # This request does not exercise custom logitproc (or custom
            # logitproc is not enabled on this server); validate against
            # reference result
            ref_toks = out_ref.outputs[0].token_ids
            if lp_toks != ref_toks:
                raise AssertionError(
                    f"Request {bdx} generated {lp_toks}, should match {ref_toks}"
                )


@create_new_process_for_each_test()
@pytest.mark.parametrize("logitproc_source", list(CustomLogitprocSource))
def test_custom_logitsprocs(monkeypatch, logitproc_source: CustomLogitprocSource):
    """Test offline Python interface for passing custom logitsprocs

    Construct an `LLM` instance which loads a custom logitproc that has a
    well-defined behavior (mask out all tokens except one `target_token`)

    Construct a reference `LLM` instance with no custom logitproc

    Pass in a batch of requests, 50% of which pass a `target_token` value
    in through `SamplingParams.extra_args`, 50% of which do not.

    Validate that
    * Requests which do not activate the custom logitproc, yield the same
      results for both `LLM` instances
    * Requests which activate the custom logitproc, only output `target_token`

    Test four scenarios, corresponding to `logitproc_source` value
    * No logitsprocs loaded - test that generated tokens match reference `LLM`
      instance output
    * Logitproc passed in via {entrypoint, class object, fully-qualified class
      name (FQCN)} - test that dummy logitproc is utilized correctly when
      provided via any of these three possible sources

    Args:
      monkeypatch: for setting env vars
      logitproc_source: what source (entrypoint, fully-qualified class name
                        (FQCN), class object, or None) the user pulls the
                        logitproc from
    """

    # Test that logitproc info is passed to workers
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    random.seed(40)

    # Choose LLM args based on logitproc source
    if logitproc_source == CustomLogitprocSource.LOGITPROC_SOURCE_NONE:
        # Scenario: the server does not load any custom logitproc
        # Every other scenario is a different way of loading a custom logitproc
        _run_test({}, logitproc_loaded=False)
        return

    if logitproc_source == CustomLogitprocSource.LOGITPROC_SOURCE_ENTRYPOINT:
        # Scenario: vLLM loads a logitproc from a preconfigured entrypoint
        # To that end, mock a dummy logitproc entrypoint
        import importlib.metadata

        importlib.metadata.entry_points = fake_entry_points  # type: ignore

        # fork is required for workers to see entrypoint patch
        monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")
        _run_test({}, logitproc_loaded=True)
        return

    kwargs: dict[str, list[str | type[LogitsProcessor]]] = {}
    if logitproc_source == CustomLogitprocSource.LOGITPROC_SOURCE_FQCN:
        # Scenario: load logitproc based on fully-qualified class name (FQCN)
        # Inject dummy module which defines logitproc
        sys.modules[DUMMY_LOGITPROC_MODULE] = dummy_module
        kwargs["logits_processors"] = [DUMMY_LOGITPROC_FQCN]
    elif logitproc_source == CustomLogitprocSource.LOGITPROC_SOURCE_CLASS:
        # Scenario: load logitproc from provided class object
        kwargs["logits_processors"] = [DummyLogitsProcessor]

    _run_test(kwargs, logitproc_loaded=True)


@create_new_process_for_each_test()
def test_custom_logitsprocs_req(monkeypatch):
    """Test passing request-level logits processor to offline Python interface

    Wrap a request-level logits processor to create a batch level logits
    processor that has a well-defined behavior (mask out all tokens except one
    `target_token`)

    Construct an `LLM` instance which loads the wrapped logits processor. Pass
    the custom logitproc as a class object.

    Construct a reference `LLM` instance with no custom logitproc

    Pass in a batch of requests, 50% of which pass a `target_token` value
    in through `SamplingParams.extra_args`, 50% of which do not.

    Validate that
    * Requests which do not activate the custom logitproc, yield the same
      results for both `LLM` instances
    * Requests which activate the custom logitproc, only output `target_token`

    Args:
      monkeypatch: for setting env vars
    """

    # Test that logitproc info is passed to workers
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    random.seed(40)
    _run_test(
        {"logits_processors": [WrappedPerReqLogitsProcessor]}, logitproc_loaded=True
    )


@create_new_process_for_each_test()
@pytest.mark.parametrize("model_scenario", ["pooling", "spec_dec"])
@pytest.mark.parametrize(
    "logitproc_source",
    [
        CustomLogitprocSource.LOGITPROC_SOURCE_ENTRYPOINT,
        CustomLogitprocSource.LOGITPROC_SOURCE_FQCN,
        CustomLogitprocSource.LOGITPROC_SOURCE_CLASS,
    ],
)
def test_rejects_custom_logitsprocs(
    monkeypatch, model_scenario: str, logitproc_source: CustomLogitprocSource
):
    """Validate that vLLM engine initialization properly rejects custom
    logitsprocs when the model is a pooling model or speculative decoding
    enabled.

    Use `LLM` entrypoint. We expect `LLM` initialization to fail before the
    logitproc is actually loaded.

    Scenario 1:
    * Mock a logitproc entrypoint
    * Validate that `LLM` does not load the logitproc

    Scenario 2:
    * Pass custom logitproc to `LLM` constructor
      * Scenario 2a: via FQCN
      * Scenario 2b: via class object
    * Validate that initialization fails with appropriate exception

    Args:
      monkeypatch: used to set environment variables
      logitproc_source: what source (entrypoint, fully-qualified class name
                        (FQCN), or class object) the user pulls the
                        logitproc from
    """
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    random.seed(40)

    test_params: dict[str, dict[str, Any]] = {
        "pooling": {
            "runner": "pooling",
            "model": POOLING_MODEL_NAME,
            "error_message": STR_POOLING_REJECTS_LOGITSPROCS,
            "speculative_config": None,
        },
        "spec_dec": {
            "runner": "auto",
            "model": MODEL_NAME,
            "error_message": STR_SPEC_DEC_REJECTS_LOGITSPROCS,
            "speculative_config": {"model": "ngram", "num_speculative_tokens": 1},
        },
    }

    config = test_params[model_scenario]

    llm_kwargs: dict[str, Any] = {
        "runner": config["runner"],
        "model": config["model"],
        "gpu_memory_utilization": 0.1,
        "speculative_config": config["speculative_config"],
    }

    if logitproc_source == CustomLogitprocSource.LOGITPROC_SOURCE_ENTRYPOINT:
        # Scenario: vLLM loads a model and ignores a logitproc that is
        # available at a preconfigured entrypoint

        # Patch in dummy logitproc entrypoint
        import importlib.metadata

        importlib.metadata.entry_points = fake_entry_points  # type: ignore

        # fork is required for entrypoint patch to be visible to workers,
        # although they should ignore the entrypoint patch anyway
        monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")

        llm = LLM(**llm_kwargs)
        # Require that no logitsprocs have been loaded
        worker = llm.llm_engine.model_executor.driver_worker.worker
        assert sum([1 for _ in worker.model_runner.input_batch.logitsprocs.all]) == 0
        return

    if logitproc_source == CustomLogitprocSource.LOGITPROC_SOURCE_FQCN:
        # Scenario: load logitproc based on fully-qualified class name (FQCN)
        llm_kwargs["logits_processors"] = [DUMMY_LOGITPROC_FQCN]
    elif logitproc_source == CustomLogitprocSource.LOGITPROC_SOURCE_CLASS:
        # Scenario: load logitproc from provided class object
        llm_kwargs["logits_processors"] = [DummyLogitsProcessor]

    with pytest.raises(ValueError, match=config["error_message"]):
        # Require that loading a model alongside the logitproc raises
        # the appropriate exception.
        LLM(**llm_kwargs)
