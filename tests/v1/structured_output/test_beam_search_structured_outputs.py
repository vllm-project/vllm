# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Beam search on the structured-output path.

Both tests cover defects that made this path unusable: the backend was never
resolved from the `auto` sentinel, and masked (`-inf`) candidates were only filtered
when the request carried an `allowed_token_ids` set.
"""

import math

import pytest

from tests.utils import create_new_process_for_each_test
from tests.v1.logits_processors.utils import DummyLogitsProcessor
from vllm import LLM
from vllm.inputs import TokensPrompt
from vllm.sampling_params import BeamSearchParams, StructuredOutputsParams

MODEL_NAME = "facebook/opt-125m"


@create_new_process_for_each_test()
def test_beam_search_resolves_the_auto_backend():
    """A beam request with structured outputs must run on the default config.

    `_backend` is resolved from the `auto` sentinel by `SamplingParams.verify()`,
    which the engine runs per request. Beam search builds its backend before any
    request exists, so `_backend` still held `auto` and the backend chain raised
    `Unsupported structured output backend: auto` -- meaning this path could not
    run at all unless a concrete backend was named in the engine config.
    """
    llm = LLM(
        model=MODEL_NAME,
        gpu_memory_utilization=0.3,
        max_model_len=256,
        disable_log_stats=True,
    )
    config = llm.llm_engine.vllm_config.structured_outputs_config
    assert config.backend == "auto", (
        f"this test is only meaningful with the default backend, got {config.backend}"
    )

    outputs = llm.beam_search(
        [TokensPrompt(prompt_token_ids=[0, 100, 200])],
        BeamSearchParams(
            beam_width=2,
            max_tokens=3,
            temperature=0.0,
            structured_outputs=StructuredOutputsParams(regex=r"[0-9]{3}"),
        ),
    )
    assert outputs and outputs[0].sequences, "beam search returned nothing"


@create_new_process_for_each_test()
def test_beam_search_drops_non_finite_candidates(monkeypatch):
    """A masked candidate is not a beam the search may take.

    Beam search expands from the returned logprobs and filtered `-inf` entries only
    when `request_allowed` was set, i.e. only for an `allowed_token_ids` request. Any
    other source of masking -- here a logits processor -- left `-inf` candidates in
    the running, and with every sibling equally impossible one of them won on score.
    Observed before the fix as a beam collapsing to a single EOS token at `-inf`.

    `processed_logprobs` is required for the mask to appear in the returned logprobs
    at all; under the default `raw_logprobs` they are computed before logits
    processors run.
    """
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")

    target_token = 128

    class MaskAllButOne(DummyLogitsProcessor):
        """Masks every token but `target_token`, for every request.

        The shipped dummy takes its target from `SamplingParams.extra_args`, which
        beam search does not forward; hard-coding it keeps this test independent of
        that.
        """

        @staticmethod
        def _get_target_token(params) -> int:
            return target_token

    llm = LLM(
        model=MODEL_NAME,
        gpu_memory_utilization=0.3,
        max_model_len=256,
        logits_processors=[MaskAllButOne],
        logprobs_mode="processed_logprobs",
        disable_log_stats=True,
    )
    prompt = TokensPrompt(prompt_token_ids=[0, 100, 200])
    outputs = llm.beam_search(
        [prompt],
        BeamSearchParams(beam_width=2, max_tokens=4, temperature=0.0),
    )

    assert outputs and outputs[0].sequences, "beam search returned nothing"
    prompt_len = len(prompt["prompt_token_ids"])
    for sdx, seq in enumerate(outputs[0].sequences):
        assert math.isfinite(seq.cum_logprob), (
            f"beam {sdx} was built from a masked candidate: cum_logprob="
            f"{seq.cum_logprob}"
        )
        generated = [int(t) for t in seq.tokens[prompt_len:]]
        assert generated, f"beam {sdx} generated nothing"
        # The processor masks everything but `target_token`, so once the filter is
        # honoured there is nothing else for a beam to take.
        assert all(t == target_token for t in generated), (
            f"beam {sdx} generated {generated}, expected only {target_token}"
        )


if __name__ == "__main__":
    pytest.main([__file__])
