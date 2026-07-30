# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import CompletionOutput, RequestOutput
from vllm.entrypoints.generate.beam_search.offline import BeamSearchOfflineMixin
from vllm.sampling_params import BeamSearchParams


class _Tokenizer:
    eos_token_id = 0

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


class _Renderer:
    def get_tokenizer(self) -> _Tokenizer:
        return _Tokenizer()


class _FakeOffline(BeamSearchOfflineMixin):
    """Drives ``beam_search`` without a real engine to inspect the per-step
    sampling params."""

    renderer = _Renderer()

    def __init__(self) -> None:
        self.captured_params: list = []

    def _preprocess_cmpl(self, prompts):
        return [
            {"type": "token", "prompt": "prompt", "prompt_token_ids": [1]}
            for _ in prompts
        ]

    def _render_and_run_requests(self, prompts, params, output_type, **kwargs):
        num_requests = len(list(prompts))
        self.captured_params.extend(params[:num_requests])
        # No logprobs -> beams do not expand, so the search stops immediately.
        return [
            RequestOutput(
                request_id=str(i),
                prompt="prompt",
                prompt_token_ids=[1],
                prompt_logprobs=None,
                outputs=[
                    CompletionOutput(
                        index=0,
                        text="",
                        token_ids=[],
                        cumulative_logprob=None,
                        logprobs=None,
                        finish_reason="length",
                    )
                ],
                finished=True,
            )
            for i in range(num_requests)
        ]


def test_offline_beam_search_disables_incremental_detokenization() -> None:
    """The search loop only needs token IDs and logprob values; text is decoded
    once at the end. Requesting detokenization every step would add redundant
    decoding work per beam, per step (see issue #49197)."""
    serving = _FakeOffline()
    params = BeamSearchParams(beam_width=2, max_tokens=4)

    serving.beam_search(["prompt"], params)

    assert serving.captured_params, "expected the engine to be invoked at least once"
    assert all(not p.detokenize for p in serving.captured_params)
