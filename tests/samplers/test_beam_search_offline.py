# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import CompletionOutput, RequestOutput
from vllm.entrypoints.generate.beam_search.offline import BeamSearchOfflineMixin
from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.v1.structured_output.backend_types import StructuredOutputOptions


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


_VOCAB_SIZE = 64


class _Grammar:
    """Grammar stub that allows a fixed set of token IDs and never terminates."""

    allowed_token_ids = (3, 7)

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        return True

    def is_terminated(self) -> bool:
        return False

    def fill_bitmask(self, bitmask: torch.Tensor, batch_index: int) -> None:
        bitmask[batch_index].zero_()
        for token_id in self.allowed_token_ids:
            bitmask[batch_index][token_id // 32] |= 1 << (token_id % 32)


class _Backend:
    def compile_grammar(self, request_type, grammar_spec) -> _Grammar:
        return _Grammar()


class _ModelConfig:
    @staticmethod
    def get_vocab_size() -> int:
        return _VOCAB_SIZE


def test_structured_output_beam_params_disable_detokenization() -> None:
    """Same as above for the structured-output path, which builds its own
    per-beam SamplingParams instead of reusing the base ones."""
    serving = _FakeOffline()
    serving.model_config = _ModelConfig()
    base_params = SamplingParams(
        logprobs=4,
        max_tokens=1,
        temperature=0.0,
        detokenize=False,
        skip_clone=True,
    )
    beams = [
        BeamSearchSequence(
            orig_prompt={"type": "token", "prompt": "prompt", "prompt_token_ids": [1]},
            tokens=[1, 3],
            logprobs=[],
        )
        for _ in range(2)
    ]

    built = serving._build_beam_sampling_params(
        beams,
        base_params,
        _Backend(),
        (StructuredOutputOptions.JSON, "{}"),
        torch.zeros(1, _VOCAB_SIZE // 32, dtype=torch.int32),
    )

    assert len(built) == len(beams)
    assert all(entry is not None for entry in built)
    assert all(not entry[0].detokenize for entry in built)
