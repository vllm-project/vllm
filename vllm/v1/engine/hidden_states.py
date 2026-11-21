# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frontend accumulation and terminal delivery of hidden-state rows."""

import torch

from vllm.exceptions import GenerationError
from vllm.outputs import CompletionOutput, WeightVersionSpan
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import FinishReason


class HiddenStatesAccumulator:
    def __init__(self, params: SamplingParams, prompt_len: int):
        outputs = params.aux_outputs or ()
        self.prompt_len = prompt_len
        self.chunks: list[torch.Tensor] | None = (
            [] if "hidden_states" in outputs else None
        )
        self.start = max(params.hidden_states_start, prompt_len - 1)
        self.end = params.hidden_states_end
        self.empty: torch.Tensor | None = None
        self.prompt_chunks: list[torch.Tensor] | None = (
            [] if "prompt_hidden_states" in outputs else None
        )
        self.prompt_start = params.hidden_states_start
        self.prompt_end = max(
            self.prompt_start,
            min(
                self.end if self.end is not None else prompt_len - 1,
                prompt_len - 1,
            ),
        )
        self.prompt_cursor = self.prompt_start
        self.prompt_versions: list[WeightVersionSpan] = []

    @classmethod
    def for_request(
        cls, params: SamplingParams | None, prompt_len: int
    ) -> "HiddenStatesAccumulator | None":
        if params is None or not params.aux_outputs:
            return None
        return cls(params, prompt_len)

    def add_generated(self, rows: torch.Tensor) -> None:
        assert self.chunks is not None
        if len(rows):
            self.chunks.append(rows)

    def add_prompt(
        self, start: int, rows: torch.Tensor, weight_version: str | None
    ) -> None:
        assert self.prompt_chunks is not None
        # Replayed prefill rows do not replace the first computation.
        offset = max(0, self.prompt_cursor - start)
        rows = rows[offset:]
        if not len(rows):
            return
        self.prompt_chunks.append(rows)
        self.prompt_cursor = start + offset + len(rows)
        if weight_version is not None:
            begin = start + offset - self.prompt_start
            spans = self.prompt_versions
            if spans and spans[-1].version == weight_version and spans[-1].end == begin:
                spans[-1].end += len(rows)
            else:
                spans.append(
                    WeightVersionSpan(weight_version, begin, begin + len(rows))
                )

    def _generated_rows(self, num_output_tokens: int) -> int:
        end = self.prompt_len - 1 + num_output_tokens
        if self.end is not None:
            end = min(end, self.end)
        return max(0, end - self.start)

    def trim(self, num_output_tokens: int) -> None:
        if self.chunks is not None:
            keep = self._generated_rows(num_output_tokens)
            for index, rows in enumerate(self.chunks):
                self.chunks[index] = rows[:keep]
                keep = max(0, keep - len(rows))

    def _join(
        self,
        chunks: list[torch.Tensor],
        expected: int,
        finish_reason: FinishReason,
        label: str,
    ) -> torch.Tensor | None:
        rows = (
            torch.cat(chunks)
            if len(chunks) > 1
            else chunks[0]
            if chunks
            else self.empty
        )
        if (
            rows is None or len(rows) != expected
        ) and finish_reason != FinishReason.ABORT:
            raise GenerationError(f"Requested {label} rows were not delivered")
        return rows

    def finalize(
        self,
        output: CompletionOutput,
        num_output_tokens: int,
        finish_reason: FinishReason,
    ) -> None:
        if finish_reason == FinishReason.ERROR:
            return
        if self.chunks is not None:
            output.hidden_states = self._join(
                self.chunks,
                self._generated_rows(num_output_tokens),
                finish_reason,
                "hidden-state",
            )
            if output.hidden_states is not None:
                output.hidden_states_start = self.start
        if self.prompt_chunks is not None:
            output.prompt_hidden_states = self._join(
                self.prompt_chunks,
                self.prompt_end - self.prompt_start,
                finish_reason,
                "prompt hidden-state",
            )
            if output.prompt_hidden_states is not None:
                output.prompt_hidden_states_start = self.prompt_start
                output.prompt_hidden_states_weight_versions = self.prompt_versions
