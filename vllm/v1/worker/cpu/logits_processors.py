# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU versions of the built-in logits processors.

They keep per-request parameters on the host and read the token history from
`LogitsProcRequestState`, so they depend only on the `LogitsProcessor`
interface, not on the GPU processors' buffers or kernels.
"""

import torch

from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.logits_processor.interface import (
    LogitsContext,
    LogitsProcessor,
    LogitsProcRequestState,
)
from vllm.v1.worker.gpu.sample.penalties import use_penalty


class _CPULogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config: VllmConfig, req_states: LogitsProcRequestState):
        self.req_states = req_states
        self.params: list = [None] * req_states.max_num_reqs

    def _active_rows(self, ctx: LogitsContext) -> list[tuple[int, int]]:
        """(row, req) for the rows of requests this processor applies to."""
        return [
            (row, req)
            for row, req in enumerate(ctx.expanded_idx_mapping.tolist())
            if self.params[req] is not None
        ]

    def _output_tokens(self, req: int, row: int, ctx: LogitsContext) -> torch.Tensor:
        """Generated tokens visible at `row`, including this step's drafts."""
        prompt_len = int(self.req_states.prompt_len.np[req])
        total_len = int(self.req_states.total_len.gpu[req])
        # input_ids at local position 0 is the last committed token.
        first_row = row - int(ctx.expanded_local_pos[row])
        return torch.cat(
            (
                self.req_states.all_token_ids.gpu[req, prompt_len:total_len],
                ctx.input_ids[first_row + 1 : row + 1].to(torch.int32),
            )
        )


class CPULogitBias(_CPULogitsProcessor):
    """Allowed token ids, logit bias and min_tokens."""

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        allowed = sampling_params.allowed_token_ids
        bias = sampling_params.logit_bias
        stop = sampling_params.all_stop_token_ids
        min_tokens = sampling_params.min_tokens
        if not (allowed or bias or (min_tokens > 0 and stop)):
            self.params[req_idx] = None
            return False
        self.params[req_idx] = (
            torch.tensor(allowed) if allowed else None,
            (torch.tensor(list(bias)), torch.tensor(list(bias.values())))
            if bias
            else None,
            torch.tensor(list(stop)) if min_tokens > 0 and stop else None,
            int(self.req_states.prompt_len.np[req_idx]) + min_tokens,
            sampling_params.structured_outputs is not None,
        )
        return True

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        for row, req in self._active_rows(ctx):
            allowed, bias, stop, min_len, restore_if_all_masked = self.params[req]
            row_logits = logits[row]
            if allowed is not None:
                allowed_logits = row_logits[allowed]
                row_logits.fill_(float("-inf"))
                row_logits[allowed] = allowed_logits
            if bias is not None:
                row_logits[bias[0]] += bias[1]
            if stop is not None and int(ctx.pos[row]) + 1 < min_len:
                stop_logits = row_logits[stop]
                row_logits[stop] = float("-inf")
                # Keep a grammar-constrained row samplable.
                if restore_if_all_masked and row_logits.max() == float("-inf"):
                    row_logits[stop] = torch.where(
                        stop_logits.isfinite(), stop_logits, float("-inf")
                    )
        return logits


class CPUPenalties(_CPULogitsProcessor):
    """Repetition, frequency and presence penalties, from the token history."""

    # Read by the model runner; None skips the GPU bin-count bookkeeping.
    output_bin_counts = None

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        self.params[req_idx] = (
            (
                sampling_params.repetition_penalty,
                sampling_params.frequency_penalty,
                sampling_params.presence_penalty,
            )
            if use_penalty(sampling_params)
            else None
        )
        return self.params[req_idx] is not None

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        rows = self._active_rows(ctx)
        if not rows:
            return logits
        # Index the flattened logits by row * vocab + token.
        vocab_size = logits.shape[1]
        flat = logits.view(-1)
        penalties = torch.tensor([(1.0, 0.0, 0.0)] * logits.shape[0])
        prompt, output = [], []
        for row, req in rows:
            penalties[row] = torch.tensor(self.params[req])
            prompt_len = int(self.req_states.prompt_len.np[req])
            offset = row * vocab_size
            prompt.append(self.req_states.all_token_ids.gpu[req, :prompt_len] + offset)
            output.append(self._output_tokens(req, row, ctx) + offset)
        output_ids, counts = torch.cat(output).long().unique(return_counts=True)
        seen = torch.cat((torch.cat(prompt).long(), output_ids)).unique()

        rep = penalties[seen // vocab_size, 0]
        x = flat[seen]
        flat[seen] = x * torch.where(x > 0, 1.0 / rep, rep)
        freq, pres = penalties[output_ids // vocab_size, 1:].unbind(1)
        flat[output_ids] = flat[output_ids] - freq * counts - pres
        return logits


class CPUBadWords(_CPULogitsProcessor):
    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        bad_words = sampling_params.bad_words_token_ids
        self.params[req_idx] = bad_words or None
        return bool(bad_words)

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        for row, req in self._active_rows(ctx):
            history = self._output_tokens(req, row, ctx).tolist()
            for *prefix, last in self.params[req]:
                if len(prefix) <= len(history) and (
                    not prefix or history[-len(prefix) :] == prefix
                ):
                    logits[row, last] = float("-inf")
        return logits
