# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in full-history n-gram blocking for Ming ASR on Model Runner V2."""

from numbers import Integral

from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsProcessor


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    @classmethod
    def validate_params(cls, sampling_params):
        size = (sampling_params.extra_args or {}).get("no_repeat_ngram_size", 0)
        if (
            isinstance(size, bool)
            or not isinstance(size, Integral)
            or not 0 <= int(size) < 2**31
        ):
            raise ValueError("no_repeat_ngram_size must be an integer in [0, 2**31)")

    def __init__(self, vllm_config, req_states):
        import torch

        from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor

        if vllm_config.speculative_config is not None:
            raise ValueError(
                "NoRepeatNGramLogitsProcessor does not support speculation"
            )
        self.req_states = req_states
        self.sizes = UvaBackedTensor(req_states.max_num_reqs, dtype=torch.int32)

    def add_request(self, req_idx, sampling_params):
        self.validate_params(sampling_params)
        self.sizes.np[req_idx] = int(
            (sampling_params.extra_args or {}).get("no_repeat_ngram_size", 0)
        )
        return bool(self.sizes.np[req_idx] > 0)

    def apply_staged_writes(self):
        self.sizes.copy_to_uva()

    def apply(self, logits, ctx):
        import torch

        sizes = set(self.sizes.np[ctx.idx_mapping_np].tolist())
        if sizes == {0}:
            return logits
        slots = ctx.expanded_idx_mapping.long()
        history = self.req_states.all_token_ids.gpu.index_select(0, slots)
        lengths = ctx.pos + 1
        for size in sizes:
            if size == 0 or size > history.shape[1]:
                continue
            windows = history.unfold(1, size, 1)
            valid = (
                torch.arange(windows.shape[1], device=logits.device)[None, :]
                <= lengths[:, None] - size
            )
            valid &= self.sizes.gpu[slots, None] == size
            if size > 1:
                positions = (
                    lengths[:, None]
                    - size
                    + 1
                    + torch.arange(size - 1, device=logits.device)[None, :]
                )
                prefix = history.gather(1, positions.clamp(min=0).long())
                valid &= (windows[:, :, :-1] == prefix[:, None, :]).all(dim=2)
            token_ids = windows[:, :, -1].long()
            # Input embeddings may have a larger vocabulary than output logits.
            valid &= (token_ids >= 0) & (token_ids < logits.shape[1])
            penalties = torch.where(valid, -float("inf"), 0.0)
            logits.scatter_add_(1, token_ids.clamp(0, logits.shape[1] - 1), penalties)

        return logits
