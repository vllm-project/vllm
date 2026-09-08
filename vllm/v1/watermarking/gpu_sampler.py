# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.watermarking.watermarker import RandomSampler, Watermarker
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.watermark import repeated_context_mask

logger = init_logger(__name__)


class GPUWatermarkSampler(Sampler):
    def __init__(
        self,
        watermarker: Watermarker,
        *args,
        deduplicate_contexts: Literal["none", "single_turn", "all"] = "single_turn",
        deduplicate_contexts_max_history: int | None = 8192,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.watermarker = watermarker
        self.deduplicate_contexts = deduplicate_contexts
        self.deduplicate_contexts_max_history = deduplicate_contexts_max_history
        self.watermarking = UvaBackedTensor(
            self.sampling_states.max_num_reqs, dtype=torch.bool
        )
        self.watermarking.np.fill(True)
        self.watermarking.copy_to_uva()

    def add_request(
        self, req_idx: int, prompt_len: int, sampling_params: SamplingParams
    ) -> None:
        super().add_request(req_idx, prompt_len, sampling_params)
        self.watermarking.np[req_idx] = sampling_params.watermarking
        if sampling_params.watermarking and sampling_params.temperature == 0:
            logger.warning_once(
                "Watermarking is enabled, but greedy decoding "
                "(temperature=0) cannot be watermarked. This request will use "
                "ordinary greedy sampling."
            )

    def apply_staged_writes(self) -> None:
        super().apply_staged_writes()
        self.watermarking.copy_to_uva()

    def _sample_random(
        self,
        processed_logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        top_k: torch.Tensor | None,
        top_p: torch.Tensor | None,
        use_flashinfer: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        enabled = self.watermarking.np[idx_mapping_np] & (
            self.sampling_states.temperature.np[idx_mapping_np] != 0
        )
        if not np.any(enabled):
            return super()._sample_random(
                processed_logits,
                expanded_idx_mapping,
                idx_mapping_np,
                pos,
                top_k,
                top_p,
                use_flashinfer,
            )

        processed_logits = apply_top_k_top_p(processed_logits, top_k, top_p)
        contexts = self._get_contexts(expanded_idx_mapping)
        repeated_contexts = None
        if self.deduplicate_contexts != "none":
            repeated_contexts = self._get_repeated_contexts(
                expanded_idx_mapping, contexts
            )

        temperatures = self.sampling_states.temperature.gpu[expanded_idx_mapping]
        needs_mixed_sampling = repeated_contexts is not None or not np.all(enabled)
        skip_mask = None
        random_sampler = None
        if needs_mixed_sampling:
            watermarking = self.watermarking.gpu[expanded_idx_mapping] & (
                temperatures != 0
            )
            if repeated_contexts is not None:
                watermarking &= ~repeated_contexts
            skip_mask = ~watermarking

            random_sampler = RandomSampler(
                expanded_idx_mapping=expanded_idx_mapping,
                temperatures=self.sampling_states.temperature.gpu,
                seeds=self.sampling_states.seeds.gpu,
                positions=pos,
                use_fp64=self.use_fp64_gumbel,
            )
        output = self.watermarker.sample(
            processed_logits,
            contexts,
            random_sampler,
            skip_mask=skip_mask,
        )
        sampled = output.token_ids
        output_logits = output.logits
        sampled = torch.where(
            temperatures == 0,
            processed_logits.argmax(dim=-1),
            sampled,
        )
        return sampled, output_logits

    def _get_repeated_contexts(
        self,
        expanded_idx_mapping: torch.Tensor,
        contexts: torch.Tensor,
    ) -> torch.Tensor:
        return repeated_context_mask(
            self.req_states.all_token_ids.gpu,
            expanded_idx_mapping,
            self.req_states.prompt_len.gpu,
            self.req_states.total_len.gpu,
            contexts,
            self.deduplicate_contexts_max_history,
            include_prompt=self.deduplicate_contexts == "all",
            skip_partial_context=self.deduplicate_contexts == "all",
        )

    def _get_contexts(
        self,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor | None = None,
        draft_sampled: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build one watermark context for each flattened verification row.

        For each verification row:

        1. Add its request-local speculative position to ``total_len`` to get
           the absolute position being sampled.
        2. Read context positions below ``total_len`` from committed output.
        3. Read later context positions from preceding draft rows.

        For example, with committed output ``[10, 11]``, proposals ``[20, 21]``,
        and context width 2, local positions ``[0, 1, 2]`` sample after prefixes
        ``[10, 11]``, ``[10, 11, 20]``, and ``[10, 11, 20, 21]``. Their contexts
        are therefore ``[10, 11]``, ``[11, 20]``, and ``[20, 21]``.

        ``draft_sampled`` has the same flattened, request-chunked layout as the
        verification logits. Its first row per request precedes the first draft
        token, so the draft index includes a one-row offset.
        """
        context_width = self.watermarker.context_width
        req_indices = expanded_idx_mapping.to(torch.int64)
        valid_reqs = req_indices >= 0
        safe_req_indices = req_indices.clamp_min(0)
        total_lens = self.req_states.total_len.gpu[safe_req_indices].to(torch.int64)
        prompt_lens = self.req_states.prompt_len.gpu[safe_req_indices].to(torch.int64)
        if expanded_local_pos is None:
            expanded_local_pos = torch.zeros_like(req_indices)
        else:
            expanded_local_pos = expanded_local_pos.to(torch.int64)
        offsets = torch.arange(
            -context_width, 0, dtype=torch.int64, device=req_indices.device
        )
        positions = (
            total_lens.unsqueeze(-1) + expanded_local_pos.unsqueeze(-1) + offsets
        )
        committed = positions < total_lens.unsqueeze(-1)
        valid_committed = (
            valid_reqs.unsqueeze(-1)
            & committed
            & (positions >= prompt_lens.unsqueeze(-1))
        )
        committed_contexts = self.req_states.all_token_ids.gpu[
            safe_req_indices.unsqueeze(-1),
            positions.clamp(0, self.req_states.all_token_ids.gpu.shape[1] - 1),
        ]
        contexts = torch.where(valid_committed, committed_contexts, -1)
        if draft_sampled is None:
            return contexts

        row_indices = torch.arange(
            len(req_indices), dtype=torch.int64, device=req_indices.device
        )
        draft_offsets = positions - total_lens.unsqueeze(-1)
        draft_indices = (
            row_indices.unsqueeze(-1)
            - expanded_local_pos.unsqueeze(-1)
            + draft_offsets
            + 1
        )
        valid_drafts = (
            valid_reqs.unsqueeze(-1)
            & ~committed
            & (draft_offsets < expanded_local_pos.unsqueeze(-1))
        )
        draft_contexts = draft_sampled[draft_indices.clamp(0, len(draft_sampled) - 1)]
        return torch.where(valid_drafts, draft_contexts, contexts)
