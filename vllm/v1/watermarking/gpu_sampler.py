# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.watermarking.watermarker import Watermarker
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.sampler import Sampler


class GPUWatermarkSampler(Sampler):
    def __init__(
        self,
        watermarker: Watermarker,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.watermarker = watermarker

    def _sample_random(
        self,
        processed_logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        _idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        top_k: torch.Tensor | None,
        top_p: torch.Tensor | None,
        _use_flashinfer: bool,
        _return_logprobs: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        processed_logits = apply_top_k_top_p(processed_logits, top_k, top_p)
        contexts = self._get_contexts(expanded_idx_mapping)
        output = self.watermarker.sample(
            processed_logits,
            contexts,
            lambda sample_logits: gumbel_sample(
                sample_logits,
                expanded_idx_mapping,
                self.sampling_states.temperature.gpu,
                self.sampling_states.seeds.gpu,
                pos,
                apply_temperature=False,
                use_fp64=self.use_fp64_gumbel,
            ),
        )
        temperatures = self.sampling_states.temperature.gpu[expanded_idx_mapping]
        sampled = torch.where(
            temperatures == 0,
            processed_logits.argmax(dim=-1),
            output.token_ids,
        )
        return sampled, output.logits

    def _get_contexts(self, expanded_idx_mapping: torch.Tensor) -> torch.Tensor:
        context_width = self.watermarker.context_width
        req_indices = expanded_idx_mapping.to(torch.int64)
        valid_reqs = req_indices >= 0
        safe_req_indices = req_indices.clamp_min(0)
        total_lens = self.req_states.total_len.gpu[safe_req_indices].to(torch.int64)
        prompt_lens = self.req_states.prompt_len.gpu[safe_req_indices].to(torch.int64)
        offsets = torch.arange(
            -context_width, 0, dtype=torch.int64, device=req_indices.device
        )
        positions = total_lens.unsqueeze(-1) + offsets
        valid_positions = valid_reqs.unsqueeze(-1) & (
            positions >= prompt_lens.unsqueeze(-1)
        )
        positions = positions.clamp_min(0)
        contexts = self.req_states.all_token_ids.gpu[
            safe_req_indices.unsqueeze(-1), positions
        ]
        return torch.where(valid_positions, contexts, -1)
