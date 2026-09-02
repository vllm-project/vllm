# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.watermarking.watermarker import Watermarker
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
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
        return_logprobs: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        enabled = self.watermarking.np[idx_mapping_np]
        if not np.any(enabled):
            return super()._sample_random(
                processed_logits,
                expanded_idx_mapping,
                idx_mapping_np,
                pos,
                top_k,
                top_p,
                use_flashinfer,
                return_logprobs,
            )

        processed_logits = apply_top_k_top_p(processed_logits, top_k, top_p)
        contexts = self._get_contexts(expanded_idx_mapping)

        def random_sample(sample_logits: torch.Tensor) -> torch.Tensor:
            return gumbel_sample(
                sample_logits,
                expanded_idx_mapping,
                self.sampling_states.temperature.gpu,
                self.sampling_states.seeds.gpu,
                pos,
                apply_temperature=False,
                use_fp64=self.use_fp64_gumbel,
            )

        output = self.watermarker.sample(
            processed_logits,
            contexts,
            random_sample,
        )
        temperatures = self.sampling_states.temperature.gpu[expanded_idx_mapping]
        watermarking = self.watermarking.gpu[expanded_idx_mapping]
        if not np.all(enabled):
            unwatermarked = random_sample(processed_logits)
            sampled = torch.where(watermarking, output.token_ids, unwatermarked)
            output_logits = output.logits
            if output.logits is not processed_logits:
                output_logits = torch.where(
                    watermarking.unsqueeze(-1), output.logits, processed_logits
                )
        else:
            sampled = output.token_ids
            output_logits = output.logits
        sampled = torch.where(
            temperatures == 0,
            processed_logits.argmax(dim=-1),
            sampled,
        )
        return sampled, output_logits

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
