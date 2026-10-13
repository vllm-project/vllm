# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.cpu.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import SamplingStates


class CPUSamplingStates(SamplingStates):
    def apply_temperature(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        temp_np = self.temperature.np[idx_mapping_np]
        if np.all((temp_np == 0.0) | (temp_np == 1.0)):
            return

        temperature = self.temperature.gpu[expanded_idx_mapping].unsqueeze(1)
        logits.div_(torch.where(temperature == 0.0, 1.0, temperature))

    def apply_min_p(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        if np.all(self.min_p.np[idx_mapping_np] == 0.0):
            return

        min_p = self.min_p.gpu[expanded_idx_mapping].unsqueeze(1)
        threshold = logits.amax(dim=1, keepdim=True) + min_p.log()
        logits.masked_fill_((min_p != 0.0) & (logits < threshold), float("-inf"))


class CPUSampler(Sampler):
    """CPU extension point for Model Runner V2 sampling."""

    sampling_states_cls = CPUSamplingStates

    def _sample_random(
        self,
        processed_logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        top_k: torch.Tensor | None,
        top_p: torch.Tensor | None,
        use_fused_sampler: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert not use_fused_sampler

        # TODO (fadara01): Add C++ fused ops for top-p / top-k sampling.
        processed_logits = apply_top_k_top_p(processed_logits, top_k, top_p)

        sampled = gumbel_sample(
            processed_logits,
            expanded_idx_mapping,
            self.sampling_states.temperature.gpu,
            self.sampling_states.seeds.gpu,
            pos,
            apply_temperature=False,
            is_drafting=False,
            use_fp64=self.use_fp64_gumbel,
        )
        return sampled, processed_logits
