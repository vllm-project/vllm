# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator


class DSpineSpeculator(DFlash2Speculator):
    _speculator_name = "DSpine"
    _candidate_top_k_key = "transition_top_k"

    def __init__(self, vllm_config, device: torch.device):
        super().__init__(vllm_config, device)
        if self.enable_adaptive_verification:
            raise ValueError("DSpine does not support adaptive verification.")

    def _prepare_candidates(self, model_output, num_reqs):
        hidden_states, final_pre_injection = model_output
        num_sample = num_reqs * self.num_speculative_steps
        sample_indices = self.sample_indices[:num_sample]
        candidate_ids, _ = self.model.compute_candidates(hidden_states[sample_indices])
        final_pre_injection = final_pre_injection[sample_indices].view(
            num_reqs, self.num_speculative_steps, -1
        )
        candidate_ids = candidate_ids.view(
            num_reqs, self.num_speculative_steps, self.top_k
        )
        anchor_token_ids = self.input_buffers.input_ids[
            : num_reqs * self.num_query_per_req : self.num_query_per_req
        ]
        scores = self.model.compute_transition_scores(
            candidate_ids, final_pre_injection, anchor_token_ids
        )
        return candidate_ids, scores
