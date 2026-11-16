# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from transformers import PreTrainedConfig

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator


class DSpineSpeculator(DFlash2Speculator):
    _speculator_name = "DSpine"
    _candidate_top_k_key = "transition_top_k"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        assert vllm_config.speculative_config is not None
        draft_config = vllm_config.speculative_config.draft_model_config.hf_config
        _maybe_override_dspine_config(draft_config)
        super().__init__(vllm_config, device)

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


def _maybe_override_dspine_config(config: PreTrainedConfig) -> None:
    """Map the original DSpine checkpoint's settings to the `dflash_config`
    keys that Speculators checkpoints use (see `update_dspine`)."""
    draft = config.dflash_config
    bridge = draft.get("selected_prefix_code_bridge")
    if bridge is None:
        return
    expected = {
        "draft_block_causal": True,
        "selected_prefix_terminal": False,
        "selected_prefix_feature_writes": True,
        "selected_prefix_feature_write_mode": "independent",
        "selected_prefix_feature_layers": list(range(1, config.num_hidden_layers + 1)),
    }
    if (
        getattr(config, "is_causal", True) is False
        or any(draft.get(key) != value for key, value in expected.items())
        or draft.get("selected_prefix_feature_next_token_rows", False)
        or draft.get("selected_prefix_feature_reliability") is not None
    ):
        raise ValueError("DSpine requires independent causal writes at every layer.")
    second = getattr(config, "code_bridge_second_pass", {})
    refine = draft.get("selected_prefix_last_write_refine", {})
    if (
        second.get("mode") != "serial_table"
        or second.get("first_exact") != config.num_hidden_layers - 1
        or refine.get("kind") != "exact_sender"
        or second.get("head_topk") != refine.get("head_topk")
    ):
        raise ValueError(
            "DSpine requires matching serial last-write refinement settings."
        )
    draft.update(
        block_size=config.block_size,
        transfer_dim=bridge["code_size"],
        message_dim=bridge["rank"],
        transition_top_k=second["head_topk"],
        transfer_space_eps=bridge["eps"],
        causal=True,
        sample_from_anchor=False,
    )
