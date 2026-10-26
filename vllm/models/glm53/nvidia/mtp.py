# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import typing

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.utils import (
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)
from vllm.models.deepseek_v32.nvidia.glm52_low_latency_gemm import (
    enable_glm52_low_latency_gemm,
)
from vllm.models.deepseek_v32.nvidia.mtp import (
    DeepseekV32MTP,
    DeepseekV32MultiTokenPredictor,
)

from . import draft as glm53_draft
from . import switch
from .ops.launch import Glm53Geometry

logger = init_logger(__name__)


class Glm53MultiTokenPredictor(DeepseekV32MultiTokenPredictor):
    """The MTP predictor of vLLM whose forwards the installed drafter
    segment may claim.
    """

    # The installed drafter segment, or None for stock vLLM.
    glm53_segment: typing.Any = None

    def set_skip_topk(self, skip: bool):
        segment = self.glm53_segment
        if segment is not None:
            segment.decode_phase(skip)
        super().set_skip_topk(skip)

    def compact_topk_indices(self, slot_ids: torch.Tensor):
        """Gather the top-k index rows at ``slot_ids`` to the front of the buffer."""
        segment = self.glm53_segment
        if segment is not None:
            segment.observe_slots(slot_ids)
            if segment.claims_proposals():
                return
        super().compact_topk_indices(slot_ids)

    def get_top_tokens(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor:
        """The claimed forward's greedy drafts, else vLLM's local-argmax drafts."""
        segment = self.glm53_segment
        tokens = (
            None if segment is None else segment.claimed_tokens(hidden_states.shape[0])
        )
        if tokens is not None:
            return tokens
        return super().get_top_tokens(hidden_states, spec_step_idx)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        segment = self.glm53_segment
        if segment is None or input_ids is None or inputs_embeds is not None:
            return super().forward(
                input_ids,
                positions,
                previous_hidden_states,
                inputs_embeds,
                spec_step_idx,
            )
        claimed = segment.forward(input_ids, positions, previous_hidden_states)
        if claimed is None:
            out = super().forward(
                input_ids,
                positions,
                previous_hidden_states,
                inputs_embeds,
                spec_step_idx,
            )
            segment.after_stock_forward(input_ids, positions, previous_hidden_states)
            return out
        layer = self.layers[
            str(self.mtp_start_layer_idx + spec_step_idx % self.num_mtp_layers)
        ]
        hidden_states = (
            claimed
            if getattr(segment, "normalized", False)
            else layer.shared_head.norm(claimed)
        )
        if hidden_states.shape[0] != positions.shape[0]:
            # Only the sampled row was computed; the speculator reads just that row.
            hidden_states = hidden_states.expand(positions.shape[0], -1)
        return hidden_states, hidden_states


class GlmMoeDsaMTP(DeepseekV32MTP):
    """The drafter: vLLM's DeepSeek-V3.2 MTP model with the drafter segment's hooks."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        spec_config = vllm_config.speculative_config
        self.vllm_config_speculative: int = (
            0 if spec_config is None else int(spec_config.num_speculative_tokens)
        )
        self.vllm_config_max_seqs: int = vllm_config.scheduler_config.max_num_seqs
        self.model = Glm53MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if self.config.model_type == "glm_moe_dsa":
            enable_glm52_low_latency_gemm(self, vllm_config.model_config.dtype)
        self.set_moe_parameters()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], self.config.hidden_size
        )

    def process_weights_after_loading(self) -> None:
        record = switch.current()
        if not record.enabled:
            return
        seqs = self.vllm_config_max_seqs
        if not switch.drafter_serves(True, self.vllm_config_speculative):
            logger.info_once(
                "GLM-5.3 latency-specialized decode with MTP %d: vLLM's MTP drafter "
                "serves (the CuTe drafter serves MTP %s)",
                self.vllm_config_speculative,
                switch.DRAFT_KS,
            )
            return
        tiers = switch.draft_tiers(seqs, self.vllm_config_speculative)
        logger.info_once(
            "GLM-5.3 latency-specialized decode with --max-num-seqs %d, MTP %d: "
            "the CuTe drafter serves steps of up to %d requests (tiers %s), vLLM's "
            "MTP drafter larger ones",
            seqs,
            self.vllm_config_speculative,
            max(tiers),
            tiers,
        )
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        geometry = Glm53Geometry.from_config(
            self.config, get_tensor_model_parallel_world_size()
        )
        forms = glm53_draft.draft_weights(
            self, geometry, get_tensor_model_parallel_rank()
        )
        glm53_draft.install(self, geometry, forms, self.vllm_config_speculative, tiers)


def register() -> None:
    """Serve vLLM's DeepSeek-V3.2 MTP architecture with ``GlmMoeDsaMTP`` in this
    process.
    """
    from vllm.model_executor.models.registry import ModelRegistry

    ModelRegistry.register_model("DeepseekV32MTPModel", f"{__name__}:GlmMoeDsaMTP")
