# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.cudagraph_dispatcher import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

logger = init_logger(__name__)


class XPressSpeculator(DFlashSpeculator):
    _speculator_name = "XPress"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        hf = self.draft_model_config.hf_config
        self.num_query_per_req = 1 + self.num_speculative_steps
        import os

        env_passes = os.environ.get("XPRESS_NUM_PASSES")
        self.num_jacobi_passes = (
            int(env_passes) if env_passes else int(getattr(hf, "xpress_num_passes", 6))
        )
        logger.info("XPress: K=%d Jacobi passes", self.num_jacobi_passes)
        # Block slot 0 holds the anchor -- a verified token. Its refined output is
        # discarded, so only its latent matters, and that reaches the draft slots
        # through the single mixer column L[:, k, 0].
        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )
        # XPress consumes the DRAFTER's own hidden width; the base class widens
        # self.hidden_size for HC-multiplexed models, which this is not.
        draft_hidden = self.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            self.max_num_tokens, draft_hidden, dtype=self.dtype, device=device
        )

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        # The DSpark loader body is architecture-agnostic (embed/lm_head sharing,
        # non-causal attention config, KV setup), so it is reused as-is.
        return load_dspark_model(target_model, self.vllm_config)

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        # Backbone forward over all query blocks, then K Jacobi passes over each
        # request's block. Both stages are fixed-shape and loop-free per pass, so the
        # whole draft step is CUDA-graph capturable -- K passes cost K matmuls, not K
        # kernel launches.
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self._jacobi_refine(num_reqs, head_hidden)

    def _jacobi_refine(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        B = self.num_query_per_req  # 1 anchor + N draft slots
        n_rows = num_reqs * B
        h_full = head_hidden[:n_rows].view(num_reqs, B, -1)
        base_full = self.model.compute_draft_logits(h_full)
        anchor_ids = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        # The anchor's own predecessor lives in the KV cache, not in any id buffer the
        # speculator can reach, so the anchor id stands in for it (see __init__).
        tok_am1 = anchor_ids
        head = self.model.model.xpress_head
        draft = head.jacobi_refine_greedy(
            base_full, h_full, anchor_ids, tok_am1, self.num_jacobi_passes
        )
        self.draft_tokens[:num_reqs, : self.num_speculative_steps] = draft
