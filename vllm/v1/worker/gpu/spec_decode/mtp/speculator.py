# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn

from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model


class MTPSpeculator(AutoRegressiveSpeculator):
    share_mtp_topk_indices: bool = False

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        draft_model = load_eagle_model(target_model, self.vllm_config)
        spec_config = self.vllm_config.speculative_config
        draft_hf_config = (
            spec_config.draft_model_config.hf_config
            if spec_config is not None
            else None
        )
        # Detect index_share_for_mtp_iteration. When True, the proposer
        # toggles skip_topk so step 0 computes MTP's own indices and
        # steps 1+ reuse them.
        self.share_mtp_topk_indices = (
            getattr(draft_hf_config, "index_share_for_mtp_iteration", False)
            and hasattr(draft_model.model, "set_skip_topk")
            and hasattr(draft_model.model, "compact_topk_indices")
        )
        return draft_model

    def on_prefill_begin(self, num_reqs: int) -> None:
        # Step 0 computes its own top-k. Unconditional, so a step that died
        # midway cannot leave reuse mode on.
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(False)

    def on_prefill_end(self, num_reqs: int) -> None:
        # Step 0 (prefill) wrote topk indices for every query token in the
        # multi-token batch. Compact them down to each request's last token so
        # steps 1+ can reuse them from the shared buffer.
        if self.share_mtp_topk_indices and self.num_speculative_steps > 1:
            self.model.model.compact_topk_indices(self.last_token_indices[:num_reqs])

    def on_multi_step_decode_begin(self, num_reqs: int) -> None:
        # Switch to reuse mode so draft steps 1+ skip the indexer op and read
        # the indices that step 0 wrote into the shared buffer.
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(True)

    def on_multi_step_decode_end(self, num_reqs: int) -> None:
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(False)
