# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn

from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
    TargetDependentARSpeculator,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backends.flashinfer import FlashInferImpl

logger = init_logger(__name__)


class MTPSpeculator(TargetDependentARSpeculator):
    share_mtp_topk_indices: bool = False
    # Attention impl that computes only the sampled draft prefill rows
    # (SpeculativeConfig.prune_draft_prefill_attention), or None.
    draft_prefill_attn_impl: "FlashInferImpl | None" = None

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
            self.vllm_config.parallel_config.prefill_context_parallel_size == 1
            and getattr(draft_hf_config, "index_share_for_mtp_iteration", False)
            and hasattr(draft_model.model, "set_skip_topk")
            and hasattr(draft_model.model, "compact_topk_indices")
        )
        if spec_config is not None and spec_config.prune_draft_prefill_attention:
            self.draft_prefill_attn_impl = self._get_draft_prefill_attn_impl(
                draft_model
            )
        return draft_model

    def _get_draft_prefill_attn_impl(
        self, draft_model: nn.Module
    ) -> "FlashInferImpl | None":
        """The draft's attention impl if its prefill can attend only the sampled
        rows, else None.

        The draft prefill samples one row per request. If its only stateful
        layer is one attention layer, that layer's KV comes from its input,
        not from attention outputs, so the other rows' attention outputs are
        never used. A deeper draft would feed them into a later layer's KV or
        state.
        """
        layers = [m for m in draft_model.modules() if isinstance(m, AttentionLayerBase)]
        layer = layers[0] if len(layers) == 1 else None
        reason: str | None
        if self.vllm_config.parallel_config.prefill_context_parallel_size > 1:
            # The draft prefill is re-laid out across ranks, so
            # last_token_indices does not index the attention's rows.
            reason = "prefill context parallelism"
        elif not isinstance(layer, Attention) or not hasattr(
            layer.impl, "draft_prefill_rows_unsupported"
        ):
            reason = "the draft needs exactly one FlashInfer attention layer"
        else:
            impl = cast("FlashInferImpl", layer.impl)
            reason = impl.draft_prefill_rows_unsupported()
            if reason is None:
                logger.info("MTP draft prefill attention pruning enabled")
                return impl
        logger.warning("prune_draft_prefill_attention is ignored: %s", reason)
        return None

    def _prefill(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        impl = self.draft_prefill_attn_impl
        if (
            impl is not None
            and cudagraph_runtime_mode != CUDAGraphMode.FULL
            and not torch.cuda.is_current_stream_capturing()
        ):
            from vllm.v1.attention.backends.flashinfer import DraftPrefillRows

            rows = self.last_token_indices[:num_reqs]
            impl.draft_prefill_rows = DraftPrefillRows(
                rows=rows,
                seq_lens=(self.input_buffers.positions[rows] + 1).to(torch.int32),
                max_seq_len=self.draft_max_seq_len,
            )
        try:
            super()._prefill(
                num_reqs,
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
                mm_inputs,
            )
        finally:
            if impl is not None:
                impl.draft_prefill_rows = None

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
