# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.v1.spec_decode.eagle import EagleProposer


class CapturingEagleProposer(EagleProposer):
    """EagleProposer that captures target hidden states per-request.

    When a KV transfer connector with ``capture_hidden_states`` is active,
    this proposer intercepts the target hidden states before each draft
    round and forwards them to the connector for async disk write.
    """

    def __init__(self, vllm_config, device: torch.device, runner=None):
        super().__init__(vllm_config, device, runner)
        self._connector = None
        self._connector_resolved = False

    def _resolve_connector(self):
        """Lazily resolve the KV transfer connector (once)."""
        if self._connector_resolved:
            return self._connector
        self._connector_resolved = True
        from vllm.distributed.kv_transfer.kv_transfer_state import (
            get_kv_transfer_group,
            has_kv_transfer_group,
        )
        if has_kv_transfer_group():
            connector = get_kv_transfer_group()
            if hasattr(connector, "capture_hidden_states"):
                self._connector = connector
        return self._connector

    def propose(
        self,
        *,
        target_hidden_states: torch.Tensor,
        target_token_ids: torch.Tensor,
        common_attn_metadata,
        **kwargs,
    ) -> torch.Tensor:
        connector = self._resolve_connector()
        if connector is not None:
            batch = self.runner.input_batch
            num_reqs = len(batch.req_ids)
            connector.capture_hidden_states(
                hidden_states=target_hidden_states,
                token_ids=target_token_ids,
                query_start_loc=common_attn_metadata.query_start_loc_cpu,
                req_ids=batch.req_ids[:num_reqs],
                lora_mapping=batch.request_lora_mapping,
                lora_lookup=batch.lora_id_to_lora_request,
            )
        return super().propose(
            target_hidden_states=target_hidden_states,
            target_token_ids=target_token_ids,
            common_attn_metadata=common_attn_metadata,
            **kwargs,
        )
