# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from vllm.compilation.backends import set_model_tag
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.config.kv_transfer import is_hs_store_cpu
from vllm.forward_context import set_forward_context
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.utils import get_draft_load_config
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.spec_decode.utils import get_pp_safe_draft_load_config

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.block_table import BlockTables
    from vllm.v1.worker.gpu.states import RequestState


@dataclass
class _HsReqScope:
    """Per-request inputs the hidden-states CPU block table needs.

    Mirrors the attributes of ``Request`` that the connector inspects
    (page capacity and BlockPool block ids) so the V2 runner can feed it
    without holding scheduler ``Request`` objects.
    """

    num_prompt_tokens: int
    # BlockPool block ids of the request so far (group 0: the MLA /
    # full-attention pool); refreshed from the runner's block tables before
    # each streamed step.
    block_ids: list[int]


class ExtractHiddenStatesSpeculator(DraftModelSpeculator):
    """Cache target hidden states while returning always-accepted draft tokens."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        req_states: "RequestState | None" = None,
    ):
        assert vllm_config.speculative_config is not None
        if vllm_config.speculative_config.draft_sample_method != "greedy":
            raise ValueError(
                "extract_hidden_states only supports draft_sample_method='greedy'"
            )
        super().__init__(vllm_config, device)

        if self.num_speculative_steps != 1:
            raise ValueError(
                "extract_hidden_states requires num_speculative_tokens to be 1"
            )

        self.supports_mm_inputs = False
        # True when the KV connector streams hidden states to per-request
        # CPU pages (hs_store_device=cpu). This speculator has no
        # CPU-page streaming path; `propose` fails loudly in that case
        # instead of silently producing empty saved files.
        self._hs_store_cpu = is_hs_store_cpu(vllm_config)
        layer_ids = getattr(
            self.draft_model_config.hf_config,
            "eagle_aux_hidden_state_layer_ids",
            None,
        )
        if not layer_ids:
            raise ValueError(
                "eagle_aux_hidden_state_layer_ids must be set in the draft "
                "model config for extract_hidden_states method"
            )

        self.num_hidden_states = len(layer_ids)
        assert isinstance(self.dtype, torch.dtype)
        self.hidden_states = torch.zeros(
            self.max_num_tokens,
            self.num_hidden_states,
            self.vllm_config.model_config.get_hidden_size(),
            dtype=self.dtype,
            device=device,
        )
        # hs_store_device=cpu: hidden states are streamed to per-request CPU
        # pages instead of a device cache-only KV group.
        self._hs_store_cpu = is_hs_store_cpu(vllm_config)
        self._hs_scope: dict[str, _HsReqScope] = {}
        # Injected by the model runner (init_speculator); used to look up each
        # request's block-table row for the CPU block table.
        self.req_states = req_states
        # Injected by the model runner after KV init (alongside
        # pcp_manager); the per-request BlockPool block ids are refreshed
        # from it before each streamed step.
        self.block_tables: "BlockTables | None" = None

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        del target_model, target_attn_layer_names
        with set_model_tag("extract_hidden_states"):
            return get_model(
                vllm_config=self.vllm_config,
                model_config=self.draft_model_config,
                load_config=get_pp_safe_draft_load_config(
                    get_draft_load_config(self.vllm_config)
                ),
            )

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        # Hidden states are only token-determined for text-only requests; MM
        # inputs are excluded from the CPU prefix index (see connector).
        self.supports_mm_inputs = False
        if len(self.draft_attn_layer_names) != 1:
            raise ValueError(
                "ExtractHiddenStatesModel should have exactly one attention "
                f"layer, found {len(self.draft_attn_layer_names)}"
            )

    def on_new_request(self, req_id: str, new_req_data: Any) -> None:
        """Record the CPU-pages scope for a newly scheduled request."""
        if not self._hs_store_cpu:
            return
        self._hs_scope[req_id] = _HsReqScope(
            num_prompt_tokens=int(getattr(new_req_data, "prompt_len", 0) or 0),
            block_ids=[],
        )

    def _refresh_hs_scope(self, input_batch: InputBatch) -> None:
        """Mirror each batched request's BlockPool block ids into its scope.

        The connector's CPU block table is addressed by the same block ids
        as the device KV, so the ids of every block the request owns so far
        must be current at stream time (new blocks extend the scatter range).
        """
        block_tables = self.block_tables
        if block_tables is None or not block_tables.block_tables:
            return
        if self.req_states is None:
            return
        for req_id in input_batch.req_ids[: input_batch.num_reqs]:
            scope = self._hs_scope.get(req_id)
            if scope is None:
                continue
            row = self.req_states.req_id_to_index.get(req_id)
            if row is None:
                continue
            # Group 0 is the MLA / full-attention pool whose block ids the
            # connector's CPU block table is addressed by.
            num_blocks = int(block_tables.num_blocks.np[0, row])
            scope.block_ids = (
                block_tables.block_tables[0].gpu[row, :num_blocks].tolist()
            )

    def on_requests_finished(self, finished_req_ids: set[str]) -> None:
        for req_id in finished_req_ids:
            self._hs_scope.pop(req_id, None)

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        del cudagraph_mode

    def capture(self) -> None:
        return None

    def _stream_hidden_states_to_cpu(
        self, input_batch: InputBatch, num_tokens: int, is_profile: bool = False
    ) -> None:
        """Copy this step's rows into the connector's per-request CPU pages."""
        from vllm.distributed.kv_transfer import (
            get_kv_transfer_group,
            has_kv_transfer_group,
        )

        if not has_kv_transfer_group():
            # The profiling / warmup steps run before the KV transfer group
            # is created and their outputs are discarded anyway: skip them
            # silently. Any other step without an active group means the
            # runner disabled the connector, so streaming silently would
            # produce empty saved files - fail loudly instead (mirrors the
            # V1 proposer's guards).
            if is_profile:
                return
            raise ValueError(
                "hs_store_device=cpu with the extract_hidden_states V2 "
                "model runner has no active KV transfer group, so hidden "
                "states would be silently lost. The KV connector is "
                "missing or was disabled at runtime."
            )
        connector = get_kv_transfer_group()
        save_fn = getattr(connector, "save_hs_batch_tokens", None)
        if save_fn is None:
            raise ValueError(
                "hs_store_device=cpu requires a KV connector that "
                "implements save_hs_batch_tokens (e.g. "
                f"ExampleHiddenStatesConnector), got {type(connector).__name__}"
            )
        num_reqs = input_batch.num_reqs
        save_fn(
            self.hidden_states[:num_tokens],
            list(input_batch.req_ids[:num_reqs]),
            input_batch.query_start_loc_np[: num_reqs + 1],
            input_batch.seq_lens[:num_reqs],
            num_tokens,
            requests=self._hs_scope,
        )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync_state: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        del (
            last_hidden_states,
            num_sampled,
            num_rejected,
            next_prefill_tokens,
            temperature,
            seeds,
            mm_inputs,
        )

        draft_tokens = last_sampled[input_batch.idx_mapping, :1]
        if skip_attn_for_dummy_run:
            return draft_tokens
        if aux_hidden_states is None:
            raise ValueError(
                "aux_hidden_states are required when using extract_hidden_states"
            )
        if len(aux_hidden_states) != self.num_hidden_states:
            raise ValueError(
                f"Expected {self.num_hidden_states} auxiliary hidden states, "
                f"got {len(aux_hidden_states)}"
            )

        stacked_hidden_states = torch.stack(aux_hidden_states, dim=1)
        num_tokens = stacked_hidden_states.shape[0]
        self.hidden_states[:num_tokens].copy_(stacked_hidden_states)

        if self._hs_store_cpu:
            # CPU-pages mode: stream this step's rows out and skip the
            # cache-only draft forward entirely (the layer has no device KV
            # group to write into; the drafter adds nothing else). Dummy runs
            # must not touch the connector (it is disabled while profiling).
            if not dummy_run:
                self._refresh_hs_scope(input_batch)
                self._stream_hidden_states_to_cpu(input_batch, num_tokens, is_profile)
            return draft_tokens

        attn_metadata, slot_mappings = self._build_drafter_attn_metadata(
            input_batch, attn_metadata, slot_mappings
        )

        draft_attn_metadata = {
            name: attn_metadata[name] for name in self.draft_attn_layer_names
        }
        draft_slot_mappings = {
            name: slot_mappings[name][:num_tokens]
            for name in self.draft_attn_layer_names
        }
        with set_forward_context(
            draft_attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=dp_sync_state and dp_sync_state.num_tokens_across_dp,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=draft_slot_mappings,
            is_padding=input_batch.is_padding[:num_tokens],
        ):
            self.model(hidden_states=self.hidden_states[:num_tokens])

        return draft_tokens
