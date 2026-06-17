# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from copy import copy

import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


class OpenPanguV2MTPProposer(EagleProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config, device, runner)
        self.runner = runner
        draft_hf_config = self.draft_model_config.hf_config
        self.n_predict = int(
            getattr(draft_hf_config, "n_predict", None)
            or getattr(draft_hf_config, "num_nextn_predict_layers", 1)
            or 1
        )
        self.use_multi_mtp_heads = self.method == "mtp" and self.n_predict > 1
        self.num_mtp_prefill_heads = min(self.n_predict, self.num_speculative_tokens)
        self._per_group_block_tables: dict[int, torch.Tensor] = {}
        self._per_group_slot_mappings: dict[int, torch.Tensor] = {}
        self._per_group_slot_mapping_buffers: dict[int, torch.Tensor] = {}
        self._block_size_by_gid: dict[int, int] = {}

    def set_per_group_attn_metadata(
        self,
        gid: int,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        self._per_group_block_tables[gid] = block_table
        self._per_group_slot_mappings[gid] = slot_mapping

    def _slot_mapping_buffer_for(self, gid: int) -> torch.Tensor:
        if gid == self.kv_cache_gid:
            return self._slot_mapping_buffer
        buf = self._per_group_slot_mapping_buffers.get(gid)
        if buf is None:
            buf = torch.zeros(self.max_positions, dtype=torch.int64, device=self.device)
            self._per_group_slot_mapping_buffers[gid] = buf
        return buf

    def _get_slot_mapping(
        self,
        num_tokens: int,
        slot_mapping: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Per-layer slot_mapping with one buffer per KV cache group."""
        per_layer: dict[str, torch.Tensor] = {}
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            buffer = self._slot_mapping_buffer_for(gid)
            source = self._per_group_slot_mappings.get(gid, slot_mapping)
            if source is not None and buffer.data_ptr() != source.data_ptr():
                n = source.shape[0]
                buffer[:n].copy_(source)
                if num_tokens > n:
                    buffer[n:num_tokens].fill_(PADDING_SLOT_ID)
            view = buffer[:num_tokens]
            for layer_name in attn_group.layer_names:
                per_layer[layer_name] = view
        return per_layer

    def _compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor:
        if self.method == "mtp":
            return self.model.compute_logits(hidden_states, spec_step_idx=spec_step_idx)
        return self.model.compute_logits(hidden_states)

    def _greedy_sample(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor:
        """Greedy-sample draft tokens from hidden states."""
        if self.use_local_argmax_reduction:
            return self.model.get_top_tokens(hidden_states)
        return self._compute_logits(hidden_states, spec_step_idx).argmax(dim=-1)

    def _sample_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self._enable_probabilistic_draft_probs or sampling_metadata.all_greedy:
            return self._greedy_sample(hidden_states, spec_step_idx), None
        logits = self._compute_logits(hidden_states, spec_step_idx)
        return self._sample_from_logits(logits, sampling_metadata)

    def propose(
        self,
        num_speculative_tokens,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens] or [3, num_tokens] when M-RoPE is enabled
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> torch.Tensor:
        if self.use_multi_mtp_heads and num_speculative_tokens > 1:
            return self.propose_multi_head_mtp(
                num_speculative_tokens=num_speculative_tokens,
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                token_indices_to_sample=token_indices_to_sample,
                common_attn_metadata=common_attn_metadata,
                sampling_metadata=sampling_metadata,
                mm_embed_inputs=mm_embed_inputs,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )
        return super().propose(
            num_speculative_tokens=num_speculative_tokens,
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            next_token_ids=next_token_ids,
            token_indices_to_sample=token_indices_to_sample,
            common_attn_metadata=common_attn_metadata,
            sampling_metadata=sampling_metadata,
            mm_embed_inputs=mm_embed_inputs,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            slot_mappings=slot_mappings,
        )

    def propose_multi_head_mtp(
        self,
        num_speculative_tokens: int,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.uses_mrope:
            raise NotImplementedError("Multi-head MTP expects 1D RoPE.")
        if num_speculative_tokens > self.n_predict:
            raise RuntimeError(
                "Multi-head MTP currently requires num_speculative_tokens <= n_predict."
            )

        self.num_speculative_tokens = num_speculative_tokens
        self._last_draft_probs = None

        num_tokens, token_indices_to_sample, common_attn_metadata = (
            self.set_inputs_first_pass(
                target_token_ids=target_token_ids,
                next_token_ids=next_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                token_indices_to_sample=token_indices_to_sample,
                cad=common_attn_metadata,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )
        )
        assert token_indices_to_sample is not None

        _, per_layer_attn_metadata = self.build_per_group_and_layer_attn_metadata(
            common_attn_metadata
        )
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens)
        )
        slot_mapping = self._get_slot_mapping(num_input_tokens)
        last_positions = self.positions[token_indices_to_sample]
        last_input_ids = self.input_ids[:num_tokens].clone()
        last_hidden_states: torch.Tensor | None = None
        last_sample_token_ids: torch.Tensor | None = None
        draft_token_ids_list: list[torch.Tensor] = []
        draft_probs_list: list[torch.Tensor] | None = None

        for spec_step_idx in range(num_speculative_tokens):
            if spec_step_idx > 0:
                assert last_hidden_states is not None
                assert last_sample_token_ids is not None
                tail_token_ids = self._get_multi_mtp_tail_token_ids(
                    last_positions, spec_step_idx, last_sample_token_ids
                )
                self.input_ids[: num_tokens - 1] = last_input_ids[1:]
                self.input_ids[token_indices_to_sample] = tail_token_ids
                last_input_ids = self.input_ids[:num_tokens].clone()
                self.hidden_states[:num_tokens] = last_hidden_states[:num_tokens]

            model_kwargs, _ = self.build_model_inputs_first_pass(
                num_tokens, num_input_tokens, mm_embed_inputs
            )
            if self.method == "mtp":
                model_kwargs["spec_step_idx"] = spec_step_idx

            if (
                spec_step_idx == 0
                and self._share_mtp_indices
                and hasattr(self.model.model, "set_skip_topk")
            ):
                self.model.model.set_skip_topk(False)

            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=slot_mapping,
            ):
                ret_hidden_states = self.model(**model_kwargs)

            if (
                spec_step_idx == 0
                and self._share_mtp_indices
                and hasattr(self.model.model, "set_skip_topk")
            ):
                self.model.model.set_skip_topk(True)

            if not self.model_returns_tuple():
                sample_hidden_states = ret_hidden_states
                last_hidden_states = ret_hidden_states
            elif spec_step_idx == 0:
                sample_hidden_states, last_hidden_states = ret_hidden_states
            else:
                _, last_hidden_states = ret_hidden_states
                sample_hidden_states = last_hidden_states

            draft_token_ids, draft_probs = self._sample_draft_tokens(
                sample_hidden_states[token_indices_to_sample],
                sampling_metadata,
                spec_step_idx=spec_step_idx,
            )
            if draft_probs is not None:
                if draft_probs_list is None:
                    draft_probs_list = []
                draft_probs_list.append(draft_probs)
            draft_token_ids_list.append(draft_token_ids)
            last_sample_token_ids = draft_token_ids.int()

        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        if draft_probs_list is not None:
            self._last_draft_probs = torch.stack(draft_probs_list, dim=1).contiguous()
        return draft_token_ids

    def _get_multi_mtp_tail_token_ids(
        self,
        last_positions: torch.Tensor,
        lookahead_steps: int,
        fallback_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return the per-request token used after shifting this MTP head input."""
        if self.runner is None:
            return fallback_token_ids

        input_batch = self.runner.input_batch
        result: torch.Tensor | None = None
        for batch_idx, last_pos in enumerate(last_positions.detach().cpu().tolist()):
            req_index = input_batch.req_id_to_index[input_batch.req_ids[batch_idx]]
            # first_input_ids is already shifted by one; each extra MTP depth
            # shifts once more, so depth 1 needs last_pos + 2.
            token_pos = int(last_pos) + lookahead_steps + 1
            if (
                token_pos < input_batch.num_tokens_no_spec[req_index]
                and input_batch.is_token_ids[req_index, token_pos]
            ):
                if result is None:
                    result = fallback_token_ids.clone()
                result[batch_idx] = int(input_batch.token_ids_cpu[req_index, token_pos])
            elif bool(self.runner.discard_request_mask.np[req_index]):
                raise RuntimeError(
                    "Multi-head MTP chunked prefill needs lookahead "
                    f"token at position {token_pos}, but it is beyond the "
                    "known prompt tokens. Keep the final prefill chunk at "
                    "least num_speculative_tokens tokens."
                )
        return fallback_token_ids if result is None else result

    def _update_positions_dependent_metadata(
        self,
        positions: torch.Tensor,
        common_attn_metadata,
        batch_size: int,
        input_batch_size: int,
        block_size: int | None = None,
    ) -> torch.Tensor:
        """Update positions, slot mappings, and sequence metadata for the
        next draft step. Returns the updated positions tensor."""
        old_positions_1d = positions[0] if self.uses_mrope else positions
        primary_gid = self.kv_cache_gid
        primary_block_size = self._block_size_by_gid.get(
            primary_gid, block_size or self.block_size
        )
        positions = super()._update_positions_dependent_metadata(
            positions,
            common_attn_metadata,
            batch_size,
            input_batch_size,
            primary_block_size,
        )

        self._per_group_slot_mappings[primary_gid] = common_attn_metadata.slot_mapping

        new_positions_1d = positions[0] if self.uses_mrope else positions
        exceeds = old_positions_1d + 1 >= self.max_model_len
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            if gid == primary_gid:
                continue
            block_table = self._per_group_block_tables.get(gid)
            if block_table is None:
                continue
            group_block_size = self._block_size_by_gid.get(gid, self.block_size)
            n_blocks = block_table.shape[1]
            bn = (
                (new_positions_1d // group_block_size)
                .clamp(max=n_blocks - 1)
                .to(torch.long)
            )
            block_ids = block_table[:batch_size].gather(1, bn.unsqueeze(1)).squeeze(1)
            sm = block_ids * group_block_size + (new_positions_1d % group_block_size)
            sm.masked_fill_(exceeds, PADDING_SLOT_ID)
            buf = self._slot_mapping_buffer_for(gid)
            buf[:batch_size].copy_(sm)
            if input_batch_size > batch_size:
                buf[batch_size:input_batch_size].fill_(PADDING_SLOT_ID)
            self._per_group_slot_mappings[gid] = buf[:batch_size]
        return positions

    def build_per_group_and_layer_attn_metadata(
        self, common_attn_metadata: CommonAttentionMetadata, draft_index: int = 0
    ) -> tuple[list[object], dict[str, object]]:
        per_group_attn_metadata: list[object] = []
        per_layer_attn_metadata: dict[str, object] = {}
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            if gid in self._per_group_block_tables:
                cm = copy(common_attn_metadata)
                # Target-model metadata may be unpadded for drafting while
                # per-group block tables are still padded for FULL CUDAGraph.
                cm.block_table_tensor = self._per_group_block_tables[gid][:num_reqs]
                slot_mapping = self._per_group_slot_mappings.get(gid)
                if slot_mapping is not None:
                    cm.slot_mapping = slot_mapping[:num_actual_tokens]
            else:
                cm = common_attn_metadata
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=cm, draft_index=draft_index
            )
            per_group_attn_metadata.append(attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        return per_group_attn_metadata, per_layer_attn_metadata

    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        """
        Initialize AttentionGroups for draft layers using kv_cache_config.
        Called from the model runner's initialize_metadata_builders.
        """
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )

        layer_to_gid: dict[str, int] = {}
        layer_to_spec: dict[str, KVCacheSpec] = {}
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            group_spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                layer_to_gid[layer_name] = gid
                if isinstance(group_spec, UniformTypeKVCacheSpecs):
                    if layer_name in group_spec.kv_cache_specs:
                        layer_to_spec[layer_name] = group_spec.kv_cache_specs[
                            layer_name
                        ]
                    else:
                        target_layer = getattr(
                            all_attn_layers.get(layer_name),
                            "kv_sharing_target_layer_name",
                            None,
                        )
                        if (
                            target_layer is not None
                            and target_layer in group_spec.kv_cache_specs
                        ):
                            layer_to_spec[layer_name] = group_spec.kv_cache_specs[
                                target_layer
                            ]
                        else:
                            layer_to_spec[layer_name] = group_spec
                else:
                    layer_to_spec[layer_name] = group_spec

        attention_groups: dict[
            tuple[int, tuple[str, str], KVCacheSpec], AttentionGroup
        ] = {}
        for layer_name in sorted(self._draft_attn_layer_names):
            if layer_name not in layer_to_spec:
                continue
            attn_layer = all_attn_layers[layer_name]
            attn_backend = attn_layer.get_attn_backend()
            spec = layer_to_spec[layer_name]
            gid = layer_to_gid[layer_name]
            # Layers with the same backend/spec may still live in different
            # KV cache groups and therefore need different block tables.
            group_key = (gid, attn_backend.full_cls_name(), spec)

            if group_key not in attention_groups:
                kernel_block_size = (
                    kernel_block_sizes[gid]
                    if kernel_block_sizes is not None and gid < len(kernel_block_sizes)
                    else None
                )
                attn_group = AttentionGroup(
                    backend=attn_backend,
                    layer_names=[layer_name],
                    kv_cache_spec=spec,
                    kv_cache_group_id=gid,
                )
                attn_group.create_metadata_builders(
                    self.vllm_config,
                    self.device,
                    kernel_block_size=kernel_block_size,
                )
                attention_groups[group_key] = attn_group
            else:
                attention_groups[group_key].layer_names.append(layer_name)

        self.draft_attn_groups = list(attention_groups.values())
        self._block_size_by_gid = {}
        if self.draft_attn_groups:
            for attn_group in self.draft_attn_groups:
                gid = attn_group.kv_cache_group_id
                self._block_size_by_gid[gid] = (
                    attn_group.get_metadata_builder().kv_cache_spec.block_size
                )
            self.kv_cache_gid = min(self._block_size_by_gid)
            self.block_size = self._block_size_by_gid[self.kv_cache_gid]
        elif kv_cache_config.kv_cache_groups:
            self.kv_cache_gid = 0
            self.block_size = kv_cache_config.kv_cache_groups[
                0
            ].kv_cache_spec.block_size
        logger.debug(
            "Initialized %d draft attention groups across KV cache gids %s",
            len(self.draft_attn_groups),
            sorted(self._block_size_by_gid),
        )
