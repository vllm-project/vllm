# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from copy import copy

import torch

from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
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
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
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
        self._multi_mtp_next_token_ids = torch.full(
            (self.num_mtp_prefill_heads, self.max_batch_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.fix_multi_mtp_kvcache = self.use_multi_mtp_heads

    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        if self.use_multi_mtp_heads:
            self.cudagraph_dispatcher.initialize_cudagraph_keys(CUDAGraphMode.NONE)
            return
        super().initialize_cudagraph_keys(cudagraph_mode)

    def prepare_next_token_ids_padded(
        self,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_mask: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_token_ids, valid_sampled_tokens_count = (
            super().prepare_next_token_ids_padded(
                sampled_token_ids,
                requests,
                gpu_input_batch,
                discard_request_mask,
                common_attn_metadata,
            )
        )
        if not self.use_multi_mtp_heads or common_attn_metadata is None:
            return next_token_ids, valid_sampled_tokens_count

        num_reqs = gpu_input_batch.num_reqs
        prepared_token_ids = self._multi_mtp_next_token_ids[
            : self.num_mtp_prefill_heads, :num_reqs
        ]
        prepared_token_ids.fill_(-1)
        prepared_token_ids[0].copy_(next_token_ids[:num_reqs])

        seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        for head_idx in range(1, self.num_mtp_prefill_heads):
            for req_idx in range(num_reqs):
                token_pos = min(
                    int(seq_lens_cpu[req_idx]) + head_idx,
                    self.max_model_len - 1,
                )
                self.backup_next_token_ids.np[req_idx] = requests[
                    gpu_input_batch.req_ids[req_idx]
                ].get_token_id(token_pos)
            self.backup_next_token_ids.copy_to_gpu(num_reqs)
            prepared_token_ids[head_idx].copy_(
                torch.where(
                    discard_request_mask[:num_reqs],
                    self.backup_next_token_ids.gpu[:num_reqs],
                    -1,
                )
            )

        return next_token_ids, valid_sampled_tokens_count

    def set_draft_attention_metadata(
        self,
        num_accepted_tokens: torch.Tensor | None,
    ) -> None:
        for attn_group in self.draft_attn_groups:
            builder = attn_group.get_metadata_builder()
            if hasattr(builder, "set_draft_attention_metadata"):
                builder.set_draft_attention_metadata(
                    num_accepted_tokens,
                )

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

    def _update_group_slot_mapping(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        final_n_token_indices: torch.Tensor,
        updated_positions: torch.Tensor,
        batch_size: int,
    ) -> None:
        flat_indices = final_n_token_indices.view(-1)

        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            block_size = self._block_size_by_gid.get(gid, self.block_size)
            block_table = self._per_group_block_tables.get(
                gid, common_attn_metadata.block_table_tensor
            )[:batch_size]

            block_nums = updated_positions // block_size
            block_offsets = updated_positions % block_size
            block_ids = block_table.gather(1, block_nums.to(torch.long))
            new_slot_mapping = block_ids * block_size + block_offsets

            if gid == self.kv_cache_gid:
                common_attn_metadata.slot_mapping[flat_indices] = new_slot_mapping.view(
                    -1
                )
                self._per_group_slot_mappings[gid] = common_attn_metadata.slot_mapping
                continue

            slot_mapping = self._per_group_slot_mappings.get(gid)
            if slot_mapping is None:
                slot_mapping = common_attn_metadata.slot_mapping.clone()
            slot_mapping[flat_indices] = new_slot_mapping.view(-1)
            self._per_group_slot_mappings[gid] = slot_mapping

    def _update_common_seq_lens(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        has_draft_tokens: torch.Tensor,
        num_rejected_tokens_gpu: torch.Tensor,
        batch_size: int,
    ) -> None:
        seq_lens = common_attn_metadata.seq_lens[:batch_size]
        num_rejected_tokens = num_rejected_tokens_gpu[:batch_size].to(seq_lens.dtype)
        seq_lens.copy_(
            torch.where(
                has_draft_tokens,
                seq_lens - num_rejected_tokens,
                seq_lens,
            )
        )
        common_attn_metadata._seq_lens_cpu = None
        common_attn_metadata._num_computed_tokens_cpu = None
        common_attn_metadata._num_computed_tokens_cache = None

    def _save_and_change_target_input(
        self,
        num_tokens,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        last_token_indices,
        common_attn_metadata: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> None:
        input_batch = getattr(self, "input_batch", None)
        if input_batch is None or getattr(input_batch, "disable_multi_mtp_cache", True):
            return
        batch_size = next_token_ids.numel()

        device = target_token_ids.device
        has_draft_tokens = self.runner.num_decode_draft_tokens.gpu[:batch_size] > 0
        num_rejected_tokens: torch.Tensor | None = None
        if num_rejected_tokens_gpu is not None:
            num_rejected_tokens = num_rejected_tokens_gpu[:batch_size]
        basic_range = torch.arange(
            1 + self.num_speculative_tokens,
            device=device,
            dtype=common_attn_metadata.query_start_loc.dtype,
        )
        final_n_token_indices = (
            common_attn_metadata.query_start_loc[1 : batch_size + 1, None]
            - self.num_speculative_tokens
            - 1
            + basic_range
        )
        final_token_ids = target_token_ids[final_n_token_indices]
        final_hidden_states = target_hidden_states[final_n_token_indices]

        previous_token_ids = input_batch.multi_mtp_target_token_ids_cache[:batch_size]
        previous_hidden_states = input_batch.multi_mtp_target_hidden_states_cache[
            :batch_size
        ]

        token_ids = torch.cat([previous_token_ids, final_token_ids], dim=1)
        hidden_states = torch.cat([previous_hidden_states, final_hidden_states], dim=1)

        selected_indices = (
            basic_range[None, :]
            + 1
            + self.num_speculative_tokens
            + torch.arange(
                batch_size,
                dtype=basic_range.dtype,
                device=device,
            )[:, None]
            * (1 + self.num_speculative_tokens)
            * 2
        )
        if num_rejected_tokens_gpu is not None:
            assert num_rejected_tokens is not None
            selected_indices = torch.where(
                has_draft_tokens[:, None],
                selected_indices - num_rejected_tokens[:, None],
                selected_indices,
            )

        selected_indices = selected_indices.view(-1)
        selected_token_ids = token_ids.view(-1)[selected_indices]
        selected_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])[
            selected_indices
        ]
        target_token_ids[final_n_token_indices.view(-1)] = selected_token_ids
        target_hidden_states[final_n_token_indices.view(-1)] = selected_hidden_states
        input_batch.multi_mtp_target_token_ids_cache[:batch_size] = (
            selected_token_ids.view(batch_size, -1)
        )
        input_batch.multi_mtp_target_hidden_states_cache[:batch_size] = (
            selected_hidden_states.view(
                batch_size, -1, selected_hidden_states.shape[-1]
            )
        )

        last_token_indices[:] = (
            common_attn_metadata.query_start_loc[1 : batch_size + 1] - 1
        )
        if num_rejected_tokens_gpu is not None:
            assert num_rejected_tokens is not None
            updated_positions = torch.where(
                has_draft_tokens[:, None],
                target_positions[final_n_token_indices] - num_rejected_tokens[:, None],
                target_positions[final_n_token_indices],
            )
            target_positions[final_n_token_indices.view(-1)] = updated_positions.view(
                -1
            )
            self._update_group_slot_mapping(
                common_attn_metadata,
                final_n_token_indices,
                updated_positions,
                batch_size,
            )
            self._update_common_seq_lens(
                common_attn_metadata,
                has_draft_tokens,
                num_rejected_tokens,
                batch_size,
            )

        num_accepted_tokens = self.runner.num_accepted_tokens.gpu[:batch_size].clone()
        window_size = 1 + self.num_speculative_tokens
        if num_rejected_tokens_gpu is not None:
            assert num_rejected_tokens is not None
            num_rejected_tokens = num_rejected_tokens.to(num_accepted_tokens.dtype)
            num_accepted_tokens.copy_(
                torch.where(
                    has_draft_tokens,
                    window_size - num_rejected_tokens,
                    window_size,
                )
            )
        self.set_draft_attention_metadata(num_accepted_tokens)

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

        self._save_and_change_target_input(
            num_tokens,
            self.input_ids,
            self.positions,
            self.hidden_states,
            next_token_ids,
            token_indices_to_sample,
            common_attn_metadata,
            num_rejected_tokens_gpu,
        )

        _, per_layer_attn_metadata = self.build_per_group_and_layer_attn_metadata(
            common_attn_metadata
        )
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens)
        )
        slot_mapping = self._get_slot_mapping(num_input_tokens)
        last_input_ids = self.input_ids[:num_tokens].clone()
        draft_token_ids_list: list[torch.Tensor] = []
        draft_probs_list: list[torch.Tensor] | None = None

        for spec_step_idx in range(num_speculative_tokens):
            current_step_idx = spec_step_idx % self.model.model.num_mtp_layers
            layer_key = str(self.model.model.mtp_start_layer_idx + current_step_idx)
            input_ids = self.input_ids[:num_input_tokens]
            self.inputs_embeds[:num_input_tokens].copy_(
                self.model.model.embed_tokens(input_ids)
            )
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            positions = self._get_positions(num_input_tokens)
            previous_hidden_states = self.hidden_states[:num_input_tokens]
            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=slot_mapping,
            ):
                sample_hidden_states = self.model.model.layers[layer_key](
                    None,
                    positions,
                    previous_hidden_states,
                    inputs_embeds,
                    current_step_idx,
                )

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
            if spec_step_idx < num_speculative_tokens - 1:
                self.input_ids[: num_tokens - 1] = last_input_ids[1:]
                self.input_ids[token_indices_to_sample] = last_sample_token_ids
                last_input_ids = self.input_ids[:num_tokens].clone()
                self.hidden_states[:num_tokens] = sample_hidden_states[:num_tokens]

        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        if draft_probs_list is not None:
            self._last_draft_probs = torch.stack(draft_probs_list, dim=1).contiguous()
        return draft_token_ids

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        if not self.use_multi_mtp_heads:
            return super().dummy_run(
                num_tokens=num_tokens,
                use_cudagraphs=use_cudagraphs,
                is_graph_capturing=is_graph_capturing,
                slot_mappings=slot_mappings,
            )

        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(
                num_tokens, use_cudagraphs=use_cudagraphs
            )
        )
        if (
            self._draft_attn_layer_names
            and slot_mappings is not None
            and next(iter(self._draft_attn_layer_names)) in slot_mappings
        ):
            slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
        else:
            slot_mapping_dict = slot_mappings or {}

        for spec_step_idx in range(self.num_mtp_prefill_heads):
            current_step_idx = spec_step_idx % self.model.model.num_mtp_layers
            layer_key = str(self.model.model.mtp_start_layer_idx + current_step_idx)
            input_ids = self.input_ids[:num_input_tokens]
            self.inputs_embeds[:num_input_tokens].copy_(
                self.model.model.embed_tokens(input_ids)
            )
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            positions = self._get_positions(num_input_tokens)
            previous_hidden_states = self.hidden_states[:num_input_tokens]
            with set_forward_context(
                None,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=slot_mapping_dict,
            ):
                self.model.model.layers[layer_key](
                    None,
                    positions,
                    previous_hidden_states,
                    inputs_embeds,
                    current_step_idx,
                )

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
