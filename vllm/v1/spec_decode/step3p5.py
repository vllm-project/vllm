# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import copy

import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config, replace
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.utils import AttentionGroup


class Step3p5MTPProposer(EagleProposer):
    """Step3.5 MTP proposer with per-layer draft-step selection."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config, device, runner)
        self._per_group_block_tables: dict[int, torch.Tensor] = {}

    def set_per_group_block_table(self, gid: int, block_table: torch.Tensor) -> None:
        self._per_group_block_tables[gid] = block_table

    def build_per_group_and_layer_attn_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int = 0,
    ) -> tuple[list[object], dict[str, object]]:
        per_group_attn_metadata: list[object] = []
        per_layer_attn_metadata: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            if gid in self._per_group_block_tables:
                cm = copy(common_attn_metadata)
                cm.block_table_tensor = self._per_group_block_tables[gid]
            else:
                cm = common_attn_metadata
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=cm,
                draft_index=draft_index,
            )
            per_group_attn_metadata.append(attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        return per_group_attn_metadata, per_layer_attn_metadata

    def _maybe_share_lm_head(self, target_language_model: torch.nn.Module) -> None:
        """Step3.5 MTP uses the lm_head stored in each MTP layer."""

        # The base MTP path shares target lm_head into shared_head.head.
        # Step3.5 checkpoints carry per-MTP-layer shared_head weights.
        return

    def _create_draft_vllm_config(self) -> VllmConfig:
        base = super()._create_draft_vllm_config()
        return replace(base, model_config=self.draft_model_config, quant_config=None)

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        """Step3.5 MTP draft layers may span multiple KV cache groups."""

    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
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
                        target_layer_name = getattr(
                            all_attn_layers.get(layer_name),
                            "kv_sharing_target_layer_name",
                            None,
                        )
                        if (
                            target_layer_name
                            and target_layer_name in group_spec.kv_cache_specs
                        ):
                            layer_to_spec[layer_name] = group_spec.kv_cache_specs[
                                target_layer_name
                            ]
                        else:
                            layer_to_spec[layer_name] = group_spec
                else:
                    layer_to_spec[layer_name] = group_spec

        attention_groups: dict[
            tuple[tuple[str, str], KVCacheSpec], AttentionGroup
        ] = {}
        for layer_name in self._draft_attn_layer_names:
            if layer_name not in layer_to_spec:
                continue
            attn_layer = all_attn_layers[layer_name]
            attn_backend = attn_layer.get_attn_backend()
            spec = layer_to_spec[layer_name]
            gid = layer_to_gid[layer_name]
            group_key = (attn_backend.full_cls_name(), spec)

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
        if self.draft_attn_groups:
            self.kv_cache_gid = self.draft_attn_groups[0].kv_cache_group_id
            self.block_size = (
                self.draft_attn_groups[0]
                .get_metadata_builder()
                .kv_cache_spec.block_size
            )
        else:
            self.kv_cache_gid = 0
            self.block_size = kv_cache_config.kv_cache_groups[
                0
            ].kv_cache_spec.block_size

    def prepare_next_token_ids_padded(
        self,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_reqs = gpu_input_batch.num_reqs
        for i in range(num_reqs):
            self.backup_next_token_ids.np[i] = requests[
                gpu_input_batch.req_ids[i]
            ].get_token_id(gpu_input_batch.num_tokens_no_spec[i] - 1)
        self.backup_next_token_ids.copy_to_gpu(num_reqs)

        valid_sampled_mask = (sampled_token_ids[:num_reqs] >= 0) & (
            sampled_token_ids[:num_reqs] < gpu_input_batch.vocab_size
        )
        valid_sampled_tokens_count = valid_sampled_mask.sum(dim=1).to(torch.int32)
        max_valid_count = valid_sampled_tokens_count.new_full(
            valid_sampled_tokens_count.shape,
            sampled_token_ids.shape[1],
        )
        valid_sampled_tokens_count = torch.minimum(
            valid_sampled_tokens_count,
            max_valid_count,
        )
        valid_sampled_tokens_count = torch.where(
            discard_request_mask[:num_reqs],
            torch.zeros_like(valid_sampled_tokens_count),
            valid_sampled_tokens_count,
        )

        gather_indices = (
            torch.clamp(valid_sampled_tokens_count, min=1).to(torch.int64) - 1
        )
        last_sampled_token_ids = sampled_token_ids[:num_reqs].gather(
            1,
            gather_indices.unsqueeze(1),
        )
        last_sampled_token_ids = last_sampled_token_ids.squeeze(1).to(torch.int32)

        use_backup = discard_request_mask[:num_reqs] | (
            valid_sampled_tokens_count == 0
        )
        next_token_ids = torch.where(
            use_backup,
            self.backup_next_token_ids.gpu[:num_reqs],
            last_sampled_token_ids,
        )
        return next_token_ids, valid_sampled_tokens_count

    def _sample_draft_tokens_for_step(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        spec_step_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self._enable_probabilistic_draft_probs or sampling_metadata.all_greedy:
            if self.use_local_argmax_reduction:
                return self.model.get_top_tokens(hidden_states), None
            logits = self.model.compute_logits(
                hidden_states, spec_step_idx=spec_step_idx
            )
            return logits.argmax(dim=-1), None

        logits = self.model.compute_logits(hidden_states, spec_step_idx=spec_step_idx)
        return self._sample_from_logits(logits, sampling_metadata)

    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
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
        self._last_draft_probs = None
        batch_size = common_attn_metadata.batch_size()

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

        per_group_attn_metadata, per_layer_attn_metadata = (
            self.build_per_group_and_layer_attn_metadata(common_attn_metadata)
        )

        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens)
        )

        model_kwargs, slot_mapping_size = self.build_model_inputs_first_pass(
            num_tokens, num_input_tokens, mm_embed_inputs
        )
        model_kwargs["spec_step_idx"] = 0

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=self._get_slot_mapping(
                slot_mapping_size, common_attn_metadata.slot_mapping
            ),
        ):
            ret_hidden_states = self.model(**model_kwargs)
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states = last_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states

        sample_hidden_states = last_hidden_states[token_indices_to_sample]

        if self.num_speculative_tokens == 1 or self.parallel_drafting:
            draft_token_ids, draft_probs = self._sample_draft_tokens_for_step(
                sample_hidden_states, sampling_metadata, spec_step_idx=0
            )
            if draft_probs is not None:
                self._last_draft_probs = draft_probs.view(
                    -1, self.num_speculative_tokens, draft_probs.shape[-1]
                ).contiguous()
            return draft_token_ids.view(-1, self.num_speculative_tokens)

        if self.uses_mrope:
            positions = self.mrope_positions[:, token_indices_to_sample]
        else:
            positions = self.positions[token_indices_to_sample]
        hidden_states = hidden_states[token_indices_to_sample]

        if self.constant_draft_positions:
            self.positions[:batch_size] = positions

        draft_token_ids, draft_probs = self._sample_draft_tokens_for_step(
            sample_hidden_states, sampling_metadata, spec_step_idx=0
        )
        draft_probs_list = None if draft_probs is None else [draft_probs]

        if self.allowed_attn_types is not None:
            for group_md in per_group_attn_metadata:
                if not isinstance(group_md, self.allowed_attn_types):
                    raise ValueError(
                        f"Unsupported attention metadata type for speculative "
                        "decoding with num_speculative_tokens > 1: "
                        f"{type(group_md)}. Supported types are: "
                        f"{self.allowed_attn_types}"
                    )

        draft_token_ids_list = [draft_token_ids]

        cudagraph_runtime_mode, input_batch_size, batch_size_across_dp = (
            self._determine_batch_execution_and_padding(batch_size)
        )

        common_attn_metadata.num_actual_tokens = batch_size
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = self.arange[: batch_size + 1]
        common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
            self.token_arange_np[: batch_size + 1]
        ).clone()

        if self.num_speculative_tokens > 1 and num_rejected_tokens_gpu is not None:
            common_attn_metadata.seq_lens -= num_rejected_tokens_gpu
            common_attn_metadata._seq_lens_cpu = None
            common_attn_metadata._num_computed_tokens_cpu = None

        block_size = self.block_size
        assert block_size > 0, "block_size has not been initialized."
        for token_index in range(self.num_speculative_tokens - 1):
            spec_step_idx = token_index + 1
            input_ids = draft_token_ids_list[-1].int()

            if not self.constant_draft_positions:
                positions = self._update_positions_dependent_metadata(
                    positions,
                    common_attn_metadata,
                    batch_size,
                    input_batch_size,
                    block_size,
                )

            if not self.constant_draft_positions or token_index == 0:
                _, per_layer_attn_metadata = (
                    self.build_per_group_and_layer_attn_metadata(
                        common_attn_metadata, draft_index=spec_step_idx
                    )
                )

            self.input_ids[:batch_size] = input_ids
            self.hidden_states[:batch_size] = hidden_states
            if self.supports_mm_inputs:
                self.inputs_embeds[:batch_size] = self.model.embed_input_ids(input_ids)

                input_ids = None
                inputs_embeds = self.inputs_embeds[:input_batch_size]
            else:
                input_ids = self.input_ids[:input_batch_size]
                inputs_embeds = None

            model_kwargs = {
                "input_ids": input_ids,
                "positions": self._get_positions(input_batch_size),
                "inputs_embeds": inputs_embeds,
                "spec_step_idx": spec_step_idx,
            }
            if self.pass_hidden_states_to_model:
                model_kwargs["hidden_states"] = self.hidden_states[:input_batch_size]

            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=input_batch_size,
                num_tokens_across_dp=batch_size_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=self._get_slot_mapping(input_batch_size),
            ):
                ret_hidden_states = self.model(**model_kwargs)
                if not self.model_returns_tuple():
                    last_hidden_states = ret_hidden_states
                    hidden_states = ret_hidden_states
                else:
                    last_hidden_states, hidden_states = ret_hidden_states

            hidden_states = hidden_states[:batch_size]
            draft_token_ids, draft_probs = self._sample_draft_tokens_for_step(
                last_hidden_states[:batch_size],
                sampling_metadata,
                spec_step_idx=spec_step_idx,
            )
            if draft_probs is not None:
                assert draft_probs_list is not None
                draft_probs_list.append(draft_probs)
            draft_token_ids_list.append(draft_token_ids)

        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        if draft_probs_list is not None:
            self._last_draft_probs = torch.stack(draft_probs_list, dim=1).contiguous()
        return draft_token_ids
