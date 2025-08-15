# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
from dataclasses import replace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from vllm.attention.layer import Attention
from vllm.config import (CompilationLevel, VllmConfig,
                         get_layers_from_vllm_config)
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models import supports_multimodal
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.platforms import current_platform
from vllm.utils import is_pin_memory_available
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.attention.backends.rocm_aiter_fa import (
    AiterFlashAttentionMetadata)
from vllm.v1.attention.backends.tree_attn import (TreeAttentionMetadata,
                                                  TreeAttentionMetadataBuilder)
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.sample.metadata import SamplingMetadata

logger = init_logger(__name__)

PADDING_SLOT_ID = -1


class EagleProposer:

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        self.draft_model_config = self.speculative_config.draft_model_config
        self.method = self.speculative_config.method

        self.runner = runner
        self.dtype = vllm_config.model_config.dtype
        self.max_model_len = vllm_config.model_config.max_model_len
        self.block_size = vllm_config.cache_config.block_size
        self.num_speculative_tokens = (
            self.speculative_config.num_speculative_tokens)
        self.max_num_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens)
        self.token_arange_np = np.arange(self.max_num_tokens)
        # We need to get the hidden size from the draft model config because
        # the draft model's hidden size can be different from the target model's
        # hidden size (e.g., Llama 3.3 70B).
        self.hidden_size = self.draft_model_config.get_hidden_size()

        self.is_multimodal_model = vllm_config.model_config \
            .is_multimodal_model

        self.use_cuda_graph = (self.vllm_config.compilation_config.level
                               == CompilationLevel.PIECEWISE and
                               not self.vllm_config.model_config.enforce_eager)
        self.cudagraph_batch_sizes = list(
            reversed(
                self.vllm_config.compilation_config.cudagraph_capture_sizes))

        # persistent buffers for cuda graph
        self.input_ids = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int32,
                                     device=device)
        self.positions = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int64,
                                     device=device)
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=device)

        max_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.arange = torch.arange(
            # We need +1 here because the arange is used to set query_start_loc,
            # which has one more element than batch_size.
            max_batch_size + 1,
            device=device,
            dtype=torch.int32,
        )

        self.inputs_embeds = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=device)

        # Parse the speculative token tree.
        spec_token_tree = self.speculative_config.speculative_token_tree
        self.tree_choices: list[tuple[int,
                                      ...]] = ast.literal_eval(spec_token_tree)
        tree_depth = len(self.tree_choices[-1])
        # Precompute per-level properties of the tree.
        num_drafts_per_level = [0] * tree_depth
        for node in self.tree_choices:
            num_drafts_per_level[len(node) - 1] += 1
        self.cu_drafts_per_level = [num_drafts_per_level[0]]
        self.child_drafts_per_level = [num_drafts_per_level[0]]
        for level in range(1, tree_depth):
            self.cu_drafts_per_level.append(self.cu_drafts_per_level[-1] +
                                            num_drafts_per_level[level])
            self.child_drafts_per_level.append(num_drafts_per_level[level] //
                                               num_drafts_per_level[level - 1])
        # Precompute draft position offsets in flattened tree.
        self.tree_draft_pos_offsets = torch.arange(
            1,
            len(self.tree_choices) + 1,
            device=device,
            dtype=torch.int32,
        ).repeat(max_batch_size, 1)

    def propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens]
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embeds: Optional[list[torch.Tensor]] = None,
    ) -> torch.Tensor:
        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]
        last_token_indices = common_attn_metadata.query_start_loc[1:] - 1

        if self.method == "eagle3":
            assert isinstance(self.model, Eagle3LlamaForCausalLM)
            target_hidden_states = self.model.combine_hidden_states(
                target_hidden_states)
            assert target_hidden_states.shape[-1] == self.hidden_size

        # Shift the input ids by one token.
        # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        self.input_ids[:num_tokens - 1] = target_token_ids[1:]
        # Replace the last token with the next token.
        # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
        self.input_ids[last_token_indices] = next_token_ids

        assert self.runner is not None

        # FIXME: need to consider multiple kv_cache_groups
        attn_metadata = self.runner.attn_groups[0][0].metadata_builder\
            .build_for_drafting(common_attn_metadata=common_attn_metadata,
                                draft_index=0)

        # At this moment, we assume all eagle layers belong to the same KV
        # cache group, thus using the same attention metadata.
        per_layer_attn_metadata = {}
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata
        if self.use_cuda_graph and \
            num_tokens <= self.cudagraph_batch_sizes[-1]:
            num_input_tokens = self.vllm_config.pad_for_cudagraph(num_tokens)
        else:
            num_input_tokens = num_tokens
        # copy inputs to buffer for cudagraph
        self.positions[:num_tokens] = target_positions
        self.hidden_states[:num_tokens] = target_hidden_states
        if self.is_multimodal_model:
            input_ids = self.input_ids[:num_tokens]
            inputs_embeds = self.model.get_input_embeddings(
                input_ids,
                multimodal_embeddings=mm_embeds or None,
            )
            self.inputs_embeds[:num_tokens] = inputs_embeds
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            input_ids = None
        else:
            inputs_embeds = None
            input_ids = self.input_ids[:num_input_tokens]

        with set_forward_context(per_layer_attn_metadata,
                                 self.vllm_config,
                                 num_tokens=num_input_tokens):
            ret_hidden_states = self.model(
                input_ids=input_ids,
                positions=self.positions[:num_input_tokens],
                hidden_states=self.hidden_states[:num_input_tokens],
                inputs_embeds=inputs_embeds,
            )
            if self.method == "deepseek_mtp":
                last_hidden_states = ret_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states
        sample_hidden_states = last_hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states, None)
        positions = target_positions[last_token_indices]
        hidden_states = hidden_states[last_token_indices]

        if isinstance(attn_metadata, TreeAttentionMetadata):
            # Draft using tree attention.
            draft_token_ids_list = self.propose_tree(
                batch_size=batch_size,
                logits=logits,
                positions=positions,
                hidden_states=hidden_states,
                common_attn_metadata=common_attn_metadata,
            )
            # [batch_size, num_tree_tokens]
            return torch.cat(draft_token_ids_list, dim=1)

        draft_token_ids = logits.argmax(dim=-1)

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            # [batch_size, 1]
            return draft_token_ids.view(-1, 1)

        # TODO: Currently, MTP module released by deepseek only has
        # one layer. Adapt this code to support multiple layers once
        # there's a multi-layer MTP module.

        # On ROCm, both AiterFlashAttention and TritonAttention
        # support multi-token eagle spec decode.
        if current_platform.is_rocm():
            assert isinstance(
                attn_metadata,
                (TritonAttentionMetadata, AiterFlashAttentionMetadata,
                 FlashAttentionMetadata))
        else:
            # Currently, only FlashAttention supports multi-token eagle spec
            # decode. This is because the code below makes assumptions about
            # attn_metadata attributes available.
            assert isinstance(attn_metadata, FlashAttentionMetadata)

        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]

        if self.use_cuda_graph and \
            batch_size <= self.cudagraph_batch_sizes[-1]:
            input_batch_size = self.vllm_config.pad_for_cudagraph(batch_size)
        else:
            input_batch_size = batch_size
        attn_metadata.num_actual_tokens = batch_size
        attn_metadata.max_query_len = 1
        attn_metadata.query_start_loc = self.arange[:batch_size + 1]
        for _ in range(self.num_speculative_tokens - 1):
            # Update the inputs.
            # cast to int32 is crucial when eagle model is compiled.
            # tensor.argmax() returns int64 by default.
            input_ids = draft_token_ids_list[-1].int()
            positions += 1

            # NOTE(woosuk): We should handle the case where the draft model
            # generates tokens beyond the max model length. Since it is complex
            # to remove such requests from the batch, we keep them in the batch
            # but adjust the position ids and slot mappings to avoid the
            # out-of-range access during the model execution. The draft tokens
            # generated with this adjustment should be ignored.
            exceeds_max_model_len = positions >= self.max_model_len
            # Mask out the position ids that exceed the max model length.
            # Otherwise, we may get out-of-range error in RoPE.
            clamped_positions = torch.where(exceeds_max_model_len, 0,
                                            positions)

            # Increment the sequence lengths.
            attn_metadata.max_seq_len += 1
            attn_metadata.seq_lens += 1
            # Consider max model length.
            attn_metadata.max_seq_len = min(attn_metadata.max_seq_len,
                                            self.max_model_len)
            # For the requests that exceed the max model length, we set the
            # sequence length to 1 to minimize their overheads in attention.
            attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)

            # Compute the slot mapping.
            block_numbers = clamped_positions // self.block_size
            block_ids = attn_metadata.block_table.gather(
                dim=1, index=block_numbers.view(-1, 1))
            block_ids = block_ids.view(-1)
            attn_metadata.slot_mapping = (block_ids * self.block_size +
                                          clamped_positions % self.block_size)
            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            attn_metadata.slot_mapping.masked_fill_(exceeds_max_model_len,
                                                    PADDING_SLOT_ID)

            # copy inputs to buffer for cudagraph
            self.input_ids[:batch_size] = input_ids
            self.positions[:batch_size] = clamped_positions
            self.hidden_states[:batch_size] = hidden_states
            if self.is_multimodal_model:
                inputs_embeds = self.model.get_input_embeddings(input_ids)
                self.inputs_embeds[:batch_size] = inputs_embeds
                inputs_embeds = self.inputs_embeds[:input_batch_size]
                input_ids = None
            else:
                inputs_embeds = None
                input_ids = self.input_ids[:input_batch_size]

            # Run the model.
            with set_forward_context(per_layer_attn_metadata,
                                     self.vllm_config,
                                     num_tokens=input_batch_size):
                last_hidden_states, hidden_states = self.model(
                    input_ids=input_ids,
                    positions=self.positions[:input_batch_size],
                    hidden_states=self.hidden_states[:input_batch_size],
                    inputs_embeds=inputs_embeds,
                )
            hidden_states = hidden_states[:batch_size]
            logits = self.model.compute_logits(last_hidden_states[:batch_size],
                                               None)
            draft_token_ids = logits.argmax(dim=-1)
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids

    def propose_tree(
        self,
        batch_size: int,
        # [num_tokens, vocab_size]
        logits: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
        # [num_tokens, hidden_size]
        hidden_states: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> list[torch.Tensor]:
        tree_attn_metadata_builder = \
            self.runner.attn_groups[0][0].metadata_builder
        assert isinstance(tree_attn_metadata_builder,
                          TreeAttentionMetadataBuilder)

        total_num_drafts = self.cu_drafts_per_level[0]
        level_num_drafts = total_num_drafts
        # Sample a draft token for each child at the tree root level.
        num_children = self.child_drafts_per_level[0]
        if num_children == 1:
            draft_token_ids = logits.argmax(dim=-1).view(batch_size, -1)
        else:
            draft_token_ids = torch.topk(logits, num_children,
                                         dim=-1).indices.view(batch_size, -1)
        draft_token_ids_list = [draft_token_ids]
        draft_hidden_states = hidden_states.view(batch_size, 1, -1)

        # Initialize empty tensors for concatenation with the level outputs.
        tree_input_ids = torch.empty(0,
                                     device=self.input_ids.device,
                                     dtype=self.input_ids.dtype)
        tree_positions = torch.empty(0,
                                     device=self.positions.device,
                                     dtype=self.positions.dtype)
        tree_hidden_states = torch.empty(0,
                                         device=self.hidden_states.device,
                                         dtype=self.hidden_states.dtype)
        # Precompute the draft token positions.
        flattened_draft_positions = (
            positions.view(batch_size, -1) +
            self.tree_draft_pos_offsets[:batch_size, :])
        tree_depth = len(self.cu_drafts_per_level)
        for level in range(tree_depth - 1):
            # Get draft positions for RoPE.
            draft_positions = positions + (level + 1)
            exceeds_max_model_len = (positions +
                                     total_num_drafts) >= self.max_model_len
            # Mask out the position ids that exceed the max model length.
            # Otherwise, we may get out-of-range error in RoPE.
            draft_positions = torch.where(
                exceeds_max_model_len,
                0,
                draft_positions,
            ).view(batch_size, -1)

            if level_num_drafts > 1:
                # Repeat the positions for each draft at this level.
                draft_positions = draft_positions.repeat_interleave(
                    level_num_drafts, dim=1)

            if num_children > 1:
                # Repeat draft hidden states for each child.
                draft_hidden_states = draft_hidden_states.repeat_interleave(
                    num_children, dim=1)

            # Concatenate the draft tokens, positions, and hidden states.
            tree_input_ids = torch.cat([tree_input_ids, draft_token_ids],
                                       dim=1)
            tree_positions = torch.cat([tree_positions, draft_positions],
                                       dim=1)
            tree_hidden_states = torch.cat(
                [tree_hidden_states, draft_hidden_states], dim=1)

            # Build new attention metadata for the next level of drafts.
            # This is necessary to support tree attention.
            query_len = total_num_drafts
            common_attn_metadata = replace(
                common_attn_metadata,
                query_start_loc=query_len * self.arange[:batch_size + 1],
                seq_lens=common_attn_metadata.seq_lens + level_num_drafts,
                num_actual_tokens=batch_size * query_len,
                max_query_len=query_len,
            )
            attn_metadata = tree_attn_metadata_builder.build_for_drafting(
                common_attn_metadata=common_attn_metadata,
                draft_index=level + 1,
            )

            # Apply new attention metadata to all layers.
            per_layer_attn_metadata = {}
            for layer_name in self.attn_layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

            # Consider max model length.
            attn_metadata.max_seq_len = min(attn_metadata.max_seq_len,
                                            self.max_model_len)
            # For the requests that exceed the max model length, we set the
            # sequence length to 1 to minimize their overheads in attention.
            attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)

            # Compute the slot mapping.
            query_positions = flattened_draft_positions[:, level:level +
                                                        query_len]
            block_numbers = query_positions // self.block_size
            block_ids = attn_metadata.block_table.gather(dim=1,
                                                         index=block_numbers)
            slot_mapping = (block_ids * self.block_size +
                            query_positions % self.block_size)
            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            slot_mapping[exceeds_max_model_len] = PADDING_SLOT_ID
            attn_metadata.slot_mapping = slot_mapping.view(-1)

            # Copy inputs to buffer for cudagraph.
            num_tokens = attn_metadata.num_actual_tokens
            input_ids = tree_input_ids.view(-1)
            self.input_ids[:num_tokens] = input_ids
            self.positions[:num_tokens] = tree_positions.view(-1)
            self.hidden_states[:num_tokens] = tree_hidden_states.view(
                num_tokens, -1)

            if self.use_cuda_graph and \
                num_tokens <= self.cudagraph_batch_sizes[-1]:
                num_input_tokens = self.vllm_config.pad_for_cudagraph(
                    num_tokens)
            else:
                num_input_tokens = num_tokens
            # Run the model.
            with set_forward_context(per_layer_attn_metadata,
                                     self.vllm_config,
                                     num_tokens=num_input_tokens):
                last_hidden_states, hidden_states = self.model(
                    input_ids=self.input_ids[:num_input_tokens],
                    positions=self.positions[:num_input_tokens],
                    hidden_states=self.hidden_states[:num_input_tokens],
                    inputs_embeds=None,
                )

            # Get the output hidden states for the draft tokens.
            draft_hidden_states = hidden_states[:num_tokens].view(
                batch_size, query_len, -1)[:, -level_num_drafts:]
            draft_last_hidden_states = last_hidden_states[:num_tokens].view(
                batch_size, query_len, -1)[:, -level_num_drafts:]

            # Get the output logits for the draft tokens.
            logits = self.model.compute_logits(
                draft_last_hidden_states.reshape(batch_size * level_num_drafts,
                                                 -1),
                None,
            )

            # Sample a draft token for each child at the next tree level.
            num_children = self.child_drafts_per_level[level + 1]
            if num_children == 1:
                draft_token_ids = logits.argmax(dim=-1).view(batch_size, -1)
            else:
                draft_token_ids = torch.topk(logits, num_children,
                                             dim=-1).indices.view(
                                                 batch_size, -1)
            draft_token_ids_list.append(draft_token_ids)

            # Update the # drafts counters for the next tree level.
            level_num_drafts = self.cu_drafts_per_level[level +
                                                        1] - total_num_drafts
            total_num_drafts = self.cu_drafts_per_level[level + 1]
        return draft_token_ids_list

    def prepare_inputs(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        # [batch_size]
        num_rejected_tokens: torch.Tensor
    ) -> tuple[CommonAttentionMetadata, torch.Tensor]:
        """
        This function is used to prepare the inputs for the spec decode.
        It updates to the common_attn_metadata to account for the rejected
        tokens (and newly sampled tokens). It also returns the token indices
        of the tokens that should be fed to the speculator.
        """
        # E.g.
        #  common_attn_metadata.query_start_loc{_cpu}:
        #         [0, q1, q1 + q2, q1 + q2 + q3]
        #  common_attn_metadata.seq_lens{_cpu}: [s1, s2, s3]
        #  num_rejected_tokens: [n1, n2, n3]
        # This function computes the intermediate values:
        #  num_tokens_per_req: [q1 - n1, q2 - n2, q3 - n3]
        # And returns:
        #  common_attn_metadata.query_start_loc{_cpu}:
        #         [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        #  common_attn_metadata.seq_lens{_cpu}:
        #         [s1 - n1 + 1, s2 - n2 + 1, s3 - n3 + 1]
        #  token_indices: [0, 1, ..., q1 - n1 - 1,
        #                  q1, q1 + 1, ..., q1 + q2 - n2 - 1,
        #                  q1 + q2, q1 + q2 + 1, ..., q1 + q2 + q3 - n3 - 1]

        device = common_attn_metadata.query_start_loc.device
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        new_seq_lens_cpu = common_attn_metadata.seq_lens_cpu \
            - num_rejected_tokens

        # [0, q1, q1 + q2, q1 + q2 + q3] -> [q1, q2, q3]
        new_query_len_per_req = (query_start_loc_cpu[1:] -
                                 query_start_loc_cpu[:-1])
        # [q1, q2, q3] -> [q1 - n1, q2 - n2, q3 - n3]
        new_num_tokens_per_req = new_query_len_per_req - num_rejected_tokens
        new_num_tokens_per_req_np = new_num_tokens_per_req.numpy()

        # [q1 - n1, q2 - n2, q3 - n3] ->
        # [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        new_query_start_loc_cpu = torch.zeros(
            query_start_loc_cpu.shape,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available())
        new_query_start_loc_np = new_query_start_loc_cpu.numpy()
        np.cumsum(new_num_tokens_per_req_np, out=new_query_start_loc_np[1:])

        total_num_tokens = new_query_start_loc_np[-1]
        # Example assuming num_tokens_per_req_np = [2, 4, 3]
        # this implies that `new_query_start_locs` is:
        # [0, 2, 6, 9] ->
        # [0, 0, 2, 2, 2, 2, 6, 6, 6]
        #  _r1_  ____r2____  ___r3__
        new_query_start_locs_expanded = np.repeat(new_query_start_loc_np[:-1],
                                                  new_num_tokens_per_req_np)
        # [0, 1, 2, 3, 4, 5, 6, 7, 8] ->
        # [0, 1, 0, 1, 2, 3, 0, 1, 2]
        #  _r1_  ____r2____  ___r3__
        token_offests = self.token_arange_np[:total_num_tokens] \
            - new_query_start_locs_expanded

        # Expand starting positions to match token pattern
        # [0, q1, q1 + q2] ->
        # [0, 0, q1, q1, q1, q1, q1 + q2, q1 + q2, q1 + q2]
        #  _r1_  _____r2_______  ___________r3____________
        old_query_start_locs_expanded = np.repeat(
            query_start_loc_cpu[:-1].numpy(), new_num_tokens_per_req_np)
        # Final token indices are:
        # [0, 1,                                   // req 1
        #  q1 + 0, q1 + 1, q1 + 2, q1 + 3,         // req 2
        #  q1 + q2 + 0, q1 + q2 + 1, q1 + q2 + 2]  // req 3
        token_indices_np = token_offests + old_query_start_locs_expanded
        token_indices = torch.from_numpy(token_indices_np).to(
            device, non_blocking=True)

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc_cpu.to(device,
                                                       non_blocking=True),
            seq_lens=new_seq_lens_cpu.to(device, non_blocking=True),
            query_start_loc_cpu=new_query_start_loc_cpu,
            seq_lens_cpu=new_seq_lens_cpu,
            num_computed_tokens_cpu=common_attn_metadata.
            num_computed_tokens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[token_indices],
            causal=True,
        )

        return spec_common_attn_metadata, token_indices

    def load_model(self, target_model: nn.Module) -> None:
        draft_model_config = \
            self.vllm_config.speculative_config.draft_model_config
        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, Attention).keys())

        from vllm.compilation.backends import set_model_tag
        with set_model_tag("eagle_head"):
            self.model = get_model(vllm_config=self.vllm_config,
                                   model_config=draft_model_config)

        draft_attn_layer_names = (
            get_layers_from_vllm_config(self.vllm_config, Attention).keys() -
            target_attn_layer_names)

        self.attn_layer_names = list(draft_attn_layer_names)

        if supports_multimodal(target_model):
            # handle multimodality
            self.model.config.image_token_index = (
                target_model.config.image_token_index)
            target_language_model = target_model.get_language_model()
        else:
            target_language_model = target_model
        # share embed_tokens with the target model if needed
        if get_pp_group().world_size == 1 \
            and self.model.model.embed_tokens.weight.shape \
                == target_language_model.model.embed_tokens.weight.shape:
            logger.info(
                "Assuming the EAGLE head shares the same vocab embedding" \
                " with the target model."
            )
            del self.model.model.embed_tokens
            self.model.model.embed_tokens = (
                target_language_model.model.embed_tokens)
        else:
            logger.info(
                "The EAGLE head's vocab embedding will be loaded separately" \
                " from the target model."
            )

        # share lm_head with the target model if needed
        # some model definition do not define lm_head explicitly
        # and reuse embed_tokens for lm_head, e.g., CohereForCausalLM
        if self.vllm_config.speculative_config.method != "eagle3" and \
                hasattr(target_language_model, "lm_head"):
            logger.info("Loading EAGLE LM head weights from the target model.")
            self.model.lm_head = target_language_model.lm_head

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
    ) -> None:
        with set_forward_context(None, self.vllm_config,
                                 num_tokens=num_tokens):
            if self.is_multimodal_model:
                input_ids = None
                inputs_embeds = self.inputs_embeds[:num_tokens]
            else:
                input_ids = self.input_ids[:num_tokens]
                inputs_embeds = None

            self.model(
                input_ids=input_ids,
                positions=self.positions[:num_tokens],
                hidden_states=self.hidden_states[:num_tokens],
                inputs_embeds=inputs_embeds,
            )

    def validate_same_kv_cache_group(self,
                                     kv_cache_config: KVCacheConfig) -> None:
        """
        Validate that all eagle layers belong to the same KVCacheGroup.
        Need this assumption to ensure all eagle layers can use the
        same AttentionMetadata.
        May extend to multiple AttentionMetadata in the future.
        """
        kv_cache_groups: dict[str, int] = {}
        for id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in kv_cache_group.layer_names:
                kv_cache_groups[layer_name] = id
        assert len(
            set([
                kv_cache_groups[layer_name]
                for layer_name in self.attn_layer_names
            ])
        ) == 1, "All eagle layers should belong to the same kv cache group"


# NOTE(woosuk): Currently, the below code is not used and we always use argmax
# to sample the draft tokens. We will use this after we find a way to manage
# the draft prob tensor.
# Refer to https://github.com/vllm-project/vllm/pull/16899 for the details.
# FIXME(woosuk): The logic here is duplicated with the main sampling code.
# We should refactor this to reuse the same sampling implementation.
def compute_probs_and_sample_next_token(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sampling_metadata.all_greedy:
        # For greedy requests, draft_probs is not used in rejection sampling.
        # Therefore, we can just return the logits.
        probs = logits
        next_token_ids = logits.argmax(dim=-1)
        return next_token_ids, probs

    is_greedy = sampling_metadata.temperature == -1
    temperature = torch.where(is_greedy, 1.0, sampling_metadata.temperature)
    logits.div_(temperature.view(-1, 1))
    probs = logits.softmax(dim=-1, dtype=torch.float32)

    # NOTE(woosuk): Currently, we ignore most of the sampling parameters in
    # generating the draft tokens. We only use the temperature. While this
    # could degrade the acceptance rate, it does not affect the distribution
    # of the generated tokens after rejection sampling.

    # TODO(woosuk): Consider seeds.
    q = torch.empty_like(probs)
    q.exponential_()
    # NOTE(woosuk): We shouldn't use `probs.div_(q)` because the draft_probs
    # will be used later for rejection sampling.
    next_token_ids = probs.div(q).argmax(dim=-1).view(-1)
    if not sampling_metadata.all_random:
        greedy_token_ids = probs.argmax(dim=-1)
        next_token_ids = torch.where(
            is_greedy,
            greedy_token_ids,
            next_token_ids,
        )
    return next_token_ids, probs
