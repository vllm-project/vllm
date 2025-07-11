# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast

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
from vllm.v1.attention.backends.flash_attn import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.utils import prepare_eagle_input_kernel

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
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        # We need to get the hidden size from the draft model config because
        # the draft model's hidden size can be different from the target model's
        # hidden size (e.g., Llama 3.3 70B).
        self.hidden_size = self.draft_model_config.get_hidden_size()

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
        # We need +1 here because the arange is used to set query_start_loc,
        # which has one more element than batch_size.
        self.arange = torch.arange(
            vllm_config.scheduler_config.max_num_seqs + 1,
            device=device,
            dtype=torch.int32,
        )

        spec_token_tree = vllm_config.speculative_config.speculative_token_tree
        self.tree_choices: list[tuple[int,
                                      ...]] = ast.literal_eval(spec_token_tree)
        tree_depth = len(self.tree_choices[-1])
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

        max_batch_size = vllm_config.scheduler_config.max_num_seqs
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
        # [num_tokens]
        target_slot_mapping: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        # [batch_size + 1] starting with 0
        cu_num_tokens: torch.Tensor,
        # [batch_size, max_num_blocks_per_req]
        block_table: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]
        last_token_indices = cu_num_tokens[1:] - 1

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

        # FA requires seq_len to have dtype int32.
        seq_lens = (target_positions[last_token_indices] + 1).int()

        if self.method in ["eagle", "eagle3", "deepseek_mtp"]:
            query_lens = cu_num_tokens[1:] - cu_num_tokens[:-1]
            max_query_len = query_lens.max().item()
            common_attn_metadata = CommonAttentionMetadata(
                query_start_loc=cu_num_tokens,
                seq_lens=seq_lens,
                num_reqs=batch_size,
                num_actual_tokens=num_tokens,
                max_query_len=max_query_len,
            )

            assert self.runner is not None

            # FIXME: need to consider multiple kv_cache_groups
            attn_metadata_builder = self.runner.attn_metadata_builders[0]
            attn_metadata = attn_metadata_builder.build_for_drafting(
                common_attn_metadata=common_attn_metadata,
                current_step=0,
                total_tokens_generated=0,
            )
        else:
            raise ValueError(f"Unsupported method: {self.method}")

        # At this moment, we assume all eagle layers belong to the same KV
        # cache group, thus using the same attention metadata.
        per_layer_attn_metadata = {}
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata

        # Copy inputs to buffer for cudagraph.
        self.positions[:num_tokens] = target_positions
        self.hidden_states[:num_tokens] = target_hidden_states

        if self.num_speculative_tokens == 1:
            # Early exit if there is only one draft token to be generated.
            if self.use_cuda_graph and num_tokens <= self.cudagraph_batch_sizes[
                    -1]:
                num_input_tokens = self.vllm_config.pad_for_cudagraph(
                    num_tokens)
            else:
                num_input_tokens = num_tokens

            with set_forward_context(per_layer_attn_metadata,
                                     self.vllm_config,
                                     num_tokens=num_input_tokens):
                ret_hidden_states = self.model(
                    self.input_ids[:num_input_tokens],
                    self.positions[:num_input_tokens],
                    self.hidden_states[:num_input_tokens],
                )
                if self.method == "deepseek_mtp":
                    last_hidden_states = ret_hidden_states
                else:
                    last_hidden_states, hidden_states = ret_hidden_states
            sample_hidden_states = last_hidden_states[last_token_indices]
            logits = self.model.compute_logits(sample_hidden_states, None)
            draft_token_ids = logits.argmax(dim=-1)
            # [batch_size, 1]
            return draft_token_ids.view(-1, 1)

        # TODO: Currently, MTP module released by deepseek only has
        # one layer. Adapt this code to support multiple layers once
        # there's a multi-layer MTP module.

        generated_tokens_list = []
        input_ids = torch.empty(0)
        positions = torch.empty(0)
        hidden_states = torch.empty(0)
        base_positions = target_positions[last_token_indices]
        flattened_draft_positions = (
            base_positions.view(batch_size, -1) +
            self.tree_draft_pos_offsets[:batch_size, :])
        tree_depth = len(self.cu_drafts_per_level)
        total_num_drafts = 0
        for level in range(tree_depth):
            if self.use_cuda_graph and num_tokens <= self.cudagraph_batch_sizes[
                    -1]:
                num_input_tokens = self.vllm_config.pad_for_cudagraph(
                    num_tokens)
            else:
                num_input_tokens = num_tokens

            # Run the model.
            with set_forward_context(per_layer_attn_metadata,
                                     self.vllm_config,
                                     num_tokens=num_input_tokens):
                last_hidden_states, _ = self.model(
                    self.input_ids[:num_input_tokens],
                    self.positions[:num_input_tokens],
                    self.hidden_states[:num_input_tokens],
                )
            last_hidden_states = last_hidden_states[:batch_size]
            logits = self.model.compute_logits(last_hidden_states, None)

            num_level_drafts = self.cu_drafts_per_level[
                level] - total_num_drafts
            total_num_drafts = self.cu_drafts_per_level[level]
            num_child_drafts = self.child_drafts_per_level[level]

            # Sample a draft token for each child.
            draft_token_ids = torch.topk(logits, num_child_drafts,
                                         dim=-1).indices
            # Get draft positions for RoPE.
            draft_positions = base_positions + (level + 1)
            exceeds_max_model_len = (base_positions +
                                     total_num_drafts) >= self.max_model_len
            # Mask out the position ids that exceed the max model length.
            # Otherwise, we may get out-of-range error in RoPE.
            clamped_draft_positions = torch.where(
                exceeds_max_model_len,
                0,
                draft_positions,
            )
            draft_positions = clamped_draft_positions.repeat_interleave(
                num_level_drafts).reshape(batch_size, -1)
            # Get draft hidden states.
            draft_hidden_states = last_hidden_states.repeat_interleave(
                num_child_drafts, dim=1).reshape(batch_size, 1, -1)

            generated_tokens_list.append(draft_token_ids)
            if total_num_drafts > (level + 1):
                # Draft branching has occurred. The drafted tokens must be used
                # as query tokens for future tree attention ops.
                query_len = total_num_drafts
                input_ids = torch.cat([input_ids, draft_token_ids], dim=1)
                positions = torch.cat([positions, draft_positions], dim=1)
                hidden_states = torch.cat([hidden_states, draft_hidden_states],
                                          dim=1)
            else:
                # Still a chain of drafts. Continue performing standard attention.
                query_len = 1
                input_ids = draft_token_ids
                positions = draft_positions
                hidden_states = draft_hidden_states

            if level == (tree_depth - 1):
                # Skip the attention metadata processing below because we already
                # have the outputs we need.
                break

            # Update the attention metadata to reflect the newly drafted tokens.
            common_attn_metadata.num_actual_tokens = batch_size * query_len
            common_attn_metadata.max_query_len = query_len
            common_attn_metadata.query_start_loc = (
                query_len * self.arange[:batch_size + 1])
            common_attn_metadata.seq_lens += num_level_drafts

            # Rebuild the attn metadata.
            attn_metadata = attn_metadata_builder.build_for_drafting(
                common_attn_metadata=common_attn_metadata,
                current_step=level + 1,
                total_tokens_generated=total_num_drafts,
            )
            # Consider max model length.
            attn_metadata.max_seq_len = min(attn_metadata.max_seq_len,
                                            self.max_model_len)
            # For the requests that exceed the max model length, we set the
            # sequence length to 1 to minimize their overheads in attention.
            attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)

            # Compute the slot mapping.
            query_positions = flattened_draft_positions[:, :query_len]
            block_numbers = query_positions // self.block_size
            block_ids = block_table.gather(dim=1, index=block_numbers)
            slot_mapping = (block_ids * self.block_size +
                            query_positions % self.block_size)
            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            slot_mapping[exceeds_max_model_len] = PADDING_SLOT_ID
            attn_metadata.slot_mapping = slot_mapping.view(-1)

            # At this moment, we assume all eagle layers belong to the same KV
            # cache group, thus using the same attention metadata.
            per_layer_attn_metadata = {}
            for layer_name in self.attn_layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

            # Copy inputs to buffer for cudagraph.
            num_tokens = common_attn_metadata.num_actual_tokens
            self.input_ids[:num_tokens] = input_ids.view(-1)
            self.positions[:num_tokens] = positions.view(-1)
            self.hidden_states[:num_tokens] = hidden_states.view(
                num_tokens, -1)

        # [batch_size, num_speculative_tokens]
        return torch.cat(generated_tokens_list, dim=1)

    @staticmethod
    def prepare_inputs(
        # [batch_size + 1]
        cu_target_query_lens: torch.Tensor,
        # [batch_size]
        num_rejected_tokens: torch.Tensor,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # cu_target_query_lens: [0, a, a + b, a + b + c]
        # num_rejected_tokens: [n1, n2, n3]
        # num_tokens_per_req: [a - n1, b - n2, c - n3]
        # cu_num_tokens: [0, a - n1, a + b - n1 - n2, a + b + c - n1 - n2 - n3]
        # token_indices: [0, 1, ..., a - n1 - 1,
        #                 a, a + 1, ..., a + b - n2 - 1,
        #                 a + b, a + b + 1, ..., a + b + c - n3 - 1]

        # [0, a, a + b, a + b + c] -> [a, b, c]
        query_len_per_req = cu_target_query_lens[1:] - cu_target_query_lens[:-1]
        # [a, b, c] -> [a - n1, b - n2, c - n3]
        num_tokens_per_req = query_len_per_req - num_rejected_tokens

        # [a - n1, b - n2, c - n3] ->
        # [0, a - n1, a + b - n1 - n2, a + b + c - n1 - n2 - n3]
        cu_num_tokens = torch.zeros_like(cu_target_query_lens)
        torch.cumsum(num_tokens_per_req, dim=0, out=cu_num_tokens[1:])
        token_indices = torch.empty(
            num_tokens,
            dtype=torch.int32,
            device=cu_target_query_lens.device,
        )
        batch_size = num_rejected_tokens.shape[0]
        BLOCK_SIZE = 1024
        prepare_eagle_input_kernel[(batch_size, )](
            token_indices,
            cu_target_query_lens,
            cu_num_tokens,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return cu_num_tokens, token_indices

    def load_model(self, target_model: nn.Module) -> None:
        draft_model_config = self.vllm_config.speculative_config.draft_model_config
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
            self.model.config.image_token_index = target_model.config.image_token_index
            target_language_model = target_model.get_language_model()
        else:
            target_language_model = target_model
        # share embed_tokens with the target model if needed
        if (get_pp_group().world_size == 1
                and self.model.model.embed_tokens.weight.shape
                == target_language_model.model.embed_tokens.weight.shape):
            logger.info(
                "Assuming the EAGLE head shares the same vocab embedding"
                " with the target model.")
            del self.model.model.embed_tokens
            self.model.model.embed_tokens = target_language_model.model.embed_tokens
        else:
            logger.info(
                "The EAGLE head's vocab embedding will be loaded separately"
                " from the target model.")

        # share lm_head with the target model if needed
        # some model definition do not define lm_head explicitly
        # and reuse embed_tokens for lm_head, e.g., CohereForCausalLM
        if self.vllm_config.speculative_config.method != "eagle3" and hasattr(
                target_language_model, "lm_head"):
            logger.info("Loading EAGLE LM head weights from the target model.")
            self.model.lm_head = target_language_model.lm_head

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
    ) -> None:
        with set_forward_context(None, self.vllm_config,
                                 num_tokens=num_tokens):
            self.model(
                self.input_ids[:num_tokens],
                self.positions[:num_tokens],
                self.hidden_states[:num_tokens],
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
        assert (len(
            set([
                kv_cache_groups[layer_name]
                for layer_name in self.attn_layer_names
            ])) == 1
                ), "All eagle layers should belong to the same kv cache group"


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
