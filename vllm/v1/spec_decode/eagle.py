# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn

from vllm.attention.layer import Attention
from vllm.config import (CompilationLevel, VllmConfig,
                         get_layers_from_vllm_config)
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.v1.attention.backends.flash_attn import (CommonAttentionMetadata,
                                                   FlashAttentionMetadata)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.utils import prepare_eagle_input_kernel

logger = init_logger(__name__)

PADDING_SLOT_ID = -1


class TreeArray:
    """Array-based representation of a tree."""

    def __init__(self, max_nodes: int, device: torch.device):
        """Initialize a tree with arrays to store node data.
        
        Args:
            max_nodes: Maximum number of nodes the tree can hold
            device: Torch device for tensors
        """
        self.max_nodes = max_nodes
        self.device = device

        # Node storage using contiguous tensors for efficient GPU access
        self.token_ids = torch.zeros(max_nodes,
                                     dtype=torch.int64,
                                     device=device)
        self.positions = torch.zeros(max_nodes,
                                     dtype=torch.int64,
                                     device=device)
        self.parent_indices = torch.full(
            (max_nodes, ), -1, dtype=torch.int64,
            device=device)  # Index of parent (-1 for root)
        self.local_probs = torch.zeros(max_nodes,
                                       dtype=torch.float32,
                                       device=device)  # P(token|parent)
        self.global_probs = torch.zeros(max_nodes,
                                        dtype=torch.float32,
                                        device=device)
        self.hidden_states = None

        # Tree structure tracking
        self.depths = torch.full((max_nodes, ),
                                 -1,
                                 dtype=torch.int32,
                                 device=device)

        # Sequence length for each node
        self.seq_lens = torch.ones(max_nodes, dtype=torch.int32, device=device)

        # Number of nodes currently in the tree (also serves as next node index)
        self.size = 0

    def add_node(self,
                 token_id: int,
                 position: int,
                 parent_idx: int,
                 local_prob: float,
                 global_prob: float,
                 hidden_state: torch.Tensor = None,
                 depth: int = 0) -> int:
        """Add a node to the tree.
        
        Args:
            token_id: Token ID for the node
            position: Position in the sequence
            parent_idx: Index of parent node (-1 for root)
            local_prob: Probability of this token given its parent
            global_prob: Joint probability of the path to this node
            hidden_state: Hidden state for this node (optional)
            depth: Depth level of the node
            
        Returns:
            Index of the newly added node
        """
        if self.size >= self.max_nodes:
            return -1

        node_idx = self.size
        self.size += 1

        # Store node data in the corresponding arrays at position node_idx
        self.token_ids[node_idx] = token_id
        self.positions[node_idx] = position
        self.parent_indices[node_idx] = parent_idx
        self.local_probs[node_idx] = local_prob
        self.global_probs[node_idx] = global_prob
        self.depths[node_idx] = depth

        # Update sequence length based on parent
        if parent_idx != -1:
            self.seq_lens[node_idx] = self.seq_lens[parent_idx] + 1
        else:
            self.seq_lens[node_idx] = 1

        # Initialize hidden states tensor if this is the first use
        if hidden_state is not None:
            if self.hidden_states is None:
                hidden_size = hidden_state.shape[-1]
                self.hidden_states = torch.zeros((self.max_nodes, hidden_size),
                                                 dtype=hidden_state.dtype,
                                                 device=self.device)

            if self.hidden_states is not None:
                self.hidden_states[node_idx] = hidden_state

        return node_idx

    def get_nodes_at_depth(self, depth: int) -> list:
        """Get indices of all nodes at a specific depth.
        
        Args:
            depth: The depth to get nodes from (0 = root level)
            
        Returns:
            List of node indices at the specified depth
        """
        nodes = (self.depths[:self.size] == depth).nonzero().flatten().tolist()
        return nodes

    def select_top_k_by_global_prob(self, node_indices: list, k: int) -> list:
        """Select top-k nodes from a list of node indices by global probability.
        
        Args:
            node_indices: List of node indices to choose from
            k: Number of nodes to select
            
        Returns:
            List of selected node indices (highest probability nodes)
        """
        if not node_indices:
            return []

        # Get global probs for these nodes
        indices_tensor = torch.tensor(node_indices, device=self.device)
        probs = self.global_probs[indices_tensor]

        # Select top-k by probability
        k = min(k, len(node_indices))
        if k == 0:
            return []

        # Get indices of highest probability nodes
        _, top_indices = probs.topk(k)
        selected = [node_indices[i] for i in top_indices.tolist()]

        return selected

    def get_selected_nodes_and_ids(self, max_tokens: int) -> list:
        """Get a list of token IDs from the most probable path in the tree.
        
        Args:
            max_tokens: Maximum number of tokens to return
            
        Returns:
            List of token IDs in the selected path
        """
        if self.size == 0:
            return []

        # Start with root (first token)
        tokens = [self.token_ids[0].item()]

        # Build path from root by selecting most probable child at each step
        current_idx = 0
        current_depth = 0

        while len(tokens) < max_tokens and current_depth < self.max_nodes:
            # Find all children of the current node
            children = []
            for i in range(self.size):
                if self.parent_indices[i] == current_idx:
                    children.append(i)

            if not children:
                # Reached a leaf node, can't go deeper
                break

            # Find the child with highest global probability
            best_child = -1
            best_prob = -1
            for child in children:
                if self.global_probs[child] > best_prob:
                    best_prob = self.global_probs[child]
                    best_child = child

            if best_child == -1:
                break

            # Add this token to the path and continue to next depth
            tokens.append(self.token_ids[best_child].item())
            current_idx = best_child
            current_depth += 1

        return tokens


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
        # We need to set +1 here because the arange is used to set
        # query_start_loc, which has one more element than batch_size.
        self.arange = torch.arange(vllm_config.scheduler_config.max_num_seqs +
                                   1,
                                   device=device,
                                   dtype=torch.int32)
        self.device = device

        # Tree-draft specific configuration
        # Use getattr with defaults to ensure backward compatibility
        self.use_tree_draft = getattr(vllm_config.speculative_config,
                                      "use_tree_draft", False)
        self.spec_tree_depth = getattr(vllm_config.speculative_config,
                                       "spec_tree_depth",
                                       self.num_speculative_tokens)
        self.num_spec_expand = getattr(vllm_config.speculative_config,
                                       "num_spec_expand", 2)

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

        if self.method in ["eagle", "eagle3"]:
            # FIXME(woosuk): The below two ops cause synchronization. Optimize.
            max_seq_len = seq_lens.max().item()
            max_num_tokens = (cu_num_tokens[1:] -
                              cu_num_tokens[:-1]).max().item()
            attn_metadata = FlashAttentionMetadata(
                num_actual_tokens=num_tokens,
                max_query_len=max_num_tokens,
                query_start_loc=cu_num_tokens,
                max_seq_len=max_seq_len,
                seq_lens=seq_lens,
                block_table=block_table,
                slot_mapping=target_slot_mapping,
                # TODO(woosuk): Support cascade attention.
                use_cascade=False,
                common_prefix_len=0,
                cu_prefix_query_lens=None,
                prefix_kv_lens=None,
                suffix_kv_lens=None,
            )
        elif self.method == "deepseek_mtp":
            query_lens = cu_num_tokens[1:] - cu_num_tokens[:-1]
            max_query_len = query_lens.max().item()

            common_attn_metadata = CommonAttentionMetadata(
                query_start_loc=cu_num_tokens, seq_lens=seq_lens)

            assert self.runner is not None

            # FIXME: need to consider multiple kv_cache_groups
            attn_metadata = self.runner.attn_metadata_builder.build(
                num_reqs=batch_size,
                num_actual_tokens=num_tokens,
                max_query_len=max_query_len,
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
        else:
            raise ValueError(f"Unsupported method: {self.method}")

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

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            # [batch_size, 1]
            return draft_token_ids.view(-1, 1)

        # Choose between chain-draft and tree-draft based on configuration
        if not self.use_tree_draft:
            # Use the original chain-draft implementation
            return self.chain_draft_propose(draft_token_ids, target_positions,
                                            hidden_states, last_token_indices,
                                            batch_size, block_table,
                                            attn_metadata,
                                            per_layer_attn_metadata)
        else:
            first_token_probs = torch.ones_like(draft_token_ids,
                                                dtype=torch.float32)

            # Use the new tree-draft implementation
            return self.tree_draft_propose(draft_token_ids, first_token_probs,
                                           target_positions, hidden_states,
                                           last_token_indices, batch_size,
                                           block_table, next_token_ids, logits,
                                           per_layer_attn_metadata)

    def chain_draft_propose(
        self,
        draft_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        hidden_states: torch.Tensor,
        last_token_indices: torch.Tensor,
        batch_size: int,
        block_table: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        per_layer_attn_metadata: dict,
    ) -> torch.Tensor:
        """Original chain-draft implementation."""
        # TODO: Currently, MTP module released by deepseek only has
        # one layer. Adapt this code to support multiple layers once
        # there's a multi-layer MTP module.

        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]

        positions = target_positions[last_token_indices]
        hidden_states = hidden_states[last_token_indices]
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
            block_ids = block_table.gather(dim=1,
                                           index=block_numbers.view(-1, 1))
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

            # Run the model.
            with set_forward_context(per_layer_attn_metadata,
                                     self.vllm_config,
                                     num_tokens=input_batch_size):
                last_hidden_states, hidden_states = self.model(
                    self.input_ids[:input_batch_size],
                    self.positions[:input_batch_size],
                    self.hidden_states[:input_batch_size],
                )
            hidden_states = hidden_states[:batch_size]
            logits = self.model.compute_logits(last_hidden_states[:batch_size],
                                               None)

            # TODO(wenlong): get more than one token for tree attention
            draft_token_ids = logits.argmax(dim=-1)
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids

    def tree_draft_propose(
        self,
        draft_token_ids: torch.Tensor,
        first_token_probs: torch.Tensor,
        target_positions: torch.Tensor,
        hidden_states: torch.Tensor,
        last_token_indices: torch.Tensor,
        batch_size: int,
        block_table: torch.Tensor,
        next_token_ids: torch.Tensor,
        logits: torch.Tensor,
        per_layer_attn_metadata: dict,
    ) -> torch.Tensor:
        """Tree-draft implementation using array-based tree structure.
        
        Pseudocode:
            1. Construct the tree structure for each request in the batch
                Construct the root nodes
            2. First forward pass (it happens in the propose(), 
            but we handle the node appending in tree_draft_propose())
                Select top-k highest tokens
                Append them to the corresponding tree
            3. Following forward pass until spec_tree_depth

        Args:
            draft_token_ids: Initial draft tokens for each sequence 
            (first token)
            first_token_probs: Probabilities of the first tokens
            target_positions: Position IDs for each token in the input
            hidden_states: Hidden states from the model
            last_token_indices: Indices of the last token in each sequence
            batch_size: Number of sequences in the batch
            block_table: Mapping from logical positions to 
            physical KV cache blocks
            next_token_ids: Next token ids to use as root nodes
            logits: Logits from the first forward pass
            
        Returns:
            Tensor of draft token IDs with shape 
            [batch_size, num_speculative_tokens]
        """

        #############################################################################
        # STEP 1: Construct the tree structure for each request in the batch
        #         Initialize with root nodes (next_token_ids)
        #############################################################################

        # Calculate maximum tree size based on our expansion algorithm
        # We only expand at most num_spec_expand nodes per level
        # This creates the following pattern:
        # Depth 0: Root node (n=1)
        # Depth 1: Expand root, creating num_spec_expand nodes
        # (n=num_spec_expand)
        # Depth 2+: Expand at most num_spec_expand nodes,
        # each adding num_spec_expand children

        # Calculate node count:
        if self.spec_tree_depth == 0:
            max_nodes_per_tree = 1  # Just root
        elif self.spec_tree_depth == 1:
            max_nodes_per_tree = 1 + self.num_spec_expand  # Root + first level
        else:
            # Root + first level +
            # (remaining levels * num_spec_expand * num_spec_expand)
            max_nodes_per_tree = (1 + self.num_spec_expand +
                                  (self.spec_tree_depth - 1) *
                                  self.num_spec_expand * self.num_spec_expand)

        # Initialize trees for each sequence in the batch
        # Each sequence gets its own tree structure
        trees = []
        for i in range(batch_size):
            tree = TreeArray(max_nodes_per_tree, self.device)
            # Create root node for each tree with the next token id
            # This is the token that was actually selected for this sequence
            tree.add_node(
                token_id=next_token_ids[i].item(),
                position=target_positions[last_token_indices[i]].item() + 1,
                parent_idx=-1,  # Root has no parent
                local_prob=1.0,
                global_prob=1.0,
                hidden_state=hidden_states[last_token_indices[i]],
                depth=0)
            trees.append(tree)

        # Early exit check - handle boundary cases
        if self.spec_tree_depth == 0:
            padded_tokens = torch.zeros(
                (batch_size, self.num_speculative_tokens),
                dtype=torch.int64,
                device=self.device)
            # Fill first token with actual draft token
            padded_tokens[:, 0] = next_token_ids
            return padded_tokens

        #############################################################################
        # STEP 2: Process first forward pass results (already done in propose())
        #         Select top-k highest probability tokens and
        #         add them as level-1 nodes
        #############################################################################

        # Generate level-1 children for each tree using the logits from
        # the first forward pass
        # NOTE(Wenlong): We do not make normalization to make
        # the sum of probabilities equal to 1,
        # We directly use the raw probs as in the figure from EAGLE-2 paper.
        # See also: https://arxiv.org/pdf/2406.16858
        topk = min(self.num_spec_expand, logits.size(-1))
        topk_values, topk_indices = logits.topk(topk, dim=-1)

        # For each tree, add top-k tokens as children of the root node
        for i in range(batch_size):
            root_node_idx = 0  # Root node is always at index 0
            position = trees[i].positions[root_node_idx].item() + 1
            parent_global_prob = trees[i].global_probs[root_node_idx].item()

            # Add k children to the root node
            for j in range(topk):
                token_id = topk_indices[i, j].item()
                token_logit = topk_values[i, j].item()

                trees[i].add_node(token_id=token_id,
                                  position=position,
                                  parent_idx=root_node_idx,
                                  local_prob=token_logit,
                                  global_prob=parent_global_prob * token_logit,
                                  hidden_state=None,
                                  depth=1)

        #############################################################################
        # STEP 3: Subsequent forward passes until reaching spec_tree_depth
        #         Selectively expand most promising nodes at each level
        #
        # STEP 3.1: forward pass
        # STEP 3.2: Select top-k nodes
        #############################################################################

        # Set up for batch processing
        # Determine batch size based on whether CUDA graph optimization is used
        if self.use_cuda_graph and \
            batch_size <= self.cudagraph_batch_sizes[-1]:
            input_batch_size = self.vllm_config.pad_for_cudagraph(batch_size)
        else:
            input_batch_size = batch_size

        # Perform tree expansion for each depth level
        # We iteratively expand the tree, level by level
        for depth in range(self.spec_tree_depth - 1):
            nodes_to_process = []
            for tree_idx, tree in enumerate(trees):
                # Get nodes at the current depth level
                nodes_at_depth = tree.get_nodes_at_depth(depth + 1)
                if nodes_at_depth:
                    # Select top-k nodes by global probability to expand
                    selected_nodes = tree.select_top_k_by_global_prob(
                        nodes_at_depth, self.num_spec_expand)
                    # Add selected nodes with their tree index
                    for node_idx in selected_nodes:
                        nodes_to_process.append((tree_idx, node_idx))

            # Skip if no nodes to process at this level
            if not nodes_to_process:
                break

            # Prepare inputs for model forward pass
            num_process_nodes = len(nodes_to_process)
            batch_input_ids = torch.zeros(num_process_nodes,
                                          dtype=torch.int64,
                                          device=self.device)
            batch_positions = torch.zeros(num_process_nodes,
                                          dtype=torch.int64,
                                          device=self.device)
            batch_hidden_states = torch.zeros(
                (num_process_nodes, self.hidden_size),
                dtype=self.dtype,
                device=self.device)

            # Fill batch tensors with node data
            for i, (tree_idx, node_idx) in enumerate(nodes_to_process):
                tree = trees[tree_idx]
                batch_input_ids[i] = tree.token_ids[node_idx]
                batch_positions[i] = tree.positions[node_idx]
                if tree.hidden_states is not None:
                    batch_hidden_states[i] = tree.hidden_states[node_idx]

            # Handle potential position clipping for model bounds
            exceeds_max_model_len = batch_positions >= self.max_model_len
            clamped_positions = torch.where(exceeds_max_model_len, 0,
                                            batch_positions)

            # For requests exceeding max model length,
            # set seq_len to 1 to minimize attention overhead
            seq_lens = torch.zeros(num_process_nodes,
                                   device=self.device,
                                   dtype=torch.int32)
            for i, (tree_idx, node_idx) in enumerate(nodes_to_process):
                tree = trees[tree_idx]
                # Get sequence length from the node
                seq_lens[i] = tree.seq_lens[node_idx]
            seq_lens.masked_fill_(exceeds_max_model_len, 1)

            # Update max_seq_len considering max model length
            max_seq_len = seq_lens.max().item()
            max_seq_len = min(max_seq_len, self.max_model_len)

            attn_metadata = FlashAttentionMetadata(
                num_actual_tokens=num_process_nodes,
                max_query_len=1,
                query_start_loc=self.arange[:num_process_nodes + 1],
                seq_lens=seq_lens,
                max_seq_len=max_seq_len,
                block_table=block_table,  # Reuse block_table from param
                slot_mapping=None,  # Will be set below
                use_cascade=False,
                common_prefix_len=0,
                cu_prefix_query_lens=None,
                prefix_kv_lens=None,
                suffix_kv_lens=None)

            # Compute the slot mapping for KV cache access
            # This maps logical positions to physical positions in the KV cache
            block_numbers = clamped_positions // self.block_size
            tree_indices = torch.tensor([tree_idx for tree_idx, _ in nodes_to_process], 
                                       device=self.device, dtype=torch.int64)
            block_ids = block_table[tree_indices, block_numbers]
            attn_metadata.slot_mapping = (block_ids * self.block_size + 
                                          clamped_positions % self.block_size)

            # Mask out slots exceeding max model length
            # This prevents updating the KV cache for these positions
            attn_metadata.slot_mapping.masked_fill_(exceeds_max_model_len,
                                                    PADDING_SLOT_ID)

            # Create per_layer_attn_metadata for this forward pass
            tree_per_layer_attn_metadata = {}
            for layer_name in self.attn_layer_names:
                tree_per_layer_attn_metadata[layer_name] = attn_metadata

            # Copy inputs to buffer for cudagraph
            self.input_ids[:num_process_nodes] = batch_input_ids
            self.positions[:num_process_nodes] = clamped_positions

            # Set up hidden states based on model type
            if self.method == 'eagle':
                self.hidden_states[:num_process_nodes] = batch_hidden_states
                forward_hidden_states = self.hidden_states
            else:
                forward_hidden_states = batch_hidden_states

            # Run model forward pass to get next token predictions
            with set_forward_context(tree_per_layer_attn_metadata,
                                     self.vllm_config,
                                     num_tokens=input_batch_size):
                last_hidden_states, output_hidden_states = self.model(
                    input_ids=self.input_ids[:input_batch_size],
                    positions=self.positions[:input_batch_size],
                    hidden_states=forward_hidden_states[:input_batch_size],
                )

            # Compute logits and get token probabilities
            # This gives us the distribution over vocabulary for each node
            logits = self.model.compute_logits(last_hidden_states, None)

            # Get top-k predictions for each node
            # We'll expand each node with its k most likely continuations
            topk = min(self.num_spec_expand, logits.size(-1))
            topk_values, topk_indices = logits.topk(topk, dim=-1)

            # For each tree, select the nodes to actually expand based
            # on global probability
            for tree_idx, tree in enumerate(trees):
                # Get indices of all nodes for this tree that were processed
                tree_node_indices = [
                    node_idx
                    for i, (t_idx, node_idx) in enumerate(nodes_to_process)
                    if t_idx == tree_idx
                ]
                process_indices = [
                    i for i, (t_idx, _) in enumerate(nodes_to_process)
                    if t_idx == tree_idx
                ]

                # If no nodes for this tree, skip
                if not tree_node_indices:
                    continue

                # Select top nodes to expand (based on global probability)
                # This limits expansion to only the most promising paths
                nodes_to_expand = tree.select_top_k_by_global_prob(
                    tree_node_indices, self.num_spec_expand)

                # For each selected node, add its top-k token expansions
                for node_idx in nodes_to_expand:
                    # Find the position in the process batch
                    process_idx = process_indices[tree_node_indices.index(
                        node_idx)]

                    # Get top-k tokens and probabilities for this node
                    node_topk_values = topk_values[process_idx]
                    node_topk_indices = topk_indices[process_idx]

                    # Add children nodes for each of the top-k tokens
                    for j in range(len(node_topk_indices)):
                        token_id = node_topk_indices[j].item()
                        token_logit = node_topk_values[j].item()

                        # Calculate new position (next position after parent)
                        position = tree.positions[node_idx].item() + 1

                        # Get global probability
                        # (parent's global_prob * this token's prob)
                        parent_global_prob = tree.global_probs[node_idx].item()
                        global_prob = parent_global_prob * token_logit

                        # Add new node to tree
                        hidden_state = (last_hidden_states[process_idx]
                                        if output_hidden_states is None else
                                        output_hidden_states[process_idx])
                        tree.add_node(
                            token_id=token_id,
                            position=position,
                            parent_idx=node_idx,
                            # Using logit directly as local_prob
                            local_prob=token_logit,
                            # Simple multiplication for path probability
                            global_prob=global_prob,
                            hidden_state=hidden_state,
                            depth=depth + 2)
        # Select final draft tokens from each tree and collect them
        # Now that we've built the trees, extract the most likely paths
        draft_tokens_list = []
        max_tokens = 0

        for tree in trees:
            # Get the most probable path through each tree
            tokens = tree.get_selected_nodes_and_ids(
                self.num_speculative_tokens)
            draft_tokens_list.append(tokens)
            max_tokens = max(max_tokens, len(tokens))

        # Pad sequences to same length
        # Ensure all sequences have the same length for batch processing
        padded_draft_tokens = []
        for tokens in draft_tokens_list:
            # If we have fewer tokens than required, pad with zeros
            padded = tokens + [0] * (max_tokens - len(tokens))
            padded_draft_tokens.append(padded[:self.num_speculative_tokens])

        # Convert to tensor with correct format
        # Return tensor of shape [batch_size, num_speculative_tokens]
        draft_token_ids = torch.tensor(padded_draft_tokens,
                                       dtype=torch.int64,
                                       device=self.device)

        return draft_token_ids

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
        query_len_per_req = (cu_target_query_lens[1:] -
                             cu_target_query_lens[:-1])
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
        draft_model_config = \
            self.vllm_config.speculative_config.draft_model_config
        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, Attention).keys())

        self.model = get_model(vllm_config=self.vllm_config,
                               model_config=draft_model_config)

        draft_attn_layer_names = (
            get_layers_from_vllm_config(self.vllm_config, Attention).keys() -
            target_attn_layer_names)

        self.attn_layer_names = list(draft_attn_layer_names)

        # share embed_tokens with the target model if needed
        if get_pp_group().world_size == 1:
            logger.info(
                "The EAGLE head shares the same vocab embedding" \
                " with the target model."
            )
            self.model.model.embed_tokens = target_model.model.embed_tokens
        else:
            logger.info(
                "Since PP > 1, the EAGLE head loaded its own vocab embedding" \
                " weights instead of sharing them with the target model."
            )

        # share lm_head with the target model if needed
        # some model definition do not define lm_head explicitly
        # and reuse embed_tokens for lm_head, e.g., CohereForCausalLM
        if self.vllm_config.speculative_config.method != "eagle3" and \
                hasattr(target_model, "lm_head"):
            logger.info("Loading EAGLE LM head weights from the target model.")
            self.model.lm_head = target_model.lm_head

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
