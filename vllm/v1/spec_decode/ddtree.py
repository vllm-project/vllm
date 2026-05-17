# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DDTree: Diffusion Draft Tree for speculative decoding.

Implements the tree construction and traversal algorithms from:
  "Accelerating Speculative Decoding with Block Diffusion Draft Trees"
  Ringel & Romano, arXiv:2604.12989

DDTree builds a draft tree from DFlash's per-position probability
distributions using a best-first heap, then verifies the whole tree
in a single target-model forward pass using ancestor-only attention.
"""

import heapq
import os

import numpy as np
import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.tree_attn import TreeAttentionMetadataBuilder
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

logger = init_logger(__name__)


def build_ddtree_tree(
    draft_logits: torch.Tensor,
    budget: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[int],
    list[dict[int, int]],
    torch.Tensor,
]:
    """Build a draft tree from DFlash per-position logits.

    Uses a best-first heap to select the ``budget`` most-probable token
    paths according to the draft model's output distributions.

    Args:
        draft_logits: Float tensor of shape ``[depth, vocab_size]``.
            Raw (un-softmaxed) logits for each speculative position,
            as produced by the DFlash draft model.
        budget: Maximum number of non-root tree nodes to expand.

    Returns:
        node_token_ids: int64[num_nodes] — token id at each non-root node.
        node_depths:    int64[num_nodes] — 1-based depth (root=0, children=1, …).
        node_ranks:     int64[num_nodes] — top-k rank at each node's depth position.
        parents:    list[num_nodes+1] — parent index per node; parents[0]==-1 (root).
        child_maps: list[num_nodes+1] of dicts — child_maps[i][token_id] = child index.
        visibility:     bool[num_nodes+1, num_nodes+1] — ancestor-only attention mask.
    """
    if budget <= 0 or draft_logits.shape[0] == 0:
        visibility = torch.zeros((1, 1), dtype=torch.bool)
        visibility[0, 0] = True
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            [-1],
            [{}],
            visibility,
        )

    depth_limit = int(draft_logits.shape[0])
    topk = min(budget, draft_logits.shape[-1])

    # Compute normalised log-probabilities for the top-k tokens at each
    # position.  Move to CPU immediately; the heap runs on the CPU.
    logits = draft_logits.float()
    top_logits, top_token_ids = torch.topk(logits, k=topk, dim=-1)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    top_log_probs_np = (
        (top_logits - log_z).to(device="cpu", dtype=torch.float32).numpy()
    )
    top_token_ids_np = top_token_ids.to(device="cpu", dtype=torch.long).numpy()

    node_token_ids_np = np.empty(budget, dtype=np.int64)
    node_depths_np = np.empty(budget, dtype=np.int64)
    node_ranks_np = np.empty(budget, dtype=np.int64)
    # parents_np[0] == -1 (root); parents_np[i] for i >= 1 is the
    # parent node index.
    parents_np = np.empty(budget + 1, dtype=np.int32)
    parents_np[0] = -1
    child_maps: list[dict[int, int]] = [{}]
    node_count = 0

    # Best-first heap. Each entry is a candidate node not yet added to the tree.
    # Entry: (-logw, ranks, parent_index, depth, rank, logw)
    #
    #   -logw        — negated accumulated path log-prob; min-heap pops the
    #                  highest log-prob candidate first
    #   ranks        — tuple of top-k indices taken at each depth along this
    #                  path, e.g. (0, 1) = rank-0 at depth 1, rank-1 at depth 2;
    #                  used only as a tiebreaker when two entries have equal logw
    #   parent_index — insertion-order index of this candidate's parent node
    #                  (root=0, first inserted node=1, second=2, ...)
    #   depth        — 1-based depth of this candidate (root is depth 0)
    #   rank         — index k into top_token_ids_np[depth-1, k] for this token;
    #                  rank 0 = most probable token at this depth position
    #   logw         — accumulated path log-prob (non-negated); kept separately
    #                  so sibling/child expansions can do arithmetic on the raw
    #                  sum without re-extracting it from the negated first field
    first_logw = float(top_log_probs_np[0, 0])
    heap: list[tuple] = [(-first_logw, (0,), 0, 1, 0, first_logw)]

    while heap and node_count < budget:
        _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)
        token_id = int(top_token_ids_np[depth - 1, rank])
        current_index = node_count + 1

        node_token_ids_np[node_count] = token_id
        node_depths_np[node_count] = depth
        node_ranks_np[node_count] = rank
        parents_np[current_index] = parent_index
        child_maps.append({})
        child_maps[parent_index][token_id] = current_index
        node_count += 1

        # Push sibling: same parent, next rank at the same depth.
        if rank + 1 < topk:
            sibling_logw = (
                logw
                - float(top_log_probs_np[depth - 1, rank])
                + float(top_log_probs_np[depth - 1, rank + 1])
            )
            heapq.heappush(
                heap,
                (
                    -sibling_logw,
                    ranks[:-1] + (rank + 1,),
                    parent_index,
                    depth,
                    rank + 1,
                    sibling_logw,
                ),
            )

        # Push first child: go one level deeper, take rank-0 token.
        if depth < depth_limit:
            child_logw = logw + float(top_log_probs_np[depth, 0])
            heapq.heappush(
                heap,
                (
                    -child_logw,
                    ranks + (0,),
                    current_index,
                    depth + 1,
                    0,
                    child_logw,
                ),
            )

    # Build the ancestor-only visibility (attention) mask.
    # visibility[i, j] == True  iff  j is an ancestor of i (or j == i).
    current_length = 1 + node_count
    visibility_np = np.zeros((current_length, current_length), dtype=np.bool_)
    visibility_np[0, 0] = True
    for idx in range(1, current_length):
        p = int(parents_np[idx])
        visibility_np[idx, :idx] = visibility_np[p, :idx]
        visibility_np[idx, idx] = True

    return (
        torch.from_numpy(node_token_ids_np[:node_count]),
        torch.from_numpy(node_depths_np[:node_count]),
        torch.from_numpy(node_ranks_np[:node_count]),
        parents_np[:current_length].tolist(),
        child_maps,
        torch.from_numpy(visibility_np),
    )


def follow_verified_tree(
    child_maps: list[dict[int, int]],
    posterior_token_ids: list[int],
) -> tuple[list[int], int]:
    """Walk the verified tree to find the longest accepted path.

    After the target model runs a forward pass over the whole tree, this
    function greedily follows the path of accepted tokens from the root.

    Args:
        child_maps: As returned by :func:`build_ddtree_tree`.
        posterior_token_ids: List of token ids sampled from the target
            model's logits, one per tree node (root included, in node
            index order).

    Returns:
        accepted_indices: Node indices (including root at 0) that form
            the accepted prefix.
        bonus_token_id: The next token id to emit after the accepted
            prefix (sampled by the target model at the last accepted
            node).
    """
    accepted_indices = [0]
    current_index = 0
    next_token = int(posterior_token_ids[current_index])

    while next_token in child_maps[current_index]:
        current_index = child_maps[current_index][next_token]
        accepted_indices.append(current_index)
        next_token = int(posterior_token_ids[current_index])

    return accepted_indices, next_token


def ddtree_verify(
    logits: torch.Tensor,
    target_logits_indices: torch.Tensor,
    bonus_logits_indices: torch.Tensor,
    draft_token_ids: torch.Tensor,
    child_maps: list[list[dict[int, int]]],
    budget: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Tree-aware verification for DDTree speculative decoding.

    After the target model's forward pass over all tree nodes (with
    ancestor-only attention), this function traces the longest accepted
    path through the tree per request using the target model's per-node
    greedy predictions.

    In contrast to flat spec-decode rejection sampling (which scans
    draft tokens left-to-right), DDTree must follow tree edges: after
    accepting a node, the next comparison is against that node's
    children, not its siblings.

    Args:
        logits:                float[num_logits, vocab_size] — target model logits.
        target_logits_indices: int[batch * budget] — logit row per draft node.
        bonus_logits_indices:  int[batch] — logit row for the last node per request.
        draft_token_ids:       int[batch * budget] — draft tokens in tree-node order.
        child_maps:            list[batch] of per-request dicts from build_ddtree_tree.
        budget:                number of draft nodes per request (root excluded).
        batch_size:            number of requests in the batch.
        device:                target device for the returned tensor.

    Returns:
        int32[batch, budget+1] — accepted path tokens + bonus token, -1 padded.
    """

    # posterior[r][i]   = target's argmax at node i's position, req r
    # posterior[r][B]   = target's argmax at the last node (bonus), req r
    # Example (batch=2, budget=3, vocab=3):
    #   target_logits_indices = [0, 1, 2, 3, 4, 5]   # 6 rows, one per (req, node) pair
    #
    #   logits[0] = [0.1, 0.9, 0.1] - token 1   (req 0, node 0)
    #   logits[1] = [0.5, 0.1, 0.3] - token 0   (req 0, node 1)
    #   logits[2] = [0.1, 0.1, 0.6] - token 2   (req 0, node 2)
    #   logits[3] = [0.8, 0.1, 0.1] - token 0   (req 1, node 0)
    #   logits[4] = [0.1, 0.6, 0.1] - token 1   (req 1, node 1)
    #   logits[5] = [0.2, 0.1, 0.7] - token 2   (req 1, node 2)
    #
    #   .argmax()  = [1, 0, 2, 0, 1, 2]
    #   .view(2,3) = [[1, 0, 2],   <- req 0
    #                 [0, 1, 2]]   <- req 1
    node_posterior = (
        logits[target_logits_indices].argmax(dim=-1).view(batch_size, budget)
    )
    bonus_posterior = logits[bonus_logits_indices].argmax(dim=-1)

    node_posterior_cpu = node_posterior.cpu().tolist()
    bonus_posterior_cpu = bonus_posterior.cpu().tolist()
    draft_tokens_cpu = draft_token_ids.view(batch_size, budget).cpu().tolist()

    _dbg = os.environ.get("DDTREE_DEBUG") == "1"

    output = torch.full((batch_size, budget + 1), -1, dtype=torch.int32)

    for r in range(batch_size):
        posterior = node_posterior_cpu[r] + [bonus_posterior_cpu[r]]

        accepted_indices, bonus_token = follow_verified_tree(child_maps[r], posterior)

        if _dbg and r == 0:
            print(
                f"[ddtree_verify] draft={draft_tokens_cpu[r]}"
                f" posterior={posterior[:budget]}"
                f" acc_len={len(accepted_indices)}"
                f" maps={child_maps[r]}"
            )

        out_pos = 0
        for node_idx in accepted_indices[1:]:
            output[r, out_pos] = draft_tokens_cpu[r][node_idx - 1]
            out_pos += 1
        output[r, out_pos] = bonus_token

    return output.to(device)


class DDTreeProposer(DFlashProposer):
    """DFlash proposer with a dynamic best-first draft tree.

    Each proposal step:
    1. Runs the DFlash draft model to obtain per-position logits.
    2. Calls :func:`build_ddtree_tree` per request on its own draft logits
       to select the ``budget`` most-probable tree nodes via a best-first heap.
    3. Updates the target model's ``TreeAttentionMetadataBuilder`` with
       the new visibility mask so the next verification pass uses the
       correct ancestor-only attention bias.
    4. Returns per-request draft tokens in tree-node order.

    Each request gets its own tree topology derived from its own draft logits.

    Requirements:
    - attention_config.backend = "TREE_ATTN": needed for per-request
      ancestor-only attention masking over the draft tree.
    - speculative_config.method = "ddtree": selects this proposer.

    Example (budget=4, num_speculative_tokens=4):

        DFlash produces marginal logits for 4 depth positions.
        For this example, assume probabilities are concentrated enough
        that the tree only branches to depth 2.
        root token = "The"

        Top-5 most probable sequences by cumulative log-prob:
          rank 1: "The" -> cat -> sat
          rank 2: "The" -> cat -> ran
          rank 3: "The" -> dog -> sat
          rank 4: "The" -> cat -> red
          rank 5: "The" -> dog -> ran

        budget=4 means: expand top-4 nodes from the heap:
          pop 1: cat   (root->cat)       -> node 1
          pop 2: sat   (root->cat->sat)  -> node 2  child of node 1
          pop 3: dog   (root->dog)       -> node 3
          pop 4: sat   (root->dog->sat)  -> node 4  child of node 3
          stop.  rank 5 (dog->ran) NOT in tree — out of budget.

                    root ("The")          <- node 0
                   /            \\
                cat               dog     <- node 1, node 3
                 |                 |
                sat               sat    <- node 2, node 4

        child_maps:
          child_maps[0] = {cat: 1, dog: 3}
          child_maps[1] = {sat: 2}
          child_maps[2] = {}                  <- leaf
          child_maps[3] = {sat: 4}
          child_maps[4] = {}                  <- leaf

        draft output: [batch, budget=4] in node-index order
          [cat, sat, dog, sat]
           ^1   ^2   ^3   ^4

        visibility mask [5, 5]  (budget+1 nodes including root)
        rule: row i attends to col j iff j is an ancestor of i (or j==i)

              0  1  2  3  4
          0 [ T  F  F  F  F ]  root
          1 [ T  T  F  F  F ]  cat           (sees root, itself)
          2 [ T  T  T  F  F ]  sat under cat (sees root, cat, itself)
          3 [ T  F  F  T  F ]  dog           (sees root, itself)
          4 [ T  F  F  T  T ]  sat under dog (sees root, dog, itself)

        note: nodes 2 and 4 are both "sat" but attend to different contexts.

        verification examples:
          target predicts [cat, sat, ...]:
            node 0 -> cat in child_maps[0] -> node 1  accepted
            node 1 -> sat in child_maps[1] -> node 2  accepted
            node 2 -> no children          -> stop
            result: "The cat sat" + bonus token

          target predicts [dog, sat, ...]:
            node 0 -> dog in child_maps[0] -> node 3  accepted
            node 3 -> sat in child_maps[3] -> node 4  accepted
            node 4 -> no children          -> stop
            result: "The dog sat" + bonus token
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "ddtree"
        super().__init__(vllm_config, device, runner)

        self._runner = runner
        self._budget = self.num_speculative_tokens
        self._child_maps: list[list[dict[int, int]]] | None = None
        self._node_depths: list[torch.Tensor] | None = None

    @override
    def build_per_group_and_layer_attn_metadata(
        self,
        cad: CommonAttentionMetadata,
        draft_index: int = 0,
    ) -> tuple[list[object], dict[str, object]]:
        # Skip DFlashProposer's causal=False assertion; tree attention enforces
        # non-causal masking via qq_bias, not via the causal flag.
        return SpecDecodeBaseProposer.build_per_group_and_layer_attn_metadata(
            self, cad, draft_index
        )

    @override
    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Build a per-request dynamic tree and return draft tokens.

        Overrides the DFlash greedy argmax with a heap-based tree built
        independently for each request from its own logits.  Each request
        gets its own tree topology (child_maps) and visibility mask, allowing
        heterogeneous batches to have different speculative paths.

        Args:
            hidden_states: [batch * depth, hidden_size] — the DFlash
                model's output hidden states for the speculative positions.

        Returns:
            [batch * budget] int64 — flattened draft token IDs in
            tree-node order, varying per request.
        """
        depth = self.num_speculative_tokens  # DFlash depth == num_spec_tokens
        batch_size = hidden_states.shape[0] // depth

        logits = self.model.compute_logits(hidden_states)
        vocab_size = logits.shape[-1]
        logits_per_req = logits.float().view(batch_size, depth, vocab_size)

        all_child_maps: list[list[dict[int, int]]] = []
        all_draft_tokens: list[torch.Tensor] = []
        all_visibility: list[torch.Tensor] = []
        all_node_depths: list[torch.Tensor] = []
        target_size = self._budget + 1  # [N+1, N+1] per request

        for r in range(batch_size):
            (
                node_token_ids,
                node_depths,
                _,
                _,
                child_maps,
                visibility,
            ) = build_ddtree_tree(logits_per_req[r], budget=self._budget)
            all_child_maps.append(child_maps)

            # Draft tokens for this request, padded to budget.
            tokens = node_token_ids.to(self.device)
            budget_actual = tokens.shape[0]
            if budget_actual < self._budget:
                tokens = torch.cat(
                    [tokens, tokens.new_zeros(self._budget - budget_actual)]
                )
            all_draft_tokens.append(tokens)

            # Node depths padded to budget; unused slots get sequential depths
            # so their positions stay consistent (non-zero depth avoids root clash).
            depths = node_depths.to(self.device, dtype=torch.long)
            if budget_actual < self._budget:
                pad = torch.arange(
                    budget_actual + 1,
                    budget_actual + 1 + (self._budget - budget_actual),
                    dtype=torch.long,
                    device=self.device,
                )
                depths = torch.cat([depths, pad])
            all_node_depths.append(depths)

            # Visibility mask padded to [target_size, target_size].
            vis_size = visibility.shape[0]  # budget_actual + 1
            if vis_size < target_size:
                padded = torch.zeros(target_size, target_size, dtype=torch.bool)
                padded[:vis_size, :vis_size] = visibility
                visibility = padded
            all_visibility.append(visibility)

        self._child_maps = all_child_maps
        self._node_depths = all_node_depths

        # Stack per-request masks: [batch, N+1, N+1] → 3D qq_bias.
        stacked = torch.stack(all_visibility, dim=0)  # [batch, N+1, N+1]
        tree_attn_bias = torch.where(
            stacked.to(self.device),
            torch.zeros(1, dtype=torch.float32, device=self.device),
            torch.full((1,), float("-inf"), dtype=torch.float32, device=self.device),
        )
        self._update_target_tree_attn_bias(tree_attn_bias)

        draft = torch.stack(all_draft_tokens, dim=0)  # [batch, budget]
        return draft.reshape(-1).to(torch.long)  # [batch * budget]

    def _update_target_tree_attn_bias(self, tree_attn_bias: torch.Tensor) -> None:
        """Push a new tree_attn_bias to all TreeAttentionMetadataBuilders.

        Called after each propose step so the target model's next
        verification pass uses the freshly-built tree topology.
        """
        if self._runner is None:
            return
        found_tree_attn = False
        for attn_groups in self._runner.attn_groups:
            for attn_group in attn_groups:
                builder = attn_group.get_metadata_builder()
                if isinstance(builder, TreeAttentionMetadataBuilder):
                    builder.tree_attn_bias = tree_attn_bias
                    # N = budget; reorder threshold = N so spec-decodes (q_len=N+1)
                    # land in region 2 (long_extend) after regular decodes (region 0),
                    # giving forward() a clean split point.
                    builder.reorder_batch_threshold = tree_attn_bias.shape[-2] - 1
                    # build() still needs to include spec-decodes in the decode
                    # bucket (not prefill), so keep decode threshold at N+1.
                    builder._tree_decode_threshold = tree_attn_bias.shape[-2]
                    found_tree_attn = True
        assert found_tree_attn, (
            "DDTreeProposer requires the target model to use the TREE_ATTN "
            "attention backend. Set attention_config.backend = 'TREE_ATTN' "
            "in your vllm config."
        )
