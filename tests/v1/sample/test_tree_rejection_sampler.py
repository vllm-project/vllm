# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Optional

import torch

from vllm.platforms import current_platform
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler
from vllm.v1.sample.tree_rejection_sampler import (PLACEHOLDER_TOKEN_ID,
                                                   TreeRejectionSampler)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.tree_spec_decode.tree_drafter_params import TreeDrafterParams

DEVICE = current_platform.device_type
VOCAB_SIZE = 100
Node = tuple[int, ...]

########################### Helper Functions ###########################


def create_tree_rejection_sampler(tree_structure: list[Node],
                                  batch_size: int) -> TreeRejectionSampler:
    tree_drafter_params = TreeDrafterParams.from_spec_token_tree(
        str(tree_structure))
    return TreeRejectionSampler(
        tree_drafter_params=tree_drafter_params,
        max_batch_size=batch_size,
        main_sampler=Sampler(),
        device=DEVICE,
    )


def get_token_id(tree: list[Node], node: Node) -> int:
    # Token id is just the position of this node in the tree.
    return tree.index(node)


def to_input_draft_token_ids(tree: list[Node], num_drafts: int,
                             draft_nodes: list[Node]) -> torch.Tensor:
    """
    Creates a tensor of draft token ids to input into the rejection sampler.
    Each given node is mapped to a unique token id. All other positions are
    given a random token id.
    """
    draft_token_ids = torch.randint(
        # Offset the random token ids by the size of the tree.
        low=len(tree),
        high=VOCAB_SIZE,
        size=(num_drafts, ),
        device=DEVICE)
    for draft_node in draft_nodes:
        # Get the draft node's position in the tree, excluding the root node.
        index = tree.index(draft_node) - 1
        # Assign unique token id to the node.
        token_id = get_token_id(tree, draft_node)
        draft_token_ids[index] = token_id
    return draft_token_ids


def to_output_token_ids(tree: list[Node], num_drafts: int,
                        accepted: list[Node], bonus: Node) -> torch.Tensor:
    """
    Creates a tensor where only the accepted and bonus nodes are mapped to
    their token ids.
    """
    output_token_ids = torch.empty(num_drafts + 1, device=DEVICE)
    output_token_ids.fill_(PLACEHOLDER_TOKEN_ID)
    for accepted_node in accepted:
        index = tree.index(accepted_node) - 1
        token_id = get_token_id(tree, accepted_node)
        output_token_ids[index] = token_id
    output_token_ids[-1] = get_token_id(tree, bonus)
    return output_token_ids


def create_logits_tensor(tree: list[Node], num_logits: int,
                         sample_map: dict[Node, Node]) -> torch.Tensor:
    """
    Helper function to create logits tensor that will produce the desired
    token ids on argmax
    """
    logits = torch.full((num_logits, VOCAB_SIZE), -100.0, device=DEVICE)
    for index in range(num_logits):
        node = tree[index]
        if node not in sample_map:
            continue
        sampled_node = sample_map[node]
        token_id = get_token_id(tree, sampled_node)
        logits[index, token_id] = 100.0
    return logits


def create_sampling_metadata(
    all_greedy: bool,
    temperature: Optional[torch.Tensor] = None,
    top_k: Optional[torch.Tensor] = None,
    top_p: Optional[torch.Tensor] = None,
    generators: Optional[dict[int, Any]] = None,
) -> SamplingMetadata:
    """
    Create a v1 sampling metadata object with all_greedy set to the given
    value. Either all greedy or all random sampling is used.
    """
    generators = generators or {}
    if all_greedy:
        temperature = None
    else:
        assert temperature is not None

    return SamplingMetadata(
        temperature=temperature,
        all_greedy=all_greedy,
        all_random=not all_greedy,
        top_p=top_p,
        top_k=top_k,
        generators=generators,
        max_num_logprobs=0,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.tensor([]),
        presence_penalties=torch.tensor([]),
        repetition_penalties=torch.tensor([]),
        output_token_ids=[],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def assert_rejection_sample(
    draft_tree: list[Node],
    spec_nodes: list[list[Node]],
    target_sample_maps: list[dict[Node, Node]],
    expected_accepted_nodes: list[list[Node]],
    expected_bonus_nodes: list[Node],
):
    num_drafts = len(draft_tree)
    # Create tree rejection sampler.
    tree_rejection_sampler = create_tree_rejection_sampler(
        draft_tree, len(spec_nodes))

    # Create the bonus level.
    last_level = len(draft_tree[-1])
    leaves = [node for node in draft_tree if len(node) == last_level]
    bonus_level = [leaf + (0, ) for leaf in leaves]
    # Create tree with root node and bonus level added.
    tree = [()] + draft_tree + bonus_level

    # Convert drafted tokens mapping to tensor representation.
    input_draft_token_ids = torch.stack(
        [to_input_draft_token_ids(tree, num_drafts, s) for s in spec_nodes])
    spec_decode_metadata = SpecDecodeMetadata.make_dummy(
        input_draft_token_ids.tolist(), device=DEVICE)

    # Generate logits that deterministically produce the given sampled
    # tokens.
    logits = torch.cat([
        create_logits_tensor(tree, num_drafts + 1, sample_map)
        for sample_map in target_sample_maps
    ])

    # Create greedy sampling metadata.
    metadata = create_sampling_metadata(all_greedy=True)

    # Rejection sample.
    output = tree_rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )

    # Compare with output with expected.
    expected_tokens = torch.stack([
        to_output_token_ids(tree, num_drafts, a, b)
        for a, b in zip(expected_accepted_nodes, expected_bonus_nodes)
    ])
    assert torch.equal(output.sampled_token_ids, expected_tokens)


########################### Tests ###########################


def test_single_node():
    """
    Test exact match for a single node.
    """
    draft_tree: list[Node] = [
        (0, ),
    ]
    drafted_tokens: list[list[Node]] = [
        [(0, )],
    ]
    target_sample_maps: list[dict[Node, Node]] = [{
        (): (0, ),
        (0, ): (0, 0),
    }]
    expected_accepted_tokens: list[list[Node]] = [
        [(0, )],
    ]
    expected_bonus_tokens: list[Node] = [
        (0, 0),
    ]
    assert_rejection_sample(draft_tree, drafted_tokens, target_sample_maps,
                            expected_accepted_tokens, expected_bonus_tokens)


def test_chain_full_acceptance():
    draft_tree: list[Node] = [
        (0, ),
        (0, 0),
        (0, 0, 0),
    ]
    drafted_tokens: list[list[Node]] = [
        [(0, ), (0, 0), (0, 0, 0)],
    ]
    target_sample_maps: list[dict[Node, Node]] = [{
        (): (0, ),
        (0, ): (0, 0),
        (0, 0): (0, 0, 0),
        (0, 0, 0): (0, 0, 0, 0)
    }]
    expected_accepted_tokens: list[list[Node]] = [
        [(0, ), (0, 0), (0, 0, 0)],
    ]
    expected_bonus_tokens: list[Node] = [
        (0, 0, 0, 0),
    ]
    assert_rejection_sample(draft_tree, drafted_tokens, target_sample_maps,
                            expected_accepted_tokens, expected_bonus_tokens)


def test_chain_partial_acceptance():
    draft_tree: list[Node] = [
        (0, ),
        (0, 0),
        (0, 0, 0),
    ]
    target_sample_maps: list[dict[Node, Node]] = [{
        (): (0, ),
        (0, ): (0, 0),
        (0, 0): (0, 0, 0),
    }]
    drafted_tokens: list[list[Node]] = [
        [(0, ), (0, 0), (0, 0)],  # Mismatch for final draft (expected (0,0,0))
    ]
    expected_accepted_tokens: list[list[Node]] = [
        [(0, ), (0, 0)],
    ]
    expected_bonus_tokens: list[Node] = [
        (0, 0, 0),
    ]
    assert_rejection_sample(draft_tree, drafted_tokens, target_sample_maps,
                            expected_accepted_tokens, expected_bonus_tokens)


def test_tree_full_acceptance():
    draft_tree: list[Node] = [(0, ), (1, ), (0, 0), (0, 1), (1, 0), (1, 1)]
    drafted_tokens: list[list[Node]] = [
        [(1, ), (1, 1)],
    ]
    target_sample_maps: list[dict[Node, Node]] = [{
        (): (1, ),
        (1, ): (1, 1),
        (1, 1): (1, 1, 0),
    }]
    expected_accepted_tokens: list[list[Node]] = [
        [(1, ), (1, 1)],
    ]
    expected_bonus_tokens: list[Node] = [
        (1, 1, 0),
    ]
    assert_rejection_sample(draft_tree, drafted_tokens, target_sample_maps,
                            expected_accepted_tokens, expected_bonus_tokens)


def test_tree_partial_acceptance():
    draft_tree: list[Node] = [(0, ), (1, ), (0, 0), (0, 1), (1, 0), (1, 1)]
    drafted_tokens: list[list[Node]] = [
        [(0, ), (0, 0)],  # Mismatch for final draft (expected (0,0))
    ]
    target_sample_maps: list[dict[Node, Node]] = [{
        (): (0, ),
        (0, ): (0, 1),
    }]
    expected_accepted_tokens: list[list[Node]] = [
        [(0, )],
    ]
    expected_bonus_tokens: list[Node] = [
        (0, 1),
    ]
    assert_rejection_sample(draft_tree, drafted_tokens, target_sample_maps,
                            expected_accepted_tokens, expected_bonus_tokens)


def test_tree_early_rejection():
    draft_tree: list[Node] = [(0, ), (1, ), (0, 0), (0, 1), (1, 0), (1, 1)]
    drafted_tokens: list[list[Node]] = [
        [(1, ), (0, 1)],  # Mismatch for the first draft (expected (0,))
    ]
    target_sample_maps: list[dict[Node, Node]] = [{
        (): (0, ),
        (0, ): (0, 0),
        (0, 0): (0, 0, 0),
    }]
    expected_accepted_tokens: list[list[Node]] = [
        [],
    ]
    expected_bonus_tokens: list[Node] = [
        (0, ),
    ]
    assert_rejection_sample(draft_tree, drafted_tokens, target_sample_maps,
                            expected_accepted_tokens, expected_bonus_tokens)


def test_tree_full_acceptance_multiple_sequences():
    draft_tree: list[Node] = [(0, ), (1, ), (0, 0), (0, 1), (1, 0), (1, 1)]
    drafted_tokens: list[list[Node]] = [
        [(0, ), (0, 1)],  # Sequence 1
        [(1, ), (1, 0)],  # Sequence 2
    ]
    target_sample_maps: list[dict[Node, Node]] = [{
        (): (0, ),
        (0, ): (0, 1),
        (0, 1): (0, 1, 0),
    }, {
        (): (1, ),
        (1, ): (1, 0),
        (1, 0): (1, 0, 0),
    }]
    expected_accepted_tokens: list[list[Node]] = [
        [(0, ), (0, 1)],
        [(1, ), (1, 0)],
    ]
    expected_bonus_tokens: list[Node] = [
        (0, 1, 0),
        (1, 0, 0),
    ]
    assert_rejection_sample(draft_tree, drafted_tokens, target_sample_maps,
                            expected_accepted_tokens, expected_bonus_tokens)


def test_tree_partial_acceptance_multiple_sequences():
    draft_tree: list[Node] = [(0, ), (1, ), (0, 0), (0, 1), (1, 0), (1, 1)]
    drafted_tokens: list[list[Node]] = [
        [(0, ), (0, 0)],  # Mismatch for the second draft (expected (0,1))
        [(0, ), (0, 1)],  # Mismatch for the first draft (expected (1,))
    ]
    target_sample_maps: list[dict[Node, Node]] = [{
        (): (0, ),
        (0, ): (0, 1),
    }, {
        (): (1, ),
        (1, ): (1, 0),
    }]
    expected_accepted_tokens: list[list[Node]] = [
        [(0, )],
        [],
    ]
    expected_bonus_tokens: list[Node] = [
        (0, 1),
        (1, ),
    ]
    assert_rejection_sample(draft_tree, drafted_tokens, target_sample_maps,
                            expected_accepted_tokens, expected_bonus_tokens)


def test_deep_tree_full_acceptance():
    draft_tree: list[Node] = [
        (0, ),
        (1, ),  # Level 1
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),  # Level 2
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1)  # Level 3
    ]
    drafted_tokens: list[list[Node]] = [
        [(1, ), (1, 1), (1, 1, 0)],
    ]
    target_sample_maps: list[dict[Node, Node]] = [{
        (): (1, ),
        (0, ): (0, 1),
        (1, ): (1, 1),
        (0, 0): (0, 0, 0),
        (1, 1): (1, 1, 0),
        (1, 1, 0): (1, 1, 0, 0),
    }]
    expected_accepted_tokens: list[list[Node]] = [
        [(1, ), (1, 1), (1, 1, 0)],
    ]
    expected_bonus_tokens: list[Node] = [
        (1, 1, 0, 0),
    ]
    assert_rejection_sample(draft_tree, drafted_tokens, target_sample_maps,
                            expected_accepted_tokens, expected_bonus_tokens)
