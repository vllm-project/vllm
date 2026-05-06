# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import (
    XgrammarBackend,
    XgrammarGrammar,
    _XgrammarDraftTree,
)

TOKENIZER = "gpt2"


def test_xgrammar_draft_tree_topology_chain():
    draft_tree = _XgrammarDraftTree(
        tree_choices=((0,), (0, 0), (0, 0, 0)),
    )

    retrieve_next_token, retrieve_next_sibling = draft_tree.topology(prefix_len=3)

    assert retrieve_next_token.tolist() == [1, 2, 3, -1]
    assert retrieve_next_sibling.tolist() == [-1, -1, -1, -1]


def test_xgrammar_draft_tree_topology_branching():
    draft_tree = _XgrammarDraftTree(
        tree_choices=((0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1)),
    )

    retrieve_next_token, retrieve_next_sibling = draft_tree.topology(prefix_len=6)

    assert retrieve_next_token.tolist() == [1, 3, 5, -1, -1, -1, -1]
    assert retrieve_next_sibling.tolist() == [-1, 2, -1, 4, -1, 6, -1]


def test_xgrammar_draft_tree_topology_truncated_prefix():
    draft_tree = _XgrammarDraftTree(
        tree_choices=((0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1)),
    )

    retrieve_next_token, retrieve_next_sibling = draft_tree.topology(prefix_len=4)

    assert retrieve_next_token.tolist() == [1, 3, -1, -1, -1]
    assert retrieve_next_sibling.tolist() == [-1, 2, -1, 4, -1]


def test_xgrammar_draft_tree_branch_token_indices():
    draft_tree = _XgrammarDraftTree(
        tree_choices=((0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1)),
    )

    assert draft_tree.branch_token_indices(prefix_len=6) == [
        (),
        (0,),
        (1,),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
    ]


def test_xgrammar_backend_builds_chain_when_speculative_tree_absent():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    backend = XgrammarBackend(
        vllm_config=SimpleNamespace(
            structured_outputs_config=StructuredOutputsConfig(backend="xgrammar"),
            speculative_config=SimpleNamespace(
                num_speculative_tokens=3,
                speculative_token_tree=None,
            ),
        ),
        tokenizer=tokenizer,
        vocab_size=tokenizer.vocab_size,
    )

    assert backend.draft_tree is not None
    assert backend.draft_tree.tree_choices == ((0,), (0, 0), (0, 0, 0))


def test_xgrammar_speculative_bitmask_chain_matches_linear_walk():
    tokenizer, backend, actual_grammar = _compile_regex_grammar(
        "[ab][cd][ef]",
        tree_choices=((0,), (0, 0), (0, 0, 0)),
    )
    _, _, expected_grammar = _compile_regex_grammar(
        "[ab][cd][ef]",
        tree_choices=((0,), (0, 0), (0, 0, 0)),
    )
    tokens = [
        _token_id(tokenizer, "a"),
        _token_id(tokenizer, "c"),
        _token_id(tokenizer, "e"),
    ]
    actual = backend.allocate_token_bitmask(len(tokens) + 1)
    expected = backend.allocate_token_bitmask(len(tokens) + 1)

    assert actual_grammar.fill_speculative_bitmask(
        actual, batch_index=0, tokens=tokens, apply_bitmask=True
    )
    _fill_linear_speculative_bitmask(expected_grammar, tokens, expected)

    assert torch.equal(actual, expected)


def test_xgrammar_speculative_bitmask_branching_matches_forked_walk():
    tokenizer, backend, actual_grammar = _compile_regex_grammar(
        "[ab][cd]",
        tree_choices=((0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1)),
    )
    _, _, expected_grammar = _compile_regex_grammar(
        "[ab][cd]",
        tree_choices=((0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1)),
    )
    tokens = [
        _token_id(tokenizer, "a"),
        _token_id(tokenizer, "b"),
        _token_id(tokenizer, "c"),
        _token_id(tokenizer, "d"),
        _token_id(tokenizer, "c"),
        _token_id(tokenizer, "d"),
    ]
    actual = backend.allocate_token_bitmask(len(tokens) + 1)
    expected = backend.allocate_token_bitmask(len(tokens) + 1)

    assert actual_grammar.fill_speculative_bitmask(
        actual, batch_index=0, tokens=tokens, apply_bitmask=True
    )
    _fill_branch_speculative_bitmask(expected_grammar, tokens, expected)

    assert torch.equal(actual, expected)


def test_xgrammar_speculative_bitmask_invalid_draft_does_not_advance():
    tokenizer, backend, grammar = _compile_regex_grammar(
        "a",
        tree_choices=((0,),),
    )
    valid_token = _token_id(tokenizer, "a")
    invalid_token = _token_id(tokenizer, "b")
    bitmask = backend.allocate_token_bitmask(2)
    accepted_before = grammar.validate_tokens([valid_token])

    assert grammar.fill_speculative_bitmask(
        bitmask,
        batch_index=0,
        tokens=[invalid_token],
        apply_bitmask=True,
    )

    assert accepted_before == [valid_token]
    assert grammar.validate_tokens([valid_token]) == accepted_before
    assert not _bitmask_allows(bitmask, row=0, token_id=invalid_token)


def test_xgrammar_speculative_bitmask_pads_unknown_suffix():
    tokenizer, backend, grammar = _compile_regex_grammar(
        "[ab][cd][ef]",
        tree_choices=((0,), (0, 0), (0, 0, 0)),
    )
    tokens = [_token_id(tokenizer, "a"), -1, -1]
    bitmask = backend.allocate_token_bitmask(len(tokens) + 1)

    assert grammar.fill_speculative_bitmask(
        bitmask, batch_index=0, tokens=tokens, apply_bitmask=True
    )

    assert not torch.equal(bitmask[0], torch.full_like(bitmask[0], -1))
    assert not torch.equal(bitmask[1], torch.full_like(bitmask[1], -1))
    assert torch.equal(bitmask[2], torch.full_like(bitmask[2], -1))
    assert torch.equal(bitmask[3], torch.full_like(bitmask[3], -1))


def _compile_regex_grammar(
    regex: str,
    tree_choices: tuple[tuple[int, ...], ...],
) -> tuple[AutoTokenizer, XgrammarBackend, XgrammarGrammar]:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    backend = XgrammarBackend(
        vllm_config=VllmConfig(
            structured_outputs_config=StructuredOutputsConfig(backend="xgrammar"),
        ),
        tokenizer=tokenizer,
        vocab_size=tokenizer.vocab_size,
    )
    grammar = backend.compile_grammar(StructuredOutputOptions.REGEX, regex)
    assert isinstance(grammar, XgrammarGrammar)
    grammar.draft_tree = _XgrammarDraftTree(tree_choices=tree_choices)
    return tokenizer, backend, grammar


def _token_id(tokenizer: AutoTokenizer, text: str) -> int:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    assert len(tokens) == 1
    return tokens[0]


def _fill_linear_speculative_bitmask(
    grammar: XgrammarGrammar,
    tokens: list[int],
    bitmask: torch.Tensor,
) -> None:
    state_advancements = 0
    for idx, token in enumerate([*tokens, -1]):
        grammar.fill_bitmask(bitmask, idx)
        if token == -1:
            continue
        assert grammar.accept_tokens("req", [token])
        state_advancements += 1
    grammar.rollback(state_advancements)


def _fill_branch_speculative_bitmask(
    grammar: XgrammarGrammar,
    tokens: list[int],
    bitmask: torch.Tensor,
) -> None:
    assert grammar.draft_tree is not None
    grammar.matcher.fill_next_token_bitmask(bitmask, 0)
    branch_token_indices = grammar.draft_tree.branch_token_indices(len(tokens))
    for node_idx in range(1, len(tokens) + 1):
        token_indices = branch_token_indices[node_idx]
        assert token_indices is not None
        matcher = grammar.matcher.fork()
        accepted = True
        for token_idx in token_indices:
            if not matcher.accept_token(tokens[token_idx]):
                accepted = False
                break
        if accepted:
            matcher.fill_next_token_bitmask(bitmask, node_idx)
        else:
            bitmask[node_idx].fill_(-1)


def _bitmask_allows(bitmask: torch.Tensor, row: int, token_id: int) -> bool:
    word = int(bitmask[row, token_id // 32].item())
    return bool(word & (1 << (token_id % 32)))
