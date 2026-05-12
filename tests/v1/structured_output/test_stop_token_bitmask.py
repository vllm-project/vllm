# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.structured_output import StructuredOutputManager


class FakeGrammar:
    def __init__(self, terminated: bool = False):
        self.terminated = terminated

    def is_terminated(self):
        return self.terminated

    def fill_bitmask(self, bitmask, batch_index):
        bitmask[batch_index].fill_(-1)


def _token_allowed(row, token_id: int) -> bool:
    word = int(row[token_id // 32].item()) & 0xFFFFFFFF
    return bool(word & (1 << (token_id % 32)))


def test_fill_bitmask_masks_stop_tokens_before_grammar_terminates():
    manager = object.__new__(StructuredOutputManager)
    manager._grammar_bitmask = torch.full((1, 4), -1, dtype=torch.int32)
    manager._full_mask = torch.tensor(-1, dtype=torch.int32)

    stop_token_ids = {0, 31, 32, 106}
    manager._fill_bitmasks(((FakeGrammar(), 0, True, stop_token_ids, True),))

    row = manager._grammar_bitmask[0]
    for token_id in stop_token_ids:
        assert not _token_allowed(row, token_id)
    assert _token_allowed(row, 1)


def test_fill_bitmask_allows_stop_tokens_after_grammar_terminates():
    manager = object.__new__(StructuredOutputManager)
    manager._grammar_bitmask = torch.zeros((1, 4), dtype=torch.int32)
    manager._full_mask = torch.tensor(-1, dtype=torch.int32)

    manager._fill_bitmasks(((FakeGrammar(terminated=True), 0, True, {106}, True),))

    assert _token_allowed(manager._grammar_bitmask[0], 106)


def test_fill_bitmask_preserves_stop_tokens_for_backend_final_masks():
    manager = object.__new__(StructuredOutputManager)
    manager._grammar_bitmask = torch.full((1, 4), -1, dtype=torch.int32)
    manager._full_mask = torch.tensor(-1, dtype=torch.int32)

    manager._fill_bitmasks(((FakeGrammar(), 0, True, {106}, False),))

    assert _token_allowed(manager._grammar_bitmask[0], 106)
