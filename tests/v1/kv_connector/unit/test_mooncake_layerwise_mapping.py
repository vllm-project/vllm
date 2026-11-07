# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import (
    mooncake_connector as mapping,
)

pytestmark = pytest.mark.cpu_test


class BlockMappingTests(unittest.TestCase):
    def test_chunk_boundary_prefix_and_tail(self):
        plan = mapping._build_pending_block_slice
        for size in (16, 32, 64):
            with self.subTest(size=size):
                result = plan(
                    [10, 11, 12],
                    [21, 22],
                    token_end=2 * size + 1,
                    available=2 * size,
                    sent=None,
                    block_size=size,
                    final=False,
                )
                self.assertEqual(result.source, (11,))
                self.assertEqual(result.destination, (21,))
                self.assertEqual(result.end_tokens, 2 * size)
                tail = plan(
                    [10, 11, 12],
                    [21, 22],
                    token_end=2 * size + 1,
                    available=2 * size + 1,
                    sent=result.end_tokens,
                    block_size=size,
                    final=True,
                )
                self.assertEqual(tail.source, (12,))
                self.assertEqual(tail.destination, (22,))
                self.assertEqual(tail.end_tokens, 3 * size)

    def test_empty_destination_is_already_satisfied(self):
        self.assertIsNone(
            mapping._build_pending_block_slice(
                [11],
                [],
                token_end=15,
                available=15,
                sent=None,
                block_size=16,
                final=True,
            )
        )

    def test_window_does_not_compact_logical_null_positions(self):
        result = mapping._build_window_block_slice(
            [0, 0, 12, 13], [0, 22, 23], token_end=63, block_size=16, null_block=0
        )
        self.assertEqual(result.source, (12, 13))
        self.assertEqual(result.destination, (22, 23))
        with self.assertRaisesRegex(ValueError, "retired"):
            mapping._build_window_block_slice(
                [0, 0, 0, 13], [22, 23], token_end=63, block_size=16, null_block=0
            )

    def test_state_boundary_and_missing_slot(self):
        result = mapping._build_state_block_slice(
            [0, 12, 0, 13], [0, 23], token_end=543, null_block=0
        )
        self.assertEqual(result.source, (13,))
        self.assertEqual(result.destination, (23,))
        self.assertEqual(result.end_tokens, 543)
        with self.assertRaisesRegex(ValueError, "missing"):
            mapping._build_state_block_slice(
                [13], [22, 23], token_end=543, null_block=0
            )

    def test_ring_does_not_expand_with_prompt_length(self):
        for tokens in (1, 63, 64, 65, 4097):
            with self.subTest(tokens=tokens):
                result = mapping._build_state_block_slice(
                    [12], [23], token_end=tokens, null_block=0
                )
                self.assertEqual(result.source, (12,))
                self.assertEqual(result.destination, (23,))
                self.assertEqual(result.end_tokens, tokens)

    def test_reject_incomplete_source(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            mapping._build_pending_block_slice(
                [11],
                [21, 22],
                token_end=17,
                available=17,
                sent=None,
                block_size=16,
                final=True,
            )


if __name__ == "__main__":
    unittest.main()
