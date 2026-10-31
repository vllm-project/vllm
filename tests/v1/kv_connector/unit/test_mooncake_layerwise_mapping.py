# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run with .venv/bin/python directly; deliberately imports no device runtime."""

import runpy
import unittest
from pathlib import Path

# Load the actual standard-library-only planner without importing vllm.__init__.
# This is not a substitute module or a mocked torch environment.
_mapping = runpy.run_path(
    str(
        Path(__file__).resolve().parents[4]
        / "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/layerwise_mapping.py"
    )
)


class BlockMappingTests(unittest.TestCase):
    def test_chunk_boundary_prefix_and_tail(self):
        map_pending_blocks = _mapping["map_pending_blocks"]
        for size in (16, 32, 64):
            with self.subTest(size=size):
                result = map_pending_blocks(
                    [10, 11, 12],
                    [21, 22],
                    token_end=2 * size + 1,
                    available=2 * size,
                    sent=None,
                    block_size=size,
                    final=False,
                )
                self.assertEqual(result, _mapping["BlockSlice"]((11,), (21,), 2 * size))
                tail = map_pending_blocks(
                    [10, 11, 12],
                    [21, 22],
                    token_end=2 * size + 1,
                    available=2 * size + 1,
                    sent=result.end_tokens,
                    block_size=size,
                    final=True,
                )
                self.assertEqual(tail, _mapping["BlockSlice"]((12,), (22,), 3 * size))

    def test_empty_destination_is_already_satisfied(self):
        self.assertIsNone(
            _mapping["map_pending_blocks"](
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
        result = _mapping["map_window_blocks"](
            [0, 0, 12, 13], [0, 22, 23], token_end=63, block_size=16, null_block=0
        )
        self.assertEqual(result, _mapping["BlockSlice"]((12, 13), (22, 23), 64))
        with self.assertRaisesRegex(ValueError, "retired"):
            _mapping["map_window_blocks"](
                [0, 0, 0, 13], [22, 23], token_end=63, block_size=16, null_block=0
            )

    def test_state_slots_and_ring_wraps(self):
        map_state_blocks = _mapping["map_state_blocks"]
        cases = [
            ([0, 12, 0, 13], [0, 23], 543, (13,), (23,)),
            ([12, 13], [22, 23], 543, (12, 13), (22, 23)),
            *[([12], [23], tokens, (12,), (23,)) for tokens in (1, 63, 64, 65, 4097)],
        ]
        for source, destination, tokens, expected_src, expected_dst in cases:
            with self.subTest(source=source, destination=destination, tokens=tokens):
                self.assertEqual(
                    map_state_blocks(
                        source, destination, token_end=tokens, null_block=0
                    ),
                    _mapping["BlockSlice"](expected_src, expected_dst, tokens),
                )
        with self.assertRaisesRegex(ValueError, "missing"):
            map_state_blocks([13], [22, 23], token_end=543, null_block=0)

    def test_reject_incomplete_source(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            _mapping["map_pending_blocks"](
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
