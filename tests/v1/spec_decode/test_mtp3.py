# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.spec_decode.metadata import MultiLayerEagleMetadata
from vllm.v1.spec_decode.multi_layer_eagle import MultiLayerEagleProposer

HIDDEN_SIZE = 3


def _make_multi_layer_eagle_metadata(
    *,
    initial_cache: list[dict],
    max_shift: int,
    device: torch.device,
) -> MultiLayerEagleMetadata:
    for row in initial_cache:
        assert "len" in row
        row_len = int(row["len"])
        assert 0 <= row_len <= max_shift

        # Test cases pad cache rows to `layer_num` (== max_shift) and specify the
        # number of valid entries via `len`.
        assert (
            len(row["token_ids"])
            == len(row["positions"])
            == len(row["slot_mapping"])
            == max_shift
        )
        assert all(v == 0 for v in row["token_ids"][row_len:])
        assert all(v == 0 for v in row["positions"][row_len:])
        assert all(v == 0 for v in row["slot_mapping"][row_len:])

    cached_len = torch.tensor(
        [min(int(row["len"]), max_shift) for row in initial_cache],
        dtype=torch.int64,
        device=device,
    )
    cached_token_ids = torch.tensor(
        [row["token_ids"] for row in initial_cache],
        dtype=torch.int32,
        device=device,
    )
    cached_positions = torch.tensor(
        [row["positions"] for row in initial_cache],
        dtype=torch.int64,
        device=device,
    )
    cached_slot_mappings = torch.tensor(
        [row["slot_mapping"] for row in initial_cache],
        dtype=torch.int64,
        device=device,
    )
    cached_hidden_states = torch.zeros(
        (len(initial_cache), max_shift, HIDDEN_SIZE),
        dtype=torch.float32,
        device=device,
    )
    return MultiLayerEagleMetadata(
        cached_len=cached_len,
        cached_token_ids=cached_token_ids,
        cached_hidden_states=cached_hidden_states,
        cached_slot_mappings=cached_slot_mappings,
        cached_positions=cached_positions,
    )


@pytest.fixture
def proposer_stub():
    if not torch.cuda.is_available():
        pytest.skip("MultiLayerEagleProposer.adjust_input is CUDA/Triton-only.")
    proposer = MultiLayerEagleProposer.__new__(MultiLayerEagleProposer)
    proposer.layer_num = 3
    return proposer


LAYER3_CASES = [
    {
        "name": "shift_0_at_sequence_end",
        "batch_size": 1,
        "initial_cache": [
            {
                "len": 0,
                "token_ids": [0, 0, 0],
                "positions": [0, 0, 0],
                "slot_mapping": [0, 0, 0],
            }
        ],
        "target_token_ids": [10, 11, 12, 13],
        "target_positions": [0, 1, 2, 3],
        "last_token_indices": [3],
        "common_attn_metadata": {
            "query_start_loc": [0, 4],
            "seq_lens": [4],
            "seq_lens_cpu": [4],
            "num_computed_tokens_cpu": [0],
            "slot_mapping": [100, 101, 102, 103],
            "max_seq_len": 4,
        },
        "expected": {
            "prev_token_ids": [10, 11, 12, 13],
            "prev_positions": [0, 1, 2, 3],
            "last_token_indices": [3],
            "seq_lens": [4],
            "slot_mapping": [100, 101, 102, 103],
            "cached": [
                {
                    "len": 3,
                    "token_ids": [11, 12, 13],
                    "positions": [1, 2, 3],
                    "slot_mapping": [101, 102, 103],
                }
            ],
        },
    },
    {
        "name": "batch2_short_seq_no_shift",
        "batch_size": 2,
        "initial_cache": [
            {
                "len": 0,
                "token_ids": [0, 0, 0],
                "positions": [0, 0, 0],
                "slot_mapping": [0, 0, 0],
            },
            {
                "len": 0,
                "token_ids": [0, 0, 0],
                "positions": [0, 0, 0],
                "slot_mapping": [0, 0, 0],
            },
        ],
        "target_token_ids": [10, 11, 20],
        "target_positions": [0, 1, 0],
        "last_token_indices": [1, 2],
        "common_attn_metadata": {
            "query_start_loc": [0, 2, 3],
            "seq_lens": [2, 1],
            "seq_lens_cpu": [2, 1],
            "num_computed_tokens_cpu": [0, 0],
            "slot_mapping": [100, 101, 200],
            "max_seq_len": 2,
        },
        "expected": {
            "prev_token_ids": [10, 11, 20],
            "prev_positions": [0, 1, 0],
            "last_token_indices": [1, 2],
            "seq_lens": [2, 1],
            "slot_mapping": [100, 101, 200],
            "cached": [
                {
                    "len": 2,
                    "token_ids": [10, 11, 0],
                    "positions": [0, 1, 0],
                    "slot_mapping": [100, 101, 0],
                },
                {
                    "len": 1,
                    "token_ids": [20, 0, 0],
                    "positions": [0, 0, 0],
                    "slot_mapping": [200, 0, 0],
                },
            ],
        },
    },
    {
        "name": "batch2_short_seq_shift_on_first",
        "batch_size": 2,
        "initial_cache": [
            {
                "len": 1,
                "token_ids": [99, 0, 0],
                "positions": [0, 0, 0],
                "slot_mapping": [999, 0, 0],
            },
            {
                "len": 0,
                "token_ids": [0, 0, 0],
                "positions": [0, 0, 0],
                "slot_mapping": [0, 0, 0],
            },
        ],
        "target_token_ids": [10, 11, 20],
        "target_positions": [1, 2, 0],
        "last_token_indices": [0, 2],
        "common_attn_metadata": {
            "query_start_loc": [0, 2, 3],
            "seq_lens": [2, 1],
            "seq_lens_cpu": [2, 1],
            "num_computed_tokens_cpu": [1, 0],
            "slot_mapping": [100, 101, 200],
            "max_seq_len": 2,
        },
        "expected": {
            "prev_token_ids": [99, 10, 20],
            "prev_positions": [0, 1, 0],
            "last_token_indices": [1, 2],
            "seq_lens": [1, 1],
            "slot_mapping": [999, 100, 200],
            "cached": [
                {
                    "len": 2,
                    "token_ids": [99, 10, 0],
                    "positions": [0, 1, 0],
                    "slot_mapping": [999, 100, 0],
                },
                {
                    "len": 1,
                    "token_ids": [20, 0, 0],
                    "positions": [0, 0, 0],
                    "slot_mapping": [200, 0, 0],
                },
            ],
        },
    },
    {
        "name": "short_seq_len_2_shift_0_cache_len_1",
        "batch_size": 1,
        "initial_cache": [
            {
                "len": 0,
                "token_ids": [0, 0, 0],
                "positions": [0, 0, 0],
                "slot_mapping": [0, 0, 0],
            }
        ],
        "target_token_ids": [7, 8],
        "target_positions": [0, 1],
        "last_token_indices": [0],
        "common_attn_metadata": {
            "query_start_loc": [0, 2],
            "seq_lens": [2],
            "seq_lens_cpu": [2],
            "num_computed_tokens_cpu": [0],
            "slot_mapping": [1000, 1001],
            "max_seq_len": 2,
        },
        "expected": {
            "prev_token_ids": [7, 8],
            "prev_positions": [0, 1],
            "last_token_indices": [0],
            "seq_lens": [2],
            "slot_mapping": [1000, 1001],
            "cached": [
                {
                    "len": 1,
                    "token_ids": [7, 0, 0],
                    "positions": [0, 0, 0],
                    "slot_mapping": [1000, 0, 0],
                }
            ],
        },
    },
    {
        "name": "short_seq_len_2_shift_1_cache_len_2",
        "batch_size": 1,
        "initial_cache": [
            {
                "len": 1,
                "token_ids": [6, 0, 0],
                "positions": [0, 0, 0],
                "slot_mapping": [999, 0, 0],
            }
        ],
        "target_token_ids": [7, 8],
        "target_positions": [1, 2],
        "last_token_indices": [0],
        "common_attn_metadata": {
            "query_start_loc": [0, 2],
            "seq_lens": [2],
            "seq_lens_cpu": [2],
            "num_computed_tokens_cpu": [1],
            "slot_mapping": [1000, 1001],
            "max_seq_len": 2,
        },
        "expected": {
            "prev_token_ids": [6, 7],
            "prev_positions": [0, 1],
            "last_token_indices": [1],
            "seq_lens": [1],
            "slot_mapping": [999, 1000],
            "cached": [
                {
                    "len": 2,
                    "token_ids": [6, 7, 0],
                    "positions": [0, 1, 0],
                    "slot_mapping": [999, 1000, 0],
                }
            ],
        },
    },
    {
        "name": "shift_bounded_by_start_pos_zero",
        "batch_size": 1,
        "initial_cache": [
            {
                "len": 0,
                "token_ids": [0, 0, 0],
                "positions": [0, 0, 0],
                "slot_mapping": [0, 0, 0],
            }
        ],
        "target_token_ids": [10, 11, 12, 13],
        "target_positions": [0, 2, 3, 4],
        "last_token_indices": [1],
        "common_attn_metadata": {
            "query_start_loc": [0, 4],
            "seq_lens": [4],
            "seq_lens_cpu": [4],
            "num_computed_tokens_cpu": [0],
            "slot_mapping": [100, 101, 102, 103],
            "max_seq_len": 4,
        },
        "expected": {
            "prev_token_ids": [10, 11, 12, 13],
            "prev_positions": [0, 2, 3, 4],
            "last_token_indices": [1],
            "seq_lens": [4],
            "slot_mapping": [100, 101, 102, 103],
            "cached": [
                {
                    "len": 2,
                    "token_ids": [10, 11, 0],
                    "positions": [0, 2, 0],
                    "slot_mapping": [100, 101, 0],
                }
            ],
        },
    },
    {
        "name": "shift_bounded_by_start_pos",
        "batch_size": 1,
        "initial_cache": [
            {
                "len": 0,
                "token_ids": [0, 0, 0],
                "positions": [0, 0, 0],
                "slot_mapping": [0, 0, 0],
            }
        ],
        "target_token_ids": [10, 11, 12, 13, 14],
        "target_positions": [0, 1, 2, 3, 4],
        "last_token_indices": [1],
        "common_attn_metadata": {
            "query_start_loc": [0, 5],
            "seq_lens": [5],
            "seq_lens_cpu": [5],
            "num_computed_tokens_cpu": [1],
            "slot_mapping": [100, 101, 102, 103, 104],
            "max_seq_len": 5,
        },
        "expected": {
            "prev_token_ids": [10, 11, 12, 13, 14],
            "prev_positions": [0, 1, 2, 3, 4],
            "last_token_indices": [1],
            "seq_lens": [5],
            "slot_mapping": [100, 101, 102, 103, 104],
            "cached": [
                {
                    "len": 2,
                    "token_ids": [10, 11, 0],
                    "positions": [0, 1, 0],
                    "slot_mapping": [100, 101, 0],
                }
            ],
        },
    },
    {
        "name": "shift_2_bounded_by_remaining",
        "batch_size": 1,
        "initial_cache": [
            {
                "len": 0,
                "token_ids": [0, 0, 0],
                "positions": [0, 0, 0],
                "slot_mapping": [0, 0, 0],
            }
        ],
        "target_token_ids": [10, 11, 12, 13, 14],
        "target_positions": [0, 1, 2, 3, 4],
        "last_token_indices": [2],
        "common_attn_metadata": {
            "query_start_loc": [0, 5],
            "seq_lens": [5],
            "seq_lens_cpu": [5],
            "num_computed_tokens_cpu": [2],
            "slot_mapping": [100, 101, 102, 103, 104],
            "max_seq_len": 5,
        },
        "expected": {
            "prev_token_ids": [10, 11, 12, 13, 14],
            "prev_positions": [0, 1, 2, 3, 4],
            "last_token_indices": [2],
            "seq_lens": [5],
            "slot_mapping": [100, 101, 102, 103, 104],
            "cached": [
                {
                    "len": 3,
                    "token_ids": [10, 11, 12],
                    "positions": [0, 1, 2],
                    "slot_mapping": [100, 101, 102],
                }
            ],
        },
    },
    {
        "name": "shift_3_full_cache_window",
        "batch_size": 1,
        "initial_cache": [
            {
                "len": 0,
                "token_ids": [0, 0, 0],
                "positions": [0, 0, 0],
                "slot_mapping": [0, 0, 0],
            }
        ],
        "target_token_ids": [20, 21, 22, 23, 24],
        "target_positions": [0, 3, 4, 5, 6],
        "last_token_indices": [1],
        "common_attn_metadata": {
            "query_start_loc": [0, 5],
            "seq_lens": [5],
            "seq_lens_cpu": [5],
            "num_computed_tokens_cpu": [3],
            "slot_mapping": [100, 101, 102, 103, 104],
            "max_seq_len": 5,
        },
        "expected": {
            "prev_token_ids": [20, 21, 22, 23, 24],
            "prev_positions": [0, 3, 4, 5, 6],
            "last_token_indices": [1],
            "seq_lens": [5],
            "slot_mapping": [100, 101, 102, 103, 104],
            "cached": [
                {
                    "len": 2,
                    "token_ids": [20, 21, 0],
                    "positions": [0, 3, 0],
                    "slot_mapping": [100, 101, 0],
                }
            ],
        },
    },
    {
        "name": "batch2_shift_1_and_1",
        "batch_size": 2,
        "initial_cache": [
            {
                "len": 0,
                "token_ids": [0, 0, 0],
                "positions": [0, 0, 0],
                "slot_mapping": [0, 0, 0],
            },
            {
                "len": 0,
                "token_ids": [0, 0, 0],
                "positions": [0, 0, 0],
                "slot_mapping": [0, 0, 0],
            },
        ],
        "target_token_ids": [10, 11, 12, 13, 20, 21, 22],
        "target_positions": [0, 1, 2, 3, 0, 1, 2],
        "last_token_indices": [1, 5],
        "common_attn_metadata": {
            "query_start_loc": [0, 4, 7],
            "seq_lens": [4, 3],
            "seq_lens_cpu": [4, 3],
            "num_computed_tokens_cpu": [1, 1],
            "slot_mapping": [100, 101, 102, 103, 200, 201, 202],
            "max_seq_len": 4,
        },
        "expected": {
            "prev_token_ids": [10, 11, 12, 13, 20, 21, 22],
            "prev_positions": [0, 1, 2, 3, 0, 1, 2],
            "last_token_indices": [1, 5],
            "seq_lens": [4, 3],
            "slot_mapping": [100, 101, 102, 103, 200, 201, 202],
            "cached": [
                {
                    "len": 2,
                    "token_ids": [10, 11, 0],
                    "positions": [0, 1, 0],
                    "slot_mapping": [100, 101, 0],
                },
                {
                    "len": 2,
                    "token_ids": [20, 21, 0],
                    "positions": [0, 1, 0],
                    "slot_mapping": [200, 201, 0],
                },
            ],
        },
    },
    {
        "name": "batch4_mixed_shifts",
        "batch_size": 4,
        "initial_cache": [
            {
                "len": 0,
                "token_ids": [0, 0, 0],
                "positions": [0, 0, 0],
                "slot_mapping": [0, 0, 0],
            },
            {
                "len": 1,
                "token_ids": [19, 0, 0],
                "positions": [0, 0, 0],
                "slot_mapping": [119, 0, 0],
            },
            {
                "len": 0,
                "token_ids": [0, 0, 0],
                "positions": [0, 0, 0],
                "slot_mapping": [0, 0, 0],
            },
            {
                "len": 0,
                "token_ids": [0, 0, 0],
                "positions": [0, 0, 0],
                "slot_mapping": [0, 0, 0],
            },
        ],
        "target_token_ids": [10, 11, 20, 21, 22, 30, 31, 32, 33, 40, 41, 42],
        "target_positions": [0, 1, 1, 2, 3, 0, 2, 3, 4, 0, 1, 2],
        "last_token_indices": [1, 2, 6, 10],
        "common_attn_metadata": {
            "query_start_loc": [0, 2, 5, 9, 12],
            "seq_lens": [2, 3, 4, 3],
            "seq_lens_cpu": [2, 3, 4, 3],
            "num_computed_tokens_cpu": [0, 1, 2, 1],
            "slot_mapping": [
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
                110,
                111,
            ],
            "max_seq_len": 4,
        },
        "expected": {
            "prev_token_ids": [10, 11, 19, 20, 21, 30, 31, 32, 33, 40, 41, 42],
            "prev_positions": [0, 1, 0, 1, 2, 0, 2, 3, 4, 0, 1, 2],
            "last_token_indices": [1, 3, 6, 10],
            "seq_lens": [2, 2, 4, 3],
            "slot_mapping": [
                100,
                101,
                119,
                102,
                103,
                105,
                106,
                107,
                108,
                109,
                110,
                111,
            ],
            "cached": [
                {
                    "len": 2,
                    "token_ids": [10, 11, 0],
                    "positions": [0, 1, 0],
                    "slot_mapping": [100, 101, 0],
                },
                {
                    "len": 2,
                    "token_ids": [19, 20, 0],
                    "positions": [0, 1, 0],
                    "slot_mapping": [119, 102, 0],
                },
                {
                    "len": 2,
                    "token_ids": [30, 31, 0],
                    "positions": [0, 2, 0],
                    "slot_mapping": [105, 106, 0],
                },
                {
                    "len": 2,
                    "token_ids": [40, 41, 0],
                    "positions": [0, 1, 0],
                    "slot_mapping": [109, 110, 0],
                },
            ],
        },
    },
    {
        "name": "batch2_shift_0_and_2",
        "batch_size": 2,
        "initial_cache": [
            {
                "len": 0,
                "token_ids": [0, 0, 0],
                "positions": [0, 0, 0],
                "slot_mapping": [0, 0, 0],
            },
            {
                "len": 0,
                "token_ids": [0, 0, 0],
                "positions": [0, 0, 0],
                "slot_mapping": [0, 0, 0],
            },
        ],
        "target_token_ids": [30, 31, 32, 40, 41, 42, 43],
        "target_positions": [0, 1, 2, 0, 3, 4, 5],
        "last_token_indices": [2, 4],
        "common_attn_metadata": {
            "query_start_loc": [0, 3, 7],
            "seq_lens": [3, 4],
            "seq_lens_cpu": [3, 4],
            "num_computed_tokens_cpu": [0, 2],
            "slot_mapping": [100, 101, 102, 200, 201, 202, 203],
            "max_seq_len": 4,
        },
        "expected": {
            "prev_token_ids": [30, 31, 32, 40, 41, 42, 43],
            "prev_positions": [0, 1, 2, 0, 3, 4, 5],
            "last_token_indices": [2, 4],
            "seq_lens": [3, 4],
            "slot_mapping": [100, 101, 102, 200, 201, 202, 203],
            "cached": [
                {
                    "len": 3,
                    "token_ids": [30, 31, 32],
                    "positions": [0, 1, 2],
                    "slot_mapping": [100, 101, 102],
                },
                {
                    "len": 2,
                    "token_ids": [40, 41, 0],
                    "positions": [0, 3, 0],
                    "slot_mapping": [200, 201, 0],
                },
            ],
        },
    },
    {
        "name": "continue_req_shift_1_cache_tail_3",
        "batch_size": 1,
        "initial_cache": [
            {
                "len": 3,
                "token_ids": [70, 71, 72],
                "positions": [7, 8, 9],
                "slot_mapping": [170, 171, 172],
            }
        ],
        "target_token_ids": [100, 101, 102, 103, 104],
        "target_positions": [10, 11, 12, 13, 14],
        "last_token_indices": [3],
        "common_attn_metadata": {
            "query_start_loc": [0, 5],
            "seq_lens": [5],
            "seq_lens_cpu": [5],
            "num_computed_tokens_cpu": [0],
            "slot_mapping": [200, 201, 202, 203, 204],
            "max_seq_len": 5,
        },
        "expected": {
            "prev_token_ids": [72, 100, 101, 102, 103],
            "prev_positions": [9, 10, 11, 12, 13],
            "last_token_indices": [4],
            "seq_lens": [4],
            "slot_mapping": [172, 200, 201, 202, 203],
            "cached": [
                {
                    "len": 3,
                    "token_ids": [101, 102, 103],
                    "positions": [11, 12, 13],
                    "slot_mapping": [201, 202, 203],
                }
            ],
        },
    },
    {
        "name": "continue_req_shift_3_cache_tail_3",
        "batch_size": 1,
        "initial_cache": [
            {
                "len": 3,
                "token_ids": [270, 271, 272],
                "positions": [27, 28, 29],
                "slot_mapping": [370, 371, 372],
            }
        ],
        "target_token_ids": [300, 301, 302, 303, 304, 305, 306],
        "target_positions": [30, 31, 32, 33, 34, 35, 36],
        "last_token_indices": [3],
        "common_attn_metadata": {
            "query_start_loc": [0, 7],
            "seq_lens": [7],
            "seq_lens_cpu": [7],
            "num_computed_tokens_cpu": [0],
            "slot_mapping": [400, 401, 402, 403, 404, 405, 406],
            "max_seq_len": 7,
        },
        "expected": {
            "prev_token_ids": [270, 271, 272, 300, 301, 302, 303],
            "prev_positions": [27, 28, 29, 30, 31, 32, 33],
            "last_token_indices": [6],
            "seq_lens": [4],
            "slot_mapping": [370, 371, 372, 400, 401, 402, 403],
            "cached": [
                {
                    "len": 3,
                    "token_ids": [301, 302, 303],
                    "positions": [31, 32, 33],
                    "slot_mapping": [401, 402, 403],
                }
            ],
        },
    },
    {
        "name": "batch3_mixed_shifts_0_1_2_all_full_cache",
        "batch_size": 3,
        "initial_cache": [
            {
                "len": 0,
                "token_ids": [0, 0, 0],
                "positions": [0, 0, 0],
                "slot_mapping": [0, 0, 0],
            },
            {
                "len": 3,
                "token_ids": [70, 71, 72],
                "positions": [7, 8, 9],
                "slot_mapping": [170, 171, 172],
            },
            {
                "len": 3,
                "token_ids": [270, 271, 272],
                "positions": [17, 18, 19],
                "slot_mapping": [370, 371, 372],
            },
        ],
        "target_token_ids": [
            10,
            11,
            12,
            13,
            100,
            101,
            102,
            103,
            104,
            200,
            201,
            202,
            203,
            204,
            205,
        ],
        "target_positions": [
            0,
            1,
            2,
            3,
            10,
            11,
            12,
            13,
            14,
            20,
            21,
            22,
            23,
            24,
            25,
        ],
        "last_token_indices": [3, 7, 12],
        "common_attn_metadata": {
            "query_start_loc": [0, 4, 9, 15],
            "seq_lens": [4, 5, 6],
            "seq_lens_cpu": [4, 5, 6],
            "num_computed_tokens_cpu": [0, 0, 0],
            "slot_mapping": [
                100,
                101,
                102,
                103,
                200,
                201,
                202,
                203,
                204,
                300,
                301,
                302,
                303,
                304,
                305,
            ],
            "max_seq_len": 6,
        },
        "expected": {
            "prev_token_ids": [
                10,
                11,
                12,
                13,
                72,
                100,
                101,
                102,
                103,
                271,
                272,
                200,
                201,
                202,
                203,
            ],
            "prev_positions": [
                0,
                1,
                2,
                3,
                9,
                10,
                11,
                12,
                13,
                18,
                19,
                20,
                21,
                22,
                23,
            ],
            "last_token_indices": [3, 8, 14],
            "seq_lens": [4, 4, 4],
            "slot_mapping": [
                100,
                101,
                102,
                103,
                172,
                200,
                201,
                202,
                203,
                371,
                372,
                300,
                301,
                302,
                303,
            ],
            "cached": [
                {
                    "len": 3,
                    "token_ids": [11, 12, 13],
                    "positions": [1, 2, 3],
                    "slot_mapping": [101, 102, 103],
                },
                {
                    "len": 3,
                    "token_ids": [101, 102, 103],
                    "positions": [11, 12, 13],
                    "slot_mapping": [201, 202, 203],
                },
                {
                    "len": 3,
                    "token_ids": [201, 202, 203],
                    "positions": [21, 22, 23],
                    "slot_mapping": [301, 302, 303],
                },
            ],
        },
    },
]


def _run_adjust_input_case(proposer_stub, case, layer_num):
    proposer = proposer_stub
    proposer.layer_num = layer_num
    max_shift = proposer.layer_num
    device = torch.device("cuda")

    initial_cache = case["initial_cache"]
    batch_size = case["batch_size"]
    assert len(initial_cache) == batch_size

    meta = case["common_attn_metadata"]
    query_start_loc_cpu = torch.tensor(
        meta["query_start_loc"], dtype=torch.int32, device="cpu"
    )
    common_attn_metadata = SimpleNamespace(
        query_start_loc=query_start_loc_cpu.to(device=device),
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=torch.tensor(meta["seq_lens"], dtype=torch.int32, device=device),
        seq_lens_cpu=torch.tensor(
            meta["seq_lens_cpu"], dtype=torch.int32, device="cpu"
        ),
        num_computed_tokens_cpu=torch.tensor(
            meta["num_computed_tokens_cpu"], dtype=torch.int32, device="cpu"
        ),
        slot_mapping=torch.tensor(
            meta["slot_mapping"], dtype=torch.int64, device=device
        ),
        max_seq_len=meta["max_seq_len"],
    )

    target_token_ids = torch.tensor(
        case["target_token_ids"], dtype=torch.int32, device=device
    )
    target_positions = torch.tensor(
        case["target_positions"], dtype=torch.int64, device=device
    )
    target_hidden_states = torch.arange(
        0, target_token_ids.numel() * HIDDEN_SIZE, dtype=torch.float32, device=device
    ).reshape(-1, HIDDEN_SIZE)
    last_token_indices = torch.tensor(
        case["last_token_indices"], dtype=torch.int32, device=device
    )

    multi_layer_eagle_metadata = _make_multi_layer_eagle_metadata(
        initial_cache=initial_cache,
        max_shift=max_shift,
        device=device,
    )

    prev_token_ids, prev_positions, _, _ = proposer.adjust_input(
        batch_size=batch_size,
        target_token_ids=target_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        last_token_indices=last_token_indices,
        common_attn_metadata=common_attn_metadata,
        multi_layer_eagle_metadata=multi_layer_eagle_metadata,
    )

    expected = case["expected"]
    assert len(expected["cached"]) == batch_size
    assert prev_token_ids.cpu().tolist() == expected["prev_token_ids"]
    assert prev_positions.cpu().tolist() == expected["prev_positions"]
    assert last_token_indices.cpu().tolist() == expected["last_token_indices"]
    assert common_attn_metadata.seq_lens.cpu().tolist() == expected["seq_lens"]
    assert common_attn_metadata.slot_mapping.cpu().tolist() == expected["slot_mapping"]

    for row, cached_expect in enumerate(expected["cached"]):
        assert cached_expect["len"] <= max_shift
        assert (
            len(cached_expect["token_ids"])
            == len(cached_expect["positions"])
            == len(cached_expect["slot_mapping"])
            == max_shift
        )

        cache_len = int(cached_expect["len"])
        assert int(multi_layer_eagle_metadata.cached_len[row].item()) == cache_len
        assert all(v == 0 for v in cached_expect["token_ids"][cache_len:])
        assert all(v == 0 for v in cached_expect["positions"][cache_len:])
        assert all(v == 0 for v in cached_expect["slot_mapping"][cache_len:])
        assert (
            multi_layer_eagle_metadata.cached_token_ids[row].cpu().tolist()
            == cached_expect["token_ids"]
        )
        assert (
            multi_layer_eagle_metadata.cached_positions[row].cpu().tolist()
            == cached_expect["positions"]
        )
        assert (
            multi_layer_eagle_metadata.cached_slot_mappings[row].cpu().tolist()
            == cached_expect["slot_mapping"]
        )


@pytest.mark.parametrize(
    "case", LAYER3_CASES, ids=[case["name"] for case in LAYER3_CASES]
)
def test_adjust_input_layer3_cases(proposer_stub, case):
    _run_adjust_input_case(proposer_stub, case, layer_num=3)
