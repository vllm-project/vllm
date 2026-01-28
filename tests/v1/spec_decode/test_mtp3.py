# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backend import AttentionMetadataBuilder
from vllm.v1.spec_decode.multi_layer_eagle import (
    DraftInputStates,
    MultiLayerEagleProposer,
)


class DummyBuilder(AttentionMetadataBuilder):
    def __init__(self, return_value: str):
        # attention metadata builders normally take multiple runtime args;
        # the test double shortcuts that setup.
        self.return_value = return_value
        self.calls: list[dict] = []
        self.kv_cache_spec = None
        self.layer_names: list[str] = []
        self.vllm_config = None
        self.device = torch.device("cpu")

    def build(
        self, common_prefix_len: int, common_attn_metadata, fast_build: bool = False
    ):
        self.calls.append(
            {
                "common_prefix_len": common_prefix_len,
                "common_attn_metadata": common_attn_metadata,
                "fast_build": fast_build,
            }
        )
        return self.return_value


@pytest.fixture
def proposer_stub():
    proposer = MultiLayerEagleProposer.__new__(MultiLayerEagleProposer)
    proposer.layer_num = 3
    proposer.running_req_ids = ["req-0"]
    proposer.attn_layer_names = ["attn_layer"]
    proposer.indexer_layer_names = ["indexer_layer"]
    proposer.attn_metadata_builder = DummyBuilder("attn_meta")
    proposer.draft_indexer_metadata_builder = DummyBuilder("indexer_meta")
    proposer.draft_input_states_pool = {
        "req-0": DraftInputStates(
            len=3,
            token_ids=torch.tensor([800, 801, 802], dtype=torch.int32),
            hidden_states=torch.tensor(
                [[30.0, 31.0, 32.0], [33.0, 34.0, 35.0], [36.0, 37.0, 38.0]]
            ),
            positions=torch.tensor([0, 1, 2], dtype=torch.int64),
            slot_mapping=torch.tensor([900, 901, 902], dtype=torch.int32),
        ),
        "req-1": DraftInputStates(
            len=2,
            token_ids=torch.tensor([910, 911], dtype=torch.int32),
            hidden_states=torch.tensor([[40.0, 41.0, 42.0], [43.0, 44.0, 45.0]]),
            positions=torch.tensor([0, 1], dtype=torch.int64),
            slot_mapping=torch.tensor([990, 991], dtype=torch.int32),
        ),
        "req-2": DraftInputStates(
            len=3,
            token_ids=torch.tensor([820, 821, 822], dtype=torch.int32),
            hidden_states=torch.tensor(
                [[46.0, 47.0, 48.0], [49.0, 50.0, 51.0], [52.0, 53.0, 54.0]]
            ),
            positions=torch.tensor([0, 1, 2], dtype=torch.int64),
            slot_mapping=torch.tensor([920, 921, 922], dtype=torch.int32),
        ),
        "req-3": DraftInputStates(
            len=3,
            token_ids=torch.tensor([830, 831, 832], dtype=torch.int32),
            hidden_states=torch.tensor(
                [[55.0, 56.0, 57.0], [58.0, 59.0, 60.0], [61.0, 62.0, 63.0]]
            ),
            positions=torch.tensor([0, 1, 2], dtype=torch.int64),
            slot_mapping=torch.tensor([930, 931, 932], dtype=torch.int32),
        ),
    }
    return proposer


LAYER3_CASES = [
    {
        "name": "layer3_shift0_sequence_end",
        "batch_size": 1,
        "running_req_ids": ["req-0"],
        "target_token_ids": [10, 11, 12, 13],
        "target_positions": [0, 1, 2, 3],
        "last_token_indices": [3],
        "common_attn_metadata": {
            "query_start_loc": [0, 4],
            "query_start_loc_cpu": [0, 4],
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
            "seq_lens_cpu": [4],
            "num_computed_tokens_cpu": [0],
            "slot_mapping": [100, 101, 102, 103],
            "max_seq_len": 4,
            "cached_by_req": {
                "req-0": {
                    "len": 3,
                    "token_ids": [11, 12, 13],
                    "positions": [1, 2, 3],
                },
            },
        },
    },
    {
        "name": "layer3_batch2_short_seq_no_shift",
        "batch_size": 2,
        "running_req_ids": ["req-0", "req-1"],
        "target_token_ids": [10, 11, 20],
        "target_positions": [0, 1, 0],
        "last_token_indices": [1, 2],
        "common_attn_metadata": {
            "query_start_loc": [0, 2, 3],
            "query_start_loc_cpu": [0, 2, 3],
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
            "seq_lens_cpu": [2, 1],
            "num_computed_tokens_cpu": [0, 0],
            "slot_mapping": [100, 101, 200],
            "max_seq_len": 2,
            "cached_by_req": {
                "req-0": {
                    "len": 2,
                    "token_ids": [10, 11],
                    "positions": [0, 1],
                },
                "req-1": {
                    "len": 1,
                    "token_ids": [20],
                    "positions": [0],
                },
            },
        },
    },
    {
        "name": "layer3_batch2_short_seq_shift_first",
        "batch_size": 2,
        "running_req_ids": ["req-0", "req-1"],
        "target_token_ids": [10, 11, 20],
        "target_positions": [1, 2, 0],
        "last_token_indices": [0, 2],
        "common_attn_metadata": {
            "query_start_loc": [0, 2, 3],
            "query_start_loc_cpu": [0, 2, 3],
            "seq_lens": [2, 1],
            "seq_lens_cpu": [2, 1],
            "num_computed_tokens_cpu": [1, 0],
            "slot_mapping": [100, 101, 200],
            "max_seq_len": 2,
        },
        "expected": {
            "prev_token_ids": [802, 10, 20],
            "prev_positions": [2, 1, 0],
            "last_token_indices": [1, 2],
            "seq_lens": [1, 1],
            "seq_lens_cpu": [1, 1],
            "num_computed_tokens_cpu": [0, 0],
            "slot_mapping": [902, 100, 200],
            "max_seq_len": 1,
            "cached_by_req": {
                "req-0": {
                    "len": 2,
                    "token_ids": [802, 10],
                    "positions": [2, 1],
                },
                "req-1": {
                    "len": 1,
                    "token_ids": [20],
                    "positions": [0],
                },
            },
        },
    },
    {
        "name": "layer3_short_seq_len2_shift0_cache1",
        "batch_size": 1,
        "running_req_ids": ["req-0"],
        "target_token_ids": [7, 8],
        "target_positions": [0, 1],
        "last_token_indices": [0],
        "common_attn_metadata": {
            "query_start_loc": [0, 2],
            "query_start_loc_cpu": [0, 2],
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
            "seq_lens_cpu": [2],
            "num_computed_tokens_cpu": [0],
            "slot_mapping": [1000, 1001],
            "max_seq_len": 2,
            "cached_by_req": {
                "req-0": {
                    "len": 1,
                    "token_ids": [7],
                    "positions": [0],
                },
            },
        },
    },
    {
        "name": "layer3_short_seq_len2_shift1_cache2",
        "batch_size": 1,
        "running_req_ids": ["req-0"],
        "target_token_ids": [7, 8],
        "target_positions": [1, 2],
        "last_token_indices": [0],
        "common_attn_metadata": {
            "query_start_loc": [0, 2],
            "query_start_loc_cpu": [0, 2],
            "seq_lens": [2],
            "seq_lens_cpu": [2],
            "num_computed_tokens_cpu": [1],
            "slot_mapping": [1000, 1001],
            "max_seq_len": 2,
        },
        "expected": {
            "prev_token_ids": [802, 7],
            "prev_positions": [2, 1],
            "last_token_indices": [1],
            "seq_lens": [1],
            "seq_lens_cpu": [1],
            "num_computed_tokens_cpu": [0],
            "slot_mapping": [902, 1000],
            "max_seq_len": 1,
            "cached_by_req": {
                "req-0": {
                    "len": 2,
                    "token_ids": [802, 7],
                    "positions": [2, 1],
                },
            },
        },
    },
    {
        "name": "layer3_shift_bounded_start_pos0",
        "batch_size": 1,
        "running_req_ids": ["req-0"],
        "target_token_ids": [10, 11, 12, 13],
        "target_positions": [0, 1, 2, 3],
        "last_token_indices": [1],
        "common_attn_metadata": {
            "query_start_loc": [0, 4],
            "query_start_loc_cpu": [0, 4],
            "seq_lens": [4],
            "seq_lens_cpu": [4],
            "num_computed_tokens_cpu": [0],
            "slot_mapping": [100, 101, 102, 103],
            "max_seq_len": 4,
        },
        "expected": {
            "prev_token_ids": [10, 11, 12, 13],
            "prev_positions": [0, 1, 2, 3],
            "last_token_indices": [1],
            "seq_lens": [4],
            "seq_lens_cpu": [4],
            "num_computed_tokens_cpu": [0],
            "slot_mapping": [100, 101, 102, 103],
            "max_seq_len": 4,
            "cached_by_req": {
                "req-0": {
                    "len": 2,
                    "token_ids": [10, 11],
                    "positions": [0, 1],
                },
            },
        },
    },
    {
        "name": "layer3_shift_bounded_start_pos",
        "batch_size": 1,
        "running_req_ids": ["req-0"],
        "target_token_ids": [10, 11, 12, 13, 14],
        "target_positions": [0, 1, 2, 3, 4],
        "last_token_indices": [1],
        "common_attn_metadata": {
            "query_start_loc": [0, 5],
            "query_start_loc_cpu": [0, 5],
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
            "seq_lens_cpu": [5],
            "num_computed_tokens_cpu": [1],
            "slot_mapping": [100, 101, 102, 103, 104],
            "max_seq_len": 5,
            "cached_by_req": {
                "req-0": {
                    "len": 2,
                    "token_ids": [10, 11],
                    "positions": [0, 1],
                },
            },
        },
    },
    {
        "name": "layer3_shift2_bounded_remaining",
        "batch_size": 1,
        "running_req_ids": ["req-0"],
        "target_token_ids": [10, 11, 12, 13, 14],
        "target_positions": [0, 1, 2, 3, 4],
        "last_token_indices": [2],
        "common_attn_metadata": {
            "query_start_loc": [0, 5],
            "query_start_loc_cpu": [0, 5],
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
            "seq_lens_cpu": [5],
            "num_computed_tokens_cpu": [2],
            "slot_mapping": [100, 101, 102, 103, 104],
            "max_seq_len": 5,
            "cached_by_req": {
                "req-0": {
                    "len": 3,
                    "token_ids": [10, 11, 12],
                    "positions": [0, 1, 2],
                },
            },
        },
    },
    {
        "name": "layer3_shift3_full_cache_window",
        "batch_size": 1,
        "running_req_ids": ["req-0"],
        "target_token_ids": [20, 21, 22, 23, 24],
        "target_positions": [0, 1, 2, 3, 4],
        "last_token_indices": [1],
        "common_attn_metadata": {
            "query_start_loc": [0, 5],
            "query_start_loc_cpu": [0, 5],
            "seq_lens": [5],
            "seq_lens_cpu": [5],
            "num_computed_tokens_cpu": [3],
            "slot_mapping": [100, 101, 102, 103, 104],
            "max_seq_len": 5,
        },
        "expected": {
            "prev_token_ids": [20, 21, 22, 23, 24],
            "prev_positions": [0, 1, 2, 3, 4],
            "last_token_indices": [1],
            "seq_lens": [5],
            "seq_lens_cpu": [5],
            "num_computed_tokens_cpu": [3],
            "slot_mapping": [100, 101, 102, 103, 104],
            "max_seq_len": 5,
            "cached_by_req": {
                "req-0": {
                    "len": 2,
                    "token_ids": [20, 21],
                    "positions": [0, 1],
                },
            },
        },
    },
    {
        "name": "layer3_batch2_shift1_and1",
        "batch_size": 2,
        "running_req_ids": ["req-0", "req-1"],
        "target_token_ids": [10, 11, 12, 13, 20, 21, 22],
        "target_positions": [0, 1, 2, 3, 0, 1, 2],
        "last_token_indices": [1, 5],
        "common_attn_metadata": {
            "query_start_loc": [0, 4, 7],
            "query_start_loc_cpu": [0, 4, 7],
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
            "seq_lens_cpu": [4, 3],
            "num_computed_tokens_cpu": [1, 1],
            "slot_mapping": [100, 101, 102, 103, 200, 201, 202],
            "max_seq_len": 4,
            "cached_by_req": {
                "req-0": {
                    "len": 2,
                    "token_ids": [10, 11],
                    "positions": [0, 1],
                },
                "req-1": {
                    "len": 2,
                    "token_ids": [20, 21],
                    "positions": [0, 1],
                },
            },
        },
    },
    {
        "name": "layer3_batch4_mixed_shifts",
        "batch_size": 4,
        "running_req_ids": ["req-0", "req-1", "req-2", "req-3"],
        "target_token_ids": [10, 11, 20, 21, 22, 30, 31, 32, 33, 40, 41, 42],
        "target_positions": [0, 1, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2],
        "last_token_indices": [1, 2, 6, 10],
        "common_attn_metadata": {
            "query_start_loc": [0, 2, 5, 9, 12],
            "query_start_loc_cpu": [0, 2, 5, 9, 12],
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
            "prev_token_ids": [10, 11, 911, 20, 21, 30, 31, 32, 33, 40, 41, 42],
            "prev_positions": [0, 1, 1, 1, 2, 0, 1, 2, 3, 0, 1, 2],
            "last_token_indices": [1, 3, 6, 10],
            "seq_lens": [2, 2, 4, 3],
            "seq_lens_cpu": [2, 2, 4, 3],
            "num_computed_tokens_cpu": [0, 0, 2, 1],
            "slot_mapping": [
                100,
                101,
                991,
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
            "max_seq_len": 4,
            "cached_by_req": {
                "req-0": {
                    "len": 2,
                    "token_ids": [10, 11],
                    "positions": [0, 1],
                },
                "req-1": {
                    "len": 2,
                    "token_ids": [911, 20],
                    "positions": [1, 1],
                },
                "req-2": {
                    "len": 2,
                    "token_ids": [30, 31],
                    "positions": [0, 1],
                },
                "req-3": {
                    "len": 2,
                    "token_ids": [40, 41],
                    "positions": [0, 1],
                },
            },
        },
    },
    {
        "name": "layer3_batch2_shift0_and2",
        "batch_size": 2,
        "running_req_ids": ["req-0", "req-1"],
        "target_token_ids": [30, 31, 32, 40, 41, 42, 43],
        "target_positions": [0, 1, 2, 0, 1, 2, 3],
        "last_token_indices": [2, 4],
        "common_attn_metadata": {
            "query_start_loc": [0, 3, 7],
            "query_start_loc_cpu": [0, 3, 7],
            "seq_lens": [3, 4],
            "seq_lens_cpu": [3, 4],
            "num_computed_tokens_cpu": [0, 2],
            "slot_mapping": [100, 101, 102, 200, 201, 202, 203],
            "max_seq_len": 4,
        },
        "expected": {
            "prev_token_ids": [30, 31, 32, 40, 41, 42, 43],
            "prev_positions": [0, 1, 2, 0, 1, 2, 3],
            "last_token_indices": [2, 4],
            "seq_lens": [3, 4],
            "seq_lens_cpu": [3, 4],
            "num_computed_tokens_cpu": [0, 2],
            "slot_mapping": [100, 101, 102, 200, 201, 202, 203],
            "max_seq_len": 4,
            "cached_by_req": {
                "req-0": {
                    "len": 3,
                    "token_ids": [30, 31, 32],
                    "positions": [0, 1, 2],
                },
                "req-1": {
                    "len": 2,
                    "token_ids": [40, 41],
                    "positions": [0, 1],
                },
            },
        },
    },
]

LAYER5_CASES = [
    {
        "name": "layer5_cache_window5",
        "batch_size": 1,
        "running_req_ids": ["req-0"],
        "target_token_ids": [1, 2, 3, 4, 5, 6],
        "target_positions": [0, 1, 2, 3, 4, 5],
        "last_token_indices": [2],
        "common_attn_metadata": {
            "query_start_loc": [0, 6],
            "query_start_loc_cpu": [0, 6],
            "seq_lens": [6],
            "seq_lens_cpu": [6],
            "num_computed_tokens_cpu": [2],
            "slot_mapping": [100, 101, 102, 103, 104, 105],
            "max_seq_len": 6,
        },
        "expected": {
            "prev_token_ids": [1, 2, 3, 4, 5, 6],
            "prev_positions": [0, 1, 2, 3, 4, 5],
            "last_token_indices": [2],
            "seq_lens": [6],
            "seq_lens_cpu": [6],
            "num_computed_tokens_cpu": [2],
            "slot_mapping": [100, 101, 102, 103, 104, 105],
            "max_seq_len": 6,
            "cached_by_req": {
                "req-0": {
                    "len": 3,
                    "token_ids": [1, 2, 3],
                    "positions": [0, 1, 2],
                },
            },
        },
    },
]


def _run_adjust_input_case(proposer_stub, case, layer_num):
    proposer_stub.layer_num = layer_num
    proposer_stub.running_req_ids = case["running_req_ids"]
    meta = case["common_attn_metadata"]
    common_attn_metadata = SimpleNamespace(
        query_start_loc=torch.tensor(meta["query_start_loc"], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor(meta["query_start_loc"], dtype=torch.int32),
        seq_lens=torch.tensor(meta["seq_lens"], dtype=torch.int32),
        seq_lens_cpu=torch.tensor(meta["seq_lens_cpu"], dtype=torch.int32),
        num_computed_tokens_cpu=torch.tensor(
            meta["num_computed_tokens_cpu"], dtype=torch.int32
        ),
        slot_mapping=torch.tensor(meta["slot_mapping"], dtype=torch.int32),
        max_seq_len=meta["max_seq_len"],
    )

    target_token_ids = torch.tensor(case["target_token_ids"], dtype=torch.int32)
    target_positions = torch.tensor(case["target_positions"], dtype=torch.int64)
    target_hidden_states = torch.arange(
        0, target_token_ids.numel() * 3, dtype=torch.float32
    ).reshape(-1, 3)
    last_token_indices = torch.tensor(case["last_token_indices"], dtype=torch.int32)

    prev_token_ids, prev_positions, _, _, _ = proposer_stub.adjust_input(
        batch_size=case["batch_size"],
        target_token_ids=target_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        last_token_indices=last_token_indices,
        common_attn_metadata=common_attn_metadata,
    )

    expected = case["expected"]
    assert prev_token_ids.tolist() == expected["prev_token_ids"]
    assert prev_positions.tolist() == expected["prev_positions"]
    assert last_token_indices.tolist() == expected["last_token_indices"]
    assert common_attn_metadata.seq_lens.tolist() == expected["seq_lens"]
    assert common_attn_metadata.seq_lens_cpu.tolist() == expected["seq_lens_cpu"]
    assert (
        common_attn_metadata.num_computed_tokens_cpu.tolist()
        == expected["num_computed_tokens_cpu"]
    )
    assert common_attn_metadata.slot_mapping.tolist() == expected["slot_mapping"]
    assert common_attn_metadata.max_seq_len == expected["max_seq_len"]

    for req_id, cached_expect in expected["cached_by_req"].items():
        cached = proposer_stub.draft_input_states_pool[req_id]
        assert cached.len == cached_expect["len"]
        assert cached.token_ids.tolist() == cached_expect["token_ids"]
        assert cached.positions.tolist() == cached_expect["positions"]


@pytest.mark.parametrize(
    "case", LAYER3_CASES, ids=[case["name"] for case in LAYER3_CASES]
)
def test_adjust_input_layer3_cases(proposer_stub, case):
    _run_adjust_input_case(proposer_stub, case, layer_num=3)


@pytest.mark.parametrize(
    "case", LAYER5_CASES, ids=[case["name"] for case in LAYER5_CASES]
)
def test_adjust_input_layer5_cases(proposer_stub, case):
    _run_adjust_input_case(proposer_stub, case, layer_num=5)
