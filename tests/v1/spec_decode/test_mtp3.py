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

HIDDEN_SIZE = 3


class DummyBuilder(AttentionMetadataBuilder):
    def __init__(self, return_value: str):
        # attention metadata builders normally take multiple runtime args;
        # the test double shortcuts that setup.
        self.return_value = return_value
        self.calls: list[dict] = []
        self.kv_cache_spec = None
        self.layer_names: list[str] = []
        self.vllm_config = None
        self.device = torch.device("cuda")

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
    if not torch.cuda.is_available():
        pytest.skip("MultiLayerEagleProposer.adjust_input is CUDA/Triton-only.")
    device = torch.device("cuda")
    proposer = MultiLayerEagleProposer.__new__(MultiLayerEagleProposer)
    proposer.layer_num = 3
    proposer.running_req_ids = ["req-0"]
    proposer.attn_layer_names = ["attn_layer"]
    proposer.indexer_layer_names = ["indexer_layer"]
    proposer.attn_metadata_builder = DummyBuilder("attn_meta")
    proposer.draft_indexer_metadata_builder = DummyBuilder("indexer_meta")
    proposer.draft_input_states_pool = {
        "req-0": DraftInputStates(
            len=torch.tensor(3, dtype=torch.int32, device=device),
            token_ids=torch.tensor([800, 801, 802], dtype=torch.int32, device=device),
            hidden_states=torch.tensor(
                [[30.0, 31.0, 32.0], [33.0, 34.0, 35.0], [36.0, 37.0, 38.0]],
                device=device,
            ),
            positions=torch.tensor([40, 41, 42], dtype=torch.int64, device=device),
            slot_mapping=torch.tensor(
                [900, 901, 902], dtype=torch.int32, device=device
            ),
        ),
        "req-1": DraftInputStates(
            len=torch.tensor(2, dtype=torch.int32, device=device),
            token_ids=torch.tensor([910, 911, 0], dtype=torch.int32, device=device),
            hidden_states=torch.tensor(
                [[40.0, 41.0, 42.0], [43.0, 44.0, 45.0], [0.0, 0.0, 0.0]], device=device
            ),
            positions=torch.tensor([60, 61, 0], dtype=torch.int64, device=device),
            slot_mapping=torch.tensor([990, 991, 0], dtype=torch.int32, device=device),
        ),
        "req-2": DraftInputStates(
            len=torch.tensor(3, dtype=torch.int32, device=device),
            token_ids=torch.tensor([820, 821, 822], dtype=torch.int32, device=device),
            hidden_states=torch.tensor(
                [[46.0, 47.0, 48.0], [49.0, 50.0, 51.0], [52.0, 53.0, 54.0]],
                device=device,
            ),
            positions=torch.tensor([50, 51, 52], dtype=torch.int64, device=device),
            slot_mapping=torch.tensor(
                [920, 921, 922], dtype=torch.int32, device=device
            ),
        ),
        "req-3": DraftInputStates(
            len=torch.tensor(3, dtype=torch.int32, device=device),
            token_ids=torch.tensor([830, 831, 832], dtype=torch.int32, device=device),
            hidden_states=torch.tensor(
                [[55.0, 56.0, 57.0], [58.0, 59.0, 60.0], [61.0, 62.0, 63.0]],
                device=device,
            ),
            positions=torch.tensor([70, 71, 72], dtype=torch.int64, device=device),
            slot_mapping=torch.tensor(
                [930, 931, 932], dtype=torch.int32, device=device
            ),
        ),
    }
    return proposer


LAYER3_CASES = [
    {
        "name": "shift_0_at_sequence_end",
        "batch_size": 1,
        "running_req_ids": ["req-0"],
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
        "name": "batch2_short_seq_no_shift",
        "batch_size": 2,
        "running_req_ids": ["req-0", "req-1"],
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
        "name": "batch2_short_seq_shift_on_first",
        "batch_size": 2,
        "running_req_ids": ["req-0", "req-1"],
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
            "prev_token_ids": [802, 10, 20],
            "prev_positions": [42, 1, 0],
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
                    "positions": [42, 1],
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
        "name": "short_seq_len_2_shift_0_cache_len_1",
        "batch_size": 1,
        "running_req_ids": ["req-0"],
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
        "name": "short_seq_len_2_shift_1_cache_len_2",
        "batch_size": 1,
        "running_req_ids": ["req-0"],
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
            "prev_token_ids": [802, 7],
            "prev_positions": [42, 1],
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
                    "positions": [42, 1],
                },
            },
        },
    },
    {
        "name": "shift_bounded_by_start_pos_zero",
        "batch_size": 1,
        "running_req_ids": ["req-0"],
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
            "seq_lens_cpu": [4],
            "num_computed_tokens_cpu": [0],
            "slot_mapping": [100, 101, 102, 103],
            "max_seq_len": 4,
            "cached_by_req": {
                "req-0": {
                    "len": 2,
                    "token_ids": [10, 11],
                    "positions": [0, 2],
                },
            },
        },
    },
    {
        "name": "shift_bounded_by_start_pos",
        "batch_size": 1,
        "running_req_ids": ["req-0"],
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
        "name": "shift_2_bounded_by_remaining",
        "batch_size": 1,
        "running_req_ids": ["req-0"],
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
        "name": "shift_3_full_cache_window",
        "batch_size": 1,
        "running_req_ids": ["req-0"],
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
            "seq_lens_cpu": [5],
            "num_computed_tokens_cpu": [3],
            "slot_mapping": [100, 101, 102, 103, 104],
            "max_seq_len": 5,
            "cached_by_req": {
                "req-0": {
                    "len": 2,
                    "token_ids": [20, 21],
                    "positions": [0, 3],
                },
            },
        },
    },
    {
        "name": "batch2_shift_1_and_1",
        "batch_size": 2,
        "running_req_ids": ["req-0", "req-1"],
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
        "name": "batch4_mixed_shifts",
        "batch_size": 4,
        "running_req_ids": ["req-0", "req-1", "req-2", "req-3"],
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
            "prev_token_ids": [10, 11, 911, 20, 21, 30, 31, 32, 33, 40, 41, 42],
            "prev_positions": [0, 1, 61, 1, 2, 0, 2, 3, 4, 0, 1, 2],
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
                    "positions": [61, 1],
                },
                "req-2": {
                    "len": 2,
                    "token_ids": [30, 31],
                    "positions": [0, 2],
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
        "name": "batch2_shift_0_and_2",
        "batch_size": 2,
        "running_req_ids": ["req-0", "req-1"],
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
                    "positions": [0, 3],
                },
            },
        },
    },
]


def _run_adjust_input_case(proposer_stub, case, layer_num):
    proposer_stub.layer_num = layer_num
    proposer_stub.running_req_ids = case["running_req_ids"]
    max_shift = proposer_stub.layer_num
    device = torch.device("cuda")

    # Tests may reuse the same proposer stub across different layer_num values.
    # Ensure the cached per-request state tensors are always stored with shape
    # [MAX_SHIFT, ...] (padded), matching the production invariant.
    padded_pool = {}
    for req_id, state in proposer_stub.draft_input_states_pool.items():
        if state.token_ids.numel() == max_shift:
            padded_pool[req_id] = state
            continue
        old = state
        max_shift_tensor = old.len.new_tensor(max_shift, dtype=torch.int32)
        new_len = torch.minimum(old.len.to(dtype=torch.int32), max_shift_tensor)
        token_ids = torch.zeros(
            (max_shift,), dtype=old.token_ids.dtype, device=old.token_ids.device
        )
        positions = torch.zeros(
            (max_shift,), dtype=old.positions.dtype, device=old.positions.device
        )
        slot_mapping = torch.zeros(
            (max_shift,), dtype=old.slot_mapping.dtype, device=old.slot_mapping.device
        )
        hidden_states = torch.zeros(
            (max_shift, old.hidden_states.shape[1]),
            dtype=old.hidden_states.dtype,
            device=old.hidden_states.device,
        )
        n_copy = min(old.token_ids.numel(), max_shift)
        token_ids[:n_copy].copy_(old.token_ids[:n_copy])
        positions[:n_copy].copy_(old.positions[:n_copy])
        slot_mapping[:n_copy].copy_(old.slot_mapping[:n_copy])
        hidden_states[:n_copy].copy_(old.hidden_states[:n_copy])
        padded_pool[req_id] = DraftInputStates(
            len=new_len,
            token_ids=token_ids,
            hidden_states=hidden_states,
            positions=positions,
            slot_mapping=slot_mapping,
        )
    proposer_stub.draft_input_states_pool = padded_pool

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
            meta["slot_mapping"], dtype=torch.int32, device=device
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

    # `MultiLayerEagleProposer.adjust_input` changed its return signature over
    # time (it now returns 4 values in this repo). Only the first two are used
    # by these tests; the rest are verified via side effects on
    # `last_token_indices`, `common_attn_metadata`, and the request cache.
    out = proposer_stub.adjust_input(
        batch_size=case["batch_size"],
        target_token_ids=target_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        last_token_indices=last_token_indices,
        common_attn_metadata=common_attn_metadata,
    )
    prev_token_ids, prev_positions = out[0], out[1]

    expected = case["expected"]
    assert prev_token_ids.cpu().tolist() == expected["prev_token_ids"]
    assert prev_positions.cpu().tolist() == expected["prev_positions"]
    assert last_token_indices.cpu().tolist() == expected["last_token_indices"]
    assert common_attn_metadata.seq_lens.cpu().tolist() == expected["seq_lens"]

    # NOTE ignore check for num_computed_tokens_cpu and seq_lens_cpu
    # assert common_attn_metadata.seq_lens_cpu.tolist(
    # ) == expected["seq_lens_cpu"]
    # query_lens = (query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).tolist()
    # expected_num_computed_tokens_cpu = [
    #     seq_len - query_len
    #     for seq_len, query_len in zip(expected["seq_lens"], query_lens)
    # ]
    # assert common_attn_metadata.num_computed_tokens_cpu.tolist(
    # ) == expected_num_computed_tokens_cpu

    assert common_attn_metadata.slot_mapping.cpu().tolist() == expected["slot_mapping"]

    for req_id, cached_expect in expected["cached_by_req"].items():
        cached = proposer_stub.draft_input_states_pool[req_id]
        cache_len = cached_expect["len"]
        expected_len = torch.tensor(
            cache_len, dtype=cached.len.dtype, device=cached.len.device
        )
        assert torch.equal(cached.len, expected_len)
        assert cached.token_ids[:cache_len].cpu().tolist() == cached_expect["token_ids"]
        if cached.positions.dim() == 1:
            assert (
                cached.positions[:cache_len].cpu().tolist()
                == cached_expect["positions"]
            )
        else:
            assert (
                cached.positions[:, :cache_len].cpu().tolist()
                == cached_expect["positions"]
            )


@pytest.mark.parametrize(
    "case", LAYER3_CASES, ids=[case["name"] for case in LAYER3_CASES]
)
def test_adjust_input_layer3_cases(proposer_stub, case):
    _run_adjust_input_case(proposer_stub, case, layer_num=3)
