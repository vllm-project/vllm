# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import CpuArchEnum, current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.v1.worker.cpu.input_batch import (
    InputBatch,
    InputBuffers,
    combine_sampled_and_draft_tokens,
    post_update,
    prepare_pos_seq_lens,
    prepare_prefill_inputs,
)
from vllm.v1.worker.cpu.rope import RopeState

pytestmark = pytest.mark.skipif(
    not current_platform.is_cpu()
    or HAS_TRITON
    or current_platform.get_cpu_architecture() == CpuArchEnum.X86,
    reason="requires a non-x86 CPU platform without Triton",
)


@pytest.mark.parametrize(
    "num_reqs,num_tokens",
    [(256, 496), (128, 512), (3, 8), (1, 7)],
)
def test_make_dummy_distributes_remainder(num_reqs: int, num_tokens: int) -> None:
    buffers = InputBuffers(
        max_num_reqs=num_reqs,
        max_num_tokens=num_tokens,
        device=torch.device("cpu"),
    )
    batch = InputBatch.make_dummy(num_reqs, num_tokens, buffers)

    max_per_req = -(-num_tokens // num_reqs)
    assert batch.num_scheduled_tokens.sum() == num_tokens
    assert batch.num_scheduled_tokens.max() == max_per_req
    assert batch.num_scheduled_tokens.min() >= num_tokens // num_reqs
    assert (batch.num_scheduled_tokens[:-1] <= batch.num_scheduled_tokens[1:]).all()

    query_lens = batch.query_start_loc_np[1:] - batch.query_start_loc_np[:-1]
    assert (query_lens == batch.num_scheduled_tokens).all()
    assert torch.equal(batch.seq_lens, torch.from_numpy(batch.num_scheduled_tokens))
    assert batch.query_start_loc_np[-1] == num_tokens
    assert torch.equal(
        batch.query_start_loc, torch.from_numpy(batch.query_start_loc_np)
    )


def test_prepare_prefill_inputs_cpu() -> None:
    input_ids_storage = torch.full((8,), -1, dtype=torch.int32)
    input_ids = input_ids_storage[1:7]
    next_prefill_storage = torch.full((1, 8), -1, dtype=torch.int32)
    next_prefill_tokens = next_prefill_storage[:, 1:7]
    all_token_ids_storage = torch.full((6, 12), -1, dtype=torch.int32)
    all_token_ids = all_token_ids_storage[:, 1:11]
    for req_state_idx in range(6):
        all_token_ids[req_state_idx] = req_state_idx * 100 + torch.arange(
            10, dtype=torch.int32
        )

    idx_mapping = torch.tensor([4, 1, 3], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 3, 5, 6], dtype=torch.int32)
    prefill_len = torch.tensor([0, 4, 0, 8, 7, 0], dtype=torch.int32)
    num_computed_tokens = torch.tensor([0, 4, 0, 4, 2, 0], dtype=torch.int32)

    prepare_prefill_inputs(
        input_ids,
        next_prefill_tokens,
        idx_mapping,
        query_start_loc,
        all_token_ids,
        prefill_len,
        num_computed_tokens,
    )

    assert input_ids.tolist() == [402, 403, 404, -1, -1, 304]
    assert next_prefill_tokens[:, 4].tolist() == [405]
    assert next_prefill_tokens[:, 3].tolist() == [305]
    assert next_prefill_tokens[:, 1].tolist() == [-1]
    assert torch.all(next_prefill_tokens[:, [0, 2, 5]] == -1)


def test_get_prompt_logprobs_token_ids_cpu() -> None:
    all_token_ids_storage = torch.full((6, 12), -1, dtype=torch.int32)
    all_token_ids = all_token_ids_storage[:, 1:11]
    for req_state_idx in range(6):
        all_token_ids[req_state_idx] = req_state_idx * 100 + torch.arange(
            10, dtype=torch.int32
        )

    token_ids = ops.get_prompt_logprobs_token_ids_cpu(
        num_tokens=6,
        query_start_loc=torch.tensor([0, 3, 5, 6], dtype=torch.int32),
        idx_mapping=torch.tensor([4, 1, 3], dtype=torch.int64),
        num_computed_tokens=torch.tensor([0, 4, 0, 1, 2, 0], dtype=torch.int32),
        all_token_ids=all_token_ids,
    )

    assert token_ids.dtype == torch.int64
    assert token_ids.tolist() == [403, 404, 405, 105, 106, 302]


def test_prepare_pos_seq_lens_cpu() -> None:
    positions_storage = torch.full((8,), -1, dtype=torch.int64)
    positions = positions_storage[1:]
    seq_lens_storage = torch.full((8,), -1, dtype=torch.int32)
    seq_lens = seq_lens_storage[1:7]

    idx_mapping = torch.tensor([3, 1, 4], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 3, 5, 6], dtype=torch.int32)
    num_computed_tokens = torch.tensor([0, 7, 0, 2, 10], dtype=torch.int32)

    prepare_pos_seq_lens(
        idx_mapping,
        query_start_loc,
        num_computed_tokens,
        positions,
        seq_lens,
    )

    assert positions.tolist() == [2, 3, 4, 7, 8, 10, -1]
    assert seq_lens.tolist() == [5, 9, 11, 0, 0, 0]
    assert positions_storage[0].item() == -1
    assert seq_lens_storage[[0, 7]].tolist() == [-1, -1]


def test_prepare_mrope_positions_cpu() -> None:
    num_dims = 3
    positions_storage = torch.full((num_dims, 10), -1, dtype=torch.int64)
    positions = positions_storage[:, 1:9]

    prefill_positions_storage = torch.full((15, 10), -1, dtype=torch.int32)
    prefill_positions = prefill_positions_storage[:, 1:9]
    for req_state_idx in range(5):
        for dim in range(num_dims):
            prefill_positions[req_state_idx * num_dims + dim] = (
                req_state_idx * 1000 + dim * 100 + torch.arange(8, dtype=torch.int32)
            )

    ops.prepare_mrope_positions_cpu(
        positions,
        prefill_positions,
        prefill_delta=torch.tensor([0, 10, 0, 20, 30], dtype=torch.int32),
        idx_mapping=torch.tensor([3, 1, 4], dtype=torch.int64),
        query_start_loc=torch.tensor([0, 3, 6, 8], dtype=torch.int32),
        prefill_lens=torch.tensor([0, 5, 0, 6, 4], dtype=torch.int32),
        num_computed_tokens=torch.tensor([0, 5, 0, 2, 3], dtype=torch.int32),
    )

    assert positions.tolist() == [
        [3002, 3003, 3004, 15, 16, 17, 4003, 34],
        [3102, 3103, 3104, 15, 16, 17, 4103, 34],
        [3202, 3203, 3204, 15, 16, 17, 4203, 34],
    ]
    assert torch.all(positions_storage[:, [0, 9]] == -1)


def test_cpu_rope_state_prepares_model_positions() -> None:
    class MRoPEModel(torch.nn.Module):
        supports_mrope = True

        def get_mrope_input_positions(
            self, input_tokens: list[int], mm_features: list
        ) -> tuple[torch.Tensor, int]:
            return torch.tensor([[100], [200]], dtype=torch.int64), 10

    state = RopeState(
        num_dims=2,
        max_num_reqs=2,
        max_num_tokens=2,
        max_model_len=2,
        device=torch.device("cpu"),
    )
    state.init_prefill_positions(
        req_idx=1,
        model=MRoPEModel(),
        prefill_token_ids=[42],
        mm_features=[],
    )

    state.prepare_positions(
        idx_mapping=torch.tensor([1], dtype=torch.int64),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        prefill_lens=torch.tensor([0, 1], dtype=torch.int32),
        num_computed_tokens=torch.tensor([0, 0], dtype=torch.int32),
    )

    assert state.get_positions(2).tolist() == [[100, 11], [200, 11]]

    state.update_prefill_positions(
        1, torch.tensor([[300], [400]], dtype=torch.int64), delta=20
    )
    state.prepare_positions(
        idx_mapping=torch.tensor([1], dtype=torch.int64),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        prefill_lens=torch.tensor([0, 1], dtype=torch.int32),
        num_computed_tokens=torch.tensor([0, 0], dtype=torch.int32),
    )

    assert state.read_prefill_positions(1, 1).tolist() == [[300], [400]]
    assert state.get_positions(2).tolist() == [[300, 21], [400, 21]]


def test_combine_sampled_and_draft_tokens_cpu_basic() -> None:
    input_ids = torch.tensor([10, 11, 12, -1], dtype=torch.int32)
    idx_mapping = torch.tensor([3, 1], dtype=torch.int64)
    last_sampled_tokens = torch.tensor([[40], [41], [42], [43]])
    query_start_loc = torch.tensor([0, 3, 4], dtype=torch.int32)
    seq_lens = torch.tensor([5, 6], dtype=torch.int32)
    prefill_len = torch.tensor([0, 5, 0, 5], dtype=torch.int32)
    draft_tokens = torch.empty((4, 0), dtype=torch.int64)
    cu_num_logits = torch.tensor([0, 1, 2], dtype=torch.int32)

    logits_indices = combine_sampled_and_draft_tokens(
        input_ids,
        idx_mapping,
        last_sampled_tokens,
        query_start_loc,
        seq_lens,
        prefill_len,
        draft_tokens,
        cu_num_logits,
        num_logits=2,
    )

    assert input_ids.tolist() == [10, 11, 12, 41]
    assert logits_indices.tolist() == [2, 3]


def test_post_update_cpu_without_optional_inputs() -> None:
    num_computed_tokens = torch.tensor([8, 13], dtype=torch.int32)
    last_sampled_tokens = torch.tensor([[70], [80]], dtype=torch.int64)
    all_token_ids = torch.full((2, 6), -1, dtype=torch.int32)
    total_len = torch.tensor([2, 3], dtype=torch.int32)

    post_update(
        idx_mapping=torch.tensor([1], dtype=torch.int64),
        num_computed_tokens=num_computed_tokens,
        last_sampled_tokens=last_sampled_tokens,
        output_bin_counts=None,
        sampled_tokens=torch.tensor([[4]], dtype=torch.int64),
        num_sampled=torch.tensor([1], dtype=torch.int32),
        num_rejected=torch.tensor([0], dtype=torch.int32),
        query_start_loc=None,
        all_token_ids=all_token_ids,
        total_len=total_len,
    )

    assert num_computed_tokens.tolist() == [8, 13]
    assert last_sampled_tokens.squeeze(1).tolist() == [70, 4]
    assert total_len.tolist() == [2, 4]
    assert all_token_ids[1].tolist() == [-1, -1, -1, 4, -1, -1]
