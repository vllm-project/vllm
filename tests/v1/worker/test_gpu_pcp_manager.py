# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.pcp_manager import PCPManager

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")


def _make_batch(
    device: torch.device,
    num_scheduled_tokens: np.ndarray,
    num_computed_tokens: np.ndarray,
    prefill_lens: np.ndarray,
    num_draft_tokens_per_req: np.ndarray,
) -> tuple[SimpleNamespace, InputBatch]:
    num_reqs = len(num_scheduled_tokens)
    query_start_loc_np = np.zeros(num_reqs + 1, dtype=np.int32)
    np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1:])
    num_tokens = int(query_start_loc_np[-1])
    idx_mapping_np = np.arange(num_reqs, dtype=np.intp)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int64, device=device)

    num_logits_per_req = num_draft_tokens_per_req + 1
    cu_num_logits_np = np.zeros(num_reqs + 1, dtype=np.int32)
    np.cumsum(num_logits_per_req, out=cu_num_logits_np[1:])

    positions = np.concatenate(
        [
            np.arange(computed, computed + scheduled, dtype=np.int64)
            for computed, scheduled in zip(num_computed_tokens, num_scheduled_tokens)
        ]
    )
    input_ids = torch.arange(1000, 1000 + num_tokens, dtype=torch.int32, device=device)
    seq_lens = torch.from_numpy(num_computed_tokens + num_scheduled_tokens).to(device)
    query_start_loc = torch.from_numpy(query_start_loc_np).to(device)
    cu_num_logits = torch.from_numpy(cu_num_logits_np).to(device)

    req_states = SimpleNamespace(
        last_sampled_tokens=torch.tensor([[101], [102]], device=device),
        prefill_len=SimpleNamespace(gpu=torch.from_numpy(prefill_lens).to(device)),
        draft_tokens=torch.tensor([[201, 202, 203], [211, 212, 213]], device=device),
    )
    input_batch = InputBatch(
        req_ids=[f"req-{i}" for i in range(num_reqs)],
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs,
        idx_mapping=idx_mapping,
        idx_mapping_np=idx_mapping_np,
        expanded_idx_mapping=idx_mapping,
        expanded_local_pos=torch.zeros(num_reqs, dtype=torch.int32, device=device),
        num_scheduled_tokens=num_scheduled_tokens,
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens,
        num_draft_tokens=int(num_draft_tokens_per_req.sum()),
        num_draft_tokens_per_req=num_draft_tokens_per_req,
        query_start_loc=query_start_loc,
        query_start_loc_np=query_start_loc_np,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=torch.from_numpy(
            num_computed_tokens + num_scheduled_tokens
        ),
        dcp_local_seq_lens=None,
        num_computed_tokens_np=num_computed_tokens,
        prefill_len_np=prefill_lens,
        num_computed_prefill_tokens_np=np.minimum(num_computed_tokens, prefill_lens),
        is_prefilling_np=num_computed_tokens < prefill_lens,
        has_prefill=bool(np.any(num_computed_tokens < prefill_lens)),
        max_seq_len_np=None,
        input_ids=input_ids,
        positions=torch.from_numpy(positions).to(device),
        is_padding=torch.zeros(num_tokens, dtype=torch.bool, device=device),
        logits_indices=torch.empty(0, dtype=torch.int64, device=device),
        cu_num_logits=cu_num_logits,
        cu_num_logits_np=cu_num_logits_np,
        has_structured_output_reqs=False,
        prompt_lens=None,
    )
    return req_states, input_batch


def _make_manager(device: torch.device, req_states: SimpleNamespace) -> PCPManager:
    return PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=device,
        req_states=req_states,
        max_num_reqs=2,
        max_num_tokens=16,
    )


def test_pcp_partitions_mtp_decode_batch():
    device = torch.device("cuda")
    req_states, global_batch = _make_batch(
        device,
        num_scheduled_tokens=np.array([4, 2], dtype=np.int32),
        num_computed_tokens=np.array([10, 20], dtype=np.int32),
        prefill_lens=np.array([5, 5], dtype=np.int32),
        num_draft_tokens_per_req=np.array([3, 1], dtype=np.int32),
    )

    local_batch = _make_manager(device, req_states).partition_batch(global_batch)

    assert local_batch.num_draft_tokens == 4
    assert local_batch.num_draft_tokens_per_req is not None
    assert local_batch.num_draft_tokens_per_req.tolist() == [3, 1]
    assert local_batch.cu_num_logits_np.tolist() == [0, 4, 6]
    assert local_batch.input_ids.tolist() == [101, 201, 202, 203, 102, 211]
    assert local_batch.logits_indices.tolist() == [0, 1, 2, 3, 4, 5]
    assert local_batch.expanded_idx_mapping.tolist() == [0, 0, 0, 0, 1, 1]
    assert local_batch.expanded_local_pos.tolist() == [0, 1, 2, 3, 0, 1]


def test_pcp_partitions_mixed_prefill_and_mtp_decode_batch():
    device = torch.device("cuda")
    req_states, global_batch = _make_batch(
        device,
        num_scheduled_tokens=np.array([4, 8], dtype=np.int32),
        num_computed_tokens=np.array([10, 0], dtype=np.int32),
        prefill_lens=np.array([5, 8], dtype=np.int32),
        num_draft_tokens_per_req=np.array([3, 0], dtype=np.int32),
    )

    local_batch = _make_manager(device, req_states).partition_batch(global_batch)

    assert local_batch.num_scheduled_tokens.tolist() == [4, 2, 2]
    assert local_batch.num_draft_tokens == 3
    assert local_batch.num_draft_tokens_per_req is not None
    assert local_batch.num_draft_tokens_per_req.tolist() == [3, 0, 0]
    assert local_batch.cu_num_logits_np.tolist() == [0, 4, 5, 6]
    assert local_batch.input_ids.tolist() == [
        101,
        201,
        202,
        203,
        1010,
        1011,
        1004,
        1005,
    ]
    assert local_batch.logits_indices.tolist() == [0, 1, 2, 3, 5, 7]
    assert local_batch.expanded_idx_mapping.tolist() == [0, 0, 0, 0, 1, 1]
    assert local_batch.expanded_local_pos.tolist() == [0, 1, 2, 3, 0, 0]
