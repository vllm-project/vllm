# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for `make_kv_sharing_fast_prefill_common_attn_metadata`.

The metadata builders read `query_start_loc_cpu` on the host as soon as the
fast-prefill metadata is built, so it must hold final values on return even
while the GPU is still busy with earlier work (async scheduling).
"""

import pytest
import torch

from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
    make_kv_sharing_fast_prefill_common_attn_metadata,
)


def _make_metadata(
    query_lens: list[int], logits_per_req: list[int], device: str
) -> tuple[CommonAttentionMetadata, list[int]]:
    num_reqs = len(query_lens)
    query_start_loc = [0]
    for q in query_lens:
        query_start_loc.append(query_start_loc[-1] + q)
    # The logits of each request are its last `k` scheduled tokens.
    logits_indices: list[int] = []
    for i, k in enumerate(logits_per_req):
        end = query_start_loc[i + 1]
        logits_indices.extend(range(end - k, end))
    expected = [0]
    for k in logits_per_req:
        expected.append(expected[-1] + k)
    # Simulate CUDA graph padding of the logits indices.
    padded = logits_indices + [logits_indices[-1]] * 3
    qsl = torch.tensor(query_start_loc, dtype=torch.int32)
    num_tokens = query_start_loc[-1]
    metadata = CommonAttentionMetadata(
        query_start_loc=qsl.to(device),
        query_start_loc_cpu=qsl,
        seq_lens=torch.tensor([q + 8 for q in query_lens], dtype=torch.int32).to(
            device
        ),
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=max(query_lens),
        max_seq_len=max(query_lens) + 8,
        block_table_tensor=torch.zeros(num_reqs, 4, dtype=torch.int32, device=device),
        slot_mapping=torch.zeros(num_tokens, dtype=torch.int64, device=device),
        logits_indices_padded=torch.tensor(padded, dtype=torch.int32, device=device),
        num_logits_indices=len(logits_indices),
        max_logits_per_req=max(logits_per_req),
    )
    return metadata, expected


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "query_lens,logits_per_req",
    [
        # One logit per request (no speculative decoding), mixed chunk sizes.
        ([15, 5, 8, 300], [1, 1, 1, 1]),
        # Several logits per request (speculative decoding).
        ([15, 5, 8], [1, 2, 1]),
        ([40, 3, 17, 9], [4, 3, 1, 2]),
    ],
)
def test_fast_prefill_query_start_loc(
    device: str, query_lens: list[int], logits_per_req: list[int]
):
    metadata, expected = _make_metadata(query_lens, logits_per_req, device)
    out = make_kv_sharing_fast_prefill_common_attn_metadata(metadata)
    assert out.query_start_loc_cpu.device.type == "cpu"
    assert out.query_start_loc_cpu.tolist() == expected
    assert out.query_start_loc.tolist() == expected
    assert out.num_reqs == len(query_lens)
    assert out.num_actual_tokens == sum(logits_per_req)
    assert out.max_query_len == max(logits_per_req)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("logits_per_req", [[1, 1, 1, 1], [2, 1, 3, 1]])
def test_fast_prefill_query_start_loc_cpu_ready_while_gpu_busy(
    logits_per_req: list[int],
):
    """With async scheduling the CPU builds metadata while the GPU still runs
    the previous step. Keep the stream busy and read the host offsets right
    away, as the attention metadata builders do."""
    metadata, expected = _make_metadata([64, 7, 33, 12], logits_per_req, "cuda")
    torch.accelerator.synchronize()
    torch.cuda._sleep(2_000_000_000)  # keep the current stream busy
    out = make_kv_sharing_fast_prefill_common_attn_metadata(metadata)
    observed = out.query_start_loc_cpu.tolist()  # no synchronization before
    torch.accelerator.synchronize()
    assert observed == expected
