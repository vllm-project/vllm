# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 kpool indexer cache writes under prefill context parallelism."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.models.glm5next.nvidia import sparse_indexer as indexer
from vllm.v1.worker.gpu import pcp_manager as pcp_module


def test_pcp_prefill_restore_matches_global_order(monkeypatch):
    """Every PCP rank pools the global prefill tokens in sequence order.

    Replicated decodes stay local, and a short prefill leaves some ranks empty.
    Poisoned padding must not reach the restored prefill inputs.
    """
    pcp_size, num_decodes, num_decode_tokens = 4, 2, 2
    lens = np.asarray([1, 1, 35, 3], dtype=np.int32)
    is_prefilling = np.asarray([False, False, True, True])
    query_start_loc = np.asarray([0, *np.cumsum(lens)], dtype=np.int32)
    num_tokens = int(query_start_loc[-1])
    manager = pcp_module.PCPManager(pcp_size, 0, torch.device("cpu"))
    segments, _ = manager._build_batch_layout(
        lens,
        np.zeros_like(lens),
        is_prefilling,
        query_start_loc,
        padded_num_tokens=num_tokens,
    )

    k = torch.randn(num_tokens, 4)
    gate = torch.randn_like(k)
    slots = torch.arange(num_tokens) * 10
    tail_slots = slots + 1

    def gathered(values):
        payload = values.new_full((pcp_size, *values.shape), float("nan"))
        for rank, rank_segments in enumerate(segments):
            for segment in rank_segments:
                payload[rank, segment.rank_local_batch_slice] = values[
                    segment.global_batch_slice
                ]
        return payload.flatten(0, 1)

    gathered_k, gathered_gate = gathered(k), gathered(gate)
    local_k, local_gate = gathered_k[:num_tokens], gathered_gate[:num_tokens]
    payloads = {
        local_k.data_ptr(): gathered_k,
        local_gate.data_ptr(): gathered_gate,
    }
    group = SimpleNamespace(all_gather=lambda value, dim=0: payloads[value.data_ptr()])
    monkeypatch.setattr(pcp_module, "get_pcp_group", lambda: group)
    cache_slots = manager._convert_to_gathered_slot_mappings(
        torch.stack((slots, tail_slots))
    )
    forward_context = SimpleNamespace(additional_kwargs={"pcp_manager": manager})
    monkeypatch.setattr(indexer, "get_forward_context", lambda: forward_context)

    def restore():
        return indexer._restore_kpool_pcp_prefill(
            local_k,
            local_gate,
            cache_slots[0],
            cache_slots[1],
            num_decodes,
            num_decode_tokens,
        )

    # Capture and dummy batches have no partition to restore.
    manager._global_batch = None
    assert restore() is None

    manager._global_batch = SimpleNamespace(
        has_prefill=True, num_reqs=len(lens), is_prefilling_np=is_prefilling
    )
    prefill = slice(num_decode_tokens, None)
    for actual, expected in zip(
        restore(), (k[prefill], gate[prefill], slots[prefill], tail_slots[prefill])
    ):
        torch.testing.assert_close(actual, expected)


def test_pcp_prefill_restore_requires_the_pcp_manager(monkeypatch):
    """Without the PCP manager, kpool prefill writes must fail loudly."""
    forward_context = SimpleNamespace(additional_kwargs={})
    monkeypatch.setattr(indexer, "get_forward_context", lambda: forward_context)
    k = torch.zeros(4, 2)
    with pytest.raises(RuntimeError, match="PCP manager"):
        indexer._restore_kpool_pcp_prefill(k, k, torch.zeros(8), None, 0, 0)
