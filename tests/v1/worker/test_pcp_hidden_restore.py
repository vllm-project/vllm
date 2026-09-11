# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu.pcp_hidden_restore as transport
import vllm.v1.worker.gpu.pcp_manager as pcp
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize(
    "queries,prefilling",
    [
        ([9, 5], [True, True]),
        ([9, 1, 17, 1], [True, False, True, False]),
        ([1, 1], [False, False]),
        ([2, 1], [False, False]),
        ([1], [True]),
    ],
)
@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("multicast", [False, True])
def test_sampling_matches_global_rows(
    monkeypatch, world, queries, prefilling, dense, multicast
):
    """Preserve sampled order and dense prompt rows across ragged/padded PCP batches."""
    q = np.array(queries, dtype=np.int32)
    prefilling = np.array(prefilling)
    starts = np.concatenate(([0], np.cumsum(q))).astype(np.int32)
    global_hidden = torch.arange(int(starts[-1]) * 3).reshape(-1, 3).float()
    managers, local = [], []
    for rank in range(world):
        manager = pcp.PCPManager(world, rank, torch.device("cpu"))
        segments, counts = manager._build_batch_layout(
            q, np.zeros_like(q), prefilling, starts
        )
        hidden = torch.zeros(max(counts), 3)
        for segment in segments[rank]:
            hidden[segment.rank_local_batch_slice] = global_hidden[
                segment.global_batch_slice
            ]
        manager._global_batch = NS(
            logits_indices=torch.tensor(starts[1:] - 1, dtype=torch.int64),
            num_reqs=len(q),
        )
        managers.append(manager)
        local.append(hidden)
    replicated = not np.any(prefilling)
    full = torch.cat(local)
    packed = (
        None
        if replicated
        else torch.cat(
            [
                hidden[manager._sample_local_row_idx]
                for manager, hidden in zip(managers, local)
            ]
        )
    )
    for manager, hidden in zip(managers, local):

        def gather(tensor, dim, manager=manager, hidden=hidden):
            assert not replicated
            assert dim == 0
            if dense:
                torch.testing.assert_close(tensor, hidden)
                return full
            torch.testing.assert_close(tensor, hidden[manager._sample_local_row_idx])
            return packed

        def restore(h, idx, order, *, num_selected_rows):
            assert not dense and not replicated
            assert num_selected_rows == len(q)
            assert packed is not None
            return packed[order]

        monkeypatch.setattr(pcp, "get_pcp_group", lambda: NS(all_gather=gather))
        if multicast:
            manager._hidden_state_restorer = NS(restore_selected=restore)
        restored, sampled, _ = manager.restore_for_sampling(
            hidden,
            needs_prompt_hidden_states=dense,
        )
        torch.testing.assert_close(sampled, global_hidden[starts[1:] - 1])
        if dense:
            torch.testing.assert_close(restored, global_hidden)
        else:
            assert manager._hidden_restore_idx is None


@pytest.mark.parametrize("failed_phase", [1, 2])
def test_multicast_failure_requires_rank_agreement(monkeypatch, failed_phase):
    calls = []

    def agree(ready, **kwargs):
        calls.append("agree")
        if calls.count("agree") == failed_phase:
            ready.zero_()

    def rendezvous(*args):
        calls.append("rendezvous")
        return NS(multicast_ptr=1)

    monkeypatch.setattr(transport.torch_symm_mem, "empty", torch.empty)
    monkeypatch.setattr(transport.torch_symm_mem, "rendezvous", rendezvous)
    monkeypatch.setattr(transport.dist, "all_reduce", agree)
    with pytest.raises(
        transport.PCPMulticastUnavailableError, match="failed on at least one"
    ):
        transport.PCPMulticastHiddenStateRestorer(
            group=NS(group_name="test", size=lambda: 4),
            device=torch.device("cpu"),
            max_num_tokens=8,
            hidden_size=16,
            dtype=torch.bfloat16,
        )
    assert calls == (
        ["agree"] if failed_phase == 1 else ["agree", "rendezvous", "agree"]
    )


@pytest.mark.parametrize("available", [False, True])
def test_factory_coordinates_fallback(monkeypatch, available):
    sentinel = object()

    def create(**kwargs):
        if not available:
            raise transport.PCPMulticastUnavailableError("unsupported")
        return sentinel

    monkeypatch.setattr(pcp.PCPManager, "validate_config", lambda *args: None)
    monkeypatch.setattr(pcp, "get_pcp_group", lambda: NS(cpu_group=None))
    monkeypatch.setattr(pcp, "PCPMulticastHiddenStateRestorer", create)
    config = NS(
        parallel_config=NS(prefill_context_parallel_size=4),
        scheduler_config=NS(max_num_seqs=8),
        model_config=NS(get_hidden_size=lambda: 16, dtype=torch.bfloat16),
    )
    result = pcp.maybe_create_pcp_hidden_state_restorer(
        config, torch.device("cpu"), False
    )
    assert result is (sentinel if available else None)


def test_prompt_logprob_worker_exposes_dense_hidden_requirement() -> None:
    worker = PromptLogprobsWorker(max_num_reqs=4)
    worker.uses_prompt_logprobs[:2] = True
    worker.in_progress_prompt_logprobs["req0"] = []
    worker.in_progress_prompt_logprobs["req1"] = []
    input_batch = NS(
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0, 8], dtype=np.int32),
        prefill_len_np=np.array([8, 8], dtype=np.int32),
    )
    prompt_lens = np.array([8, 8, 0, 0], dtype=np.int32)

    assert worker.needs_prompt_hidden_states(input_batch, prompt_lens)

    input_batch.num_computed_prefill_tokens_np[:] = 8
    assert not worker.needs_prompt_hidden_states(input_batch, prompt_lens)

    input_batch.num_computed_prefill_tokens_np[:] = 0
    input_batch.prefill_len_np[0] = 8
    prompt_lens[0] = 4
    worker.uses_prompt_logprobs[1] = False
    assert not worker.needs_prompt_hidden_states(input_batch, prompt_lens)


def test_prompt_logprob_worker_skips_mask_without_active_requests() -> None:
    worker = PromptLogprobsWorker(max_num_reqs=4)

    assert not worker.needs_prompt_hidden_states(
        NS(),
        np.empty(0, dtype=np.int32),
    )
