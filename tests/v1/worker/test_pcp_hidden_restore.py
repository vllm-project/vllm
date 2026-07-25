# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu.pcp_hidden_restore as pcp_hidden_restore
import vllm.v1.worker.gpu.pcp_manager as pcp_manager
from vllm.v1.worker.gpu.pcp_hidden_restore import (
    PCPMulticastHiddenStateRestorer,
    PCPMulticastUnavailableError,
)
from vllm.v1.worker.gpu.pcp_manager import PCPManager
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker


def test_manager_uses_local_rows_for_pure_decode_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pcp_manager,
        "async_copy_to_gpu",
        lambda array, *, device: torch.from_numpy(array.copy()).to(device),
    )

    class NoCollectiveGroup:
        def all_gather(self, hidden_states: torch.Tensor, dim: int) -> torch.Tensor:
            pytest.fail("Pure decode sampling must not communicate.")

    monkeypatch.setattr(pcp_manager, "get_pcp_group", NoCollectiveGroup)
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=1,
        device=torch.device("cpu"),
    )
    manager._build_batch_layout(
        num_scheduled_tokens=np.array([1, 1, 1], dtype=np.int32),
        num_computed_tokens=np.array([8, 16, 32], dtype=np.int32),
        is_prefilling=np.array([False, False, False]),
        query_start_loc_np=np.array([0, 1, 2, 3], dtype=np.int32),
    )
    manager._global_batch = SimpleNamespace(
        num_reqs=3,
        logits_indices=torch.tensor([0, 1, 2]),
    )
    hidden_states = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    )

    output = manager.restore_sample_hidden_states(hidden_states)

    torch.testing.assert_close(output, hidden_states)
    assert output.data_ptr() == hidden_states.data_ptr()
    assert manager._sample_local_row_idx is None
    assert manager._sample_restore_idx is None
    assert manager._hidden_restore_idx is None
    assert manager._hidden_restore_idx_cpu is None


def test_manager_indexes_resumed_decode_backlog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pcp_manager,
        "async_copy_to_gpu",
        lambda array, *, device: torch.from_numpy(array.copy()).to(device),
    )
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=1,
        device=torch.device("cpu"),
    )
    manager._build_batch_layout(
        num_scheduled_tokens=np.array([2, 1], dtype=np.int32),
        num_computed_tokens=np.array([8, 16], dtype=np.int32),
        is_prefilling=np.array([False, False]),
        query_start_loc_np=np.array([0, 2, 3], dtype=np.int32),
    )
    manager._global_batch = SimpleNamespace(
        num_reqs=2,
        logits_indices=torch.tensor([1, 2]),
    )
    hidden_states = torch.tensor([[10.0], [11.0], [20.0]])

    output = manager.restore_sample_hidden_states(hidden_states)

    torch.testing.assert_close(output, torch.tensor([[11.0], [20.0]]))


def test_manager_uses_dense_restore_only_when_prompt_rows_are_needed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoCompactRestore(PCPMulticastHiddenStateRestorer):
        def __init__(self) -> None:
            pass

        def restore_selected(self, *args: object, **kwargs: object) -> torch.Tensor:
            pytest.fail("Prompt logprobs must use the dense restore path.")

        def close(self) -> None:
            pass

    class FakeGroup:
        def all_gather(self, hidden_states: torch.Tensor, dim: int) -> torch.Tensor:
            assert dim == 0
            return torch.cat((hidden_states, hidden_states + 10), dim=0)

    monkeypatch.setattr(pcp_manager, "get_pcp_group", FakeGroup)
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        hidden_state_restorer=NoCompactRestore(),
    )
    manager._global_batch = SimpleNamespace(logits_indices=torch.tensor([1]))
    manager._hidden_restore_idx = torch.tensor([0, 2, 1])
    hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    restored, sampled, _ = manager.restore_for_sampling(
        hidden_states,
        needs_prompt_hidden_states=True,
    )

    torch.testing.assert_close(
        restored,
        torch.tensor([[1.0, 2.0], [11.0, 12.0], [3.0, 4.0]]),
    )
    torch.testing.assert_close(sampled, restored[[1]])


def test_manager_materializes_dense_restore_map_only_on_demand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pcp_manager,
        "async_copy_to_gpu",
        lambda array, *, device: torch.from_numpy(array.copy()).to(device),
    )

    class FakeGroup:
        def all_gather(self, hidden_states: torch.Tensor, dim: int) -> torch.Tensor:
            assert dim == 0
            torch.testing.assert_close(
                hidden_states, torch.tensor([[3.0], [0.0], [4.0]])
            )
            return torch.tensor([[3.0], [0.0], [4.0], [1.0], [2.0], [5.0]])

    monkeypatch.setattr(pcp_manager, "get_pcp_group", FakeGroup)
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
    )
    manager._build_batch_layout(
        num_scheduled_tokens=np.array([4, 2], dtype=np.int32),
        num_computed_tokens=np.array([0, 0], dtype=np.int32),
        is_prefilling=np.array([True, True]),
        query_start_loc_np=np.array([0, 4, 6], dtype=np.int32),
    )
    manager._global_batch = SimpleNamespace(
        num_tokens=6,
        is_prefilling_np=np.array([True, True]),
    )

    assert manager._hidden_restore_idx is None
    assert manager._hidden_restore_idx_cpu is None
    restored = manager.restore_full_hidden_states(torch.tensor([[3.0], [0.0], [4.0]]))

    torch.testing.assert_close(restored, torch.arange(6, dtype=torch.float32)[:, None])
    assert manager._hidden_restore_idx_cpu is not None
    torch.testing.assert_close(
        manager._hidden_restore_idx,
        torch.tensor([1, 3, 4, 0, 2, 5]),
    )


def test_manager_compacts_only_sampled_rows_for_multicast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pcp_manager,
        "async_copy_to_gpu",
        lambda array, *, device: torch.from_numpy(array.copy()).to(device),
    )

    class FakeMulticastRestorer(PCPMulticastHiddenStateRestorer):
        def __init__(self) -> None:
            self.call: (
                tuple[
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    int,
                ]
                | None
            ) = None

        def restore_selected(
            self,
            hidden_states: torch.Tensor,
            local_row_indices: torch.Tensor,
            restore_indices: torch.Tensor,
            *,
            num_selected_rows: int,
        ) -> torch.Tensor:
            self.call = (
                hidden_states,
                local_row_indices,
                restore_indices,
                num_selected_rows,
            )
            return torch.empty((num_selected_rows, hidden_states.shape[1]))

        def close(self) -> None:
            pass

    restorer = FakeMulticastRestorer()
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=1,
        device=torch.device("cpu"),
        hidden_state_restorer=restorer,
    )
    manager._build_batch_layout(
        num_scheduled_tokens=np.array([4, 2], dtype=np.int32),
        num_computed_tokens=np.array([0, 0], dtype=np.int32),
        is_prefilling=np.array([True, True]),
        query_start_loc_np=np.array([0, 4, 6], dtype=np.int32),
    )
    manager._global_batch = SimpleNamespace(logits_indices=torch.tensor([3, 5]))
    hidden_states = torch.empty((3, 7), dtype=torch.bfloat16)

    output = manager.restore_sample_hidden_states(hidden_states)

    assert output.shape == (2, 7)
    assert restorer.call is not None
    assert restorer.call[0].data_ptr() == hidden_states.data_ptr()
    torch.testing.assert_close(restorer.call[1], torch.tensor([2]))
    torch.testing.assert_close(restorer.call[2], torch.tensor([0, 1]))
    assert restorer.call[3] == 2


def test_manager_keeps_collective_for_one_row_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoMulticastRestore(PCPMulticastHiddenStateRestorer):
        def __init__(self) -> None:
            pass

        def restore_selected(self, *args: object, **kwargs: object) -> torch.Tensor:
            pytest.fail("A one-row prefill must avoid multicast packing.")

        def close(self) -> None:
            pass

    class FakeGroup:
        def all_gather(self, hidden_states: torch.Tensor, dim: int) -> torch.Tensor:
            assert dim == 0
            torch.testing.assert_close(hidden_states, torch.tensor([[1.0, 2.0]]))
            return torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

    monkeypatch.setattr(pcp_manager, "get_pcp_group", FakeGroup)
    manager = PCPManager(
        pcp_world_size=4,
        pcp_rank=0,
        device=torch.device("cpu"),
        hidden_state_restorer=NoMulticastRestore(),
    )
    manager._global_batch = SimpleNamespace(logits_indices=torch.tensor([0]))
    manager._sample_local_row_idx = torch.tensor([0])
    manager._sample_restore_idx = torch.tensor([0])

    output = manager.restore_sample_hidden_states(torch.tensor([[1.0, 2.0]]))

    torch.testing.assert_close(output, torch.tensor([[1.0, 2.0]]))


def test_manager_compacts_only_sampled_rows_for_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeGroup:
        def all_gather(self, hidden_states: torch.Tensor, dim: int) -> torch.Tensor:
            assert dim == 0
            torch.testing.assert_close(
                hidden_states,
                torch.tensor([[5.0, 6.0]]),
            )
            return torch.tensor([[1.0, 2.0], [5.0, 6.0]])

    monkeypatch.setattr(pcp_manager, "get_pcp_group", FakeGroup)
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=1,
        device=torch.device("cpu"),
    )
    manager._global_batch = SimpleNamespace(logits_indices=torch.tensor([0, 2]))
    manager._sample_local_row_idx = torch.tensor([2])
    manager._sample_restore_idx = torch.tensor([1, 0])
    hidden_states = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    )

    output = manager.restore_sample_hidden_states(hidden_states)

    torch.testing.assert_close(
        output,
        torch.tensor([[5.0, 6.0], [1.0, 2.0]]),
    )


def test_pcp4_mixed_batch_compact_collective_uses_generated_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pcp_manager,
        "async_copy_to_gpu",
        lambda array, *, device: torch.from_numpy(array.copy()).to(device),
    )
    num_scheduled_tokens = np.array([9, 1, 17, 1], dtype=np.int32)
    num_computed_tokens = np.array([0, 8, 0, 32], dtype=np.int32)
    is_prefilling = np.array([True, False, True, False])
    query_start_loc = np.array([0, 9, 10, 27, 28], dtype=np.int32)
    global_hidden_states = torch.arange(28, dtype=torch.float32).unsqueeze(1)

    managers = [
        PCPManager(
            pcp_world_size=4,
            pcp_rank=rank,
            device=torch.device("cpu"),
        )
        for rank in range(4)
    ]
    local_hidden_states = []
    for rank, manager in enumerate(managers):
        segments_by_rank, per_rank_num_tokens = manager._build_batch_layout(
            num_scheduled_tokens=num_scheduled_tokens,
            num_computed_tokens=num_computed_tokens,
            is_prefilling=is_prefilling,
            query_start_loc_np=query_start_loc,
        )
        padded_num_tokens = max(per_rank_num_tokens)
        local_hidden = torch.full((padded_num_tokens, 1), -1.0)
        for segment in segments_by_rank[rank]:
            local_hidden[segment.rank_local_batch_slice] = global_hidden_states[
                segment.global_batch_slice
            ]
        local_hidden_states.append(local_hidden)
        manager._global_batch = SimpleNamespace(
            logits_indices=torch.tensor([8, 9, 26, 27])
        )

    packed_by_rank = []
    for manager, local_hidden in zip(managers, local_hidden_states, strict=True):
        assert manager._sample_local_row_idx is not None
        packed_by_rank.append(local_hidden[manager._sample_local_row_idx])

    class FakeGroup:
        def all_gather(self, hidden_states: torch.Tensor, dim: int) -> torch.Tensor:
            assert dim == 0
            torch.testing.assert_close(hidden_states, packed_by_rank[2])
            return torch.cat(packed_by_rank, dim=0)

    monkeypatch.setattr(pcp_manager, "get_pcp_group", FakeGroup)
    output = managers[2].restore_sample_hidden_states(local_hidden_states[2])

    torch.testing.assert_close(output, global_hidden_states[[8, 9, 26, 27]])
    assert {
        int(index // packed_by_rank[0].shape[0])
        for index in managers[2]._sample_restore_idx.tolist()
    } == {0, 2, 3}


def test_manager_reuses_preallocated_sample_index_buffer() -> None:
    manager = PCPManager(
        pcp_world_size=4,
        pcp_rank=0,
        device=torch.device("cpu"),
        max_num_reqs=8,
    )
    layout = {
        "num_scheduled_tokens": np.array([9, 5], dtype=np.int32),
        "num_computed_tokens": np.array([0, 0], dtype=np.int32),
        "is_prefilling": np.array([True, True]),
        "query_start_loc_np": np.array([0, 9, 14], dtype=np.int32),
    }

    manager._build_batch_layout(**layout)
    assert manager._sample_index_buffers
    first_buffer_ptr = manager._sample_index_buffers[0].data_ptr()
    first_local_ptr = manager._sample_local_row_idx.data_ptr()

    manager._build_batch_layout(**layout)
    second_local_ptr = manager._sample_local_row_idx.data_ptr()
    manager._build_batch_layout(**layout)

    assert manager._sample_index_buffers[0].data_ptr() == first_buffer_ptr
    assert second_local_ptr != first_local_ptr
    assert manager._sample_local_row_idx.data_ptr() == first_local_ptr


def test_multicast_allocation_failure_is_coordinated_before_rendezvous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeGroup:
        group_name = "pcp"

        def size(self) -> int:
            return 4

    monkeypatch.setattr(
        pcp_hidden_restore.torch_symm_mem,
        "empty",
        lambda *args, **kwargs: torch.empty(args[0], dtype=kwargs["dtype"]),
    )

    def report_peer_allocation_failure(
        ready: torch.Tensor,
        **kwargs: object,
    ) -> None:
        ready.zero_()

    monkeypatch.setattr(
        pcp_hidden_restore.dist,
        "all_reduce",
        report_peer_allocation_failure,
    )
    monkeypatch.setattr(
        pcp_hidden_restore.torch_symm_mem,
        "rendezvous",
        lambda *args, **kwargs: pytest.fail(
            "No rank may rendezvous after a peer allocation failure."
        ),
    )

    with pytest.raises(
        PCPMulticastUnavailableError,
        match="allocation failed on at least one PCP rank",
    ):
        PCPMulticastHiddenStateRestorer(
            group=FakeGroup(),  # type: ignore[arg-type]
            device=torch.device("cpu"),
            max_num_tokens=8,
            hidden_size=16,
            dtype=torch.bfloat16,
        )


def test_multicast_rendezvous_failure_is_coordinated_before_backend_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeGroup:
        group_name = "pcp"

        def size(self) -> int:
            return 4

    monkeypatch.setattr(
        pcp_hidden_restore.torch_symm_mem,
        "empty",
        lambda *args, **kwargs: torch.empty(args[0], dtype=kwargs["dtype"]),
    )
    monkeypatch.setattr(
        pcp_hidden_restore.torch_symm_mem,
        "rendezvous",
        lambda *args, **kwargs: SimpleNamespace(multicast_ptr=1),
    )
    votes = 0

    def report_peer_rendezvous_failure(
        ready: torch.Tensor,
        **kwargs: object,
    ) -> None:
        nonlocal votes
        votes += 1
        if votes == 2:
            ready.zero_()

    monkeypatch.setattr(
        pcp_hidden_restore.dist,
        "all_reduce",
        report_peer_rendezvous_failure,
    )

    with pytest.raises(
        PCPMulticastUnavailableError,
        match="initialization failed on at least one PCP rank",
    ):
        PCPMulticastHiddenStateRestorer(
            group=FakeGroup(),  # type: ignore[arg-type]
            device=torch.device("cpu"),
            max_num_tokens=8,
            hidden_size=16,
            dtype=torch.bfloat16,
        )
    assert votes == 2


def test_compact_restorer_is_selected_automatically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=4),
        scheduler_config=SimpleNamespace(max_num_seqs=64),
        model_config=SimpleNamespace(
            get_hidden_size=lambda: 7168,
            dtype=torch.bfloat16,
        ),
    )
    monkeypatch.setattr(
        PCPManager,
        "validate_config",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        pcp_manager,
        "get_pcp_group",
        lambda: SimpleNamespace(cpu_group="pcp-group"),
    )
    monkeypatch.setattr(
        pcp_manager,
        "PCPMulticastHiddenStateRestorer",
        lambda **kwargs: sentinel,
    )

    output = pcp_manager.maybe_create_pcp_hidden_state_restorer(
        config,  # type: ignore[arg-type]
        torch.device("cpu"),
        supports_mm_inputs=False,
    )

    assert output is sentinel


def test_compact_restorer_falls_back_when_multicast_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=4),
        scheduler_config=SimpleNamespace(max_num_seqs=64),
        model_config=SimpleNamespace(
            get_hidden_size=lambda: 7168,
            dtype=torch.bfloat16,
        ),
    )
    monkeypatch.setattr(
        PCPManager,
        "validate_config",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        pcp_manager,
        "get_pcp_group",
        lambda: SimpleNamespace(cpu_group="pcp-group"),
    )

    def unavailable(**kwargs: object) -> None:
        raise PCPMulticastUnavailableError("not supported")

    monkeypatch.setattr(
        pcp_manager,
        "PCPMulticastHiddenStateRestorer",
        unavailable,
    )

    output = pcp_manager.maybe_create_pcp_hidden_state_restorer(
        config,  # type: ignore[arg-type]
        torch.device("cpu"),
        supports_mm_inputs=False,
    )

    assert output is None


def test_prompt_logprob_worker_exposes_dense_hidden_requirement() -> None:
    worker = PromptLogprobsWorker(max_num_reqs=4)
    worker.uses_prompt_logprobs[:2] = True
    worker.in_progress_prompt_logprobs["req0"] = []
    worker.in_progress_prompt_logprobs["req1"] = []
    input_batch = SimpleNamespace(
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
        SimpleNamespace(),
        np.empty(0, dtype=np.int32),
    )
