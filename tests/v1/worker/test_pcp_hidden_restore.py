# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu.pcp_hidden_restore as hidden_restore
import vllm.v1.worker.gpu.pcp_manager as pcp_manager
from vllm.v1.worker.gpu.pcp_hidden_restore import (
    PCPHiddenStateRestorer,
    direct_hidden_state_restore,
)
from vllm.v1.worker.gpu.pcp_manager import PCPManager


def _valid_inputs(
    *,
    num_rows: int = 2,
    hidden_size: int = 4,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty((num_rows, hidden_size), dtype=dtype),
        torch.empty((2, 8, hidden_size), dtype=dtype),
        torch.arange(num_rows, dtype=torch.int64),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_direct_hidden_restore_accepts_empty_supported_inputs(
    dtype: torch.dtype,
) -> None:
    direct_hidden_state_restore(*_valid_inputs(num_rows=0, dtype=dtype))


@pytest.mark.parametrize(
    "replacement,error",
    [
        ({"hidden_states": torch.empty((2, 2, 2))}, "expects a 2D input"),
        ({"peer_output": torch.empty((2, 8))}, "shape \\[peer, token, hidden\\]"),
        (
            {"hidden_states": torch.empty((2, 4), dtype=torch.float32)},
            "supports BF16 or FP16",
        ),
        (
            {"peer_output": torch.empty((2, 8, 4), dtype=torch.float16)},
            "dtypes must match",
        ),
        (
            {"peer_output": torch.empty((2, 8, 3), dtype=torch.bfloat16)},
            "hidden sizes do not match",
        ),
        (
            {"global_row_indices": torch.empty(3, dtype=torch.int64)},
            "indices must match the local row count",
        ),
        (
            {"global_row_indices": torch.empty(2, dtype=torch.float32)},
            "indices must use int32 or int64",
        ),
    ],
)
def test_direct_hidden_restore_rejects_invalid_metadata(
    replacement: dict[str, torch.Tensor],
    error: str,
) -> None:
    names = ("hidden_states", "peer_output", "global_row_indices")
    inputs = dict(zip(names, _valid_inputs(), strict=True))
    inputs.update(replacement)

    with pytest.raises(ValueError, match=error):
        direct_hidden_state_restore(**inputs)


def test_direct_hidden_restore_requires_colocated_tensors() -> None:
    hidden_states, peer_output, global_rows = _valid_inputs()

    with pytest.raises(ValueError, match="input and peer output must share a device"):
        direct_hidden_state_restore(
            hidden_states,
            peer_output.to("meta"),
            global_rows,
        )
    with pytest.raises(ValueError, match="indices must be on"):
        direct_hidden_state_restore(
            hidden_states,
            peer_output,
            global_rows.to("meta"),
        )


@pytest.mark.parametrize(
    "max_num_tokens,hidden_size,dtype,error",
    [
        (0, 8, torch.bfloat16, "dimensions must be positive"),
        (8, 0, torch.bfloat16, "dimensions must be positive"),
        (8, 8, torch.float32, "supports BF16 or FP16"),
    ],
)
def test_restorer_rejects_unsupported_allocation(
    max_num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        PCPHiddenStateRestorer(
            group=None,  # type: ignore[arg-type]
            device=torch.device("cpu"),
            max_num_tokens=max_num_tokens,
            hidden_size=hidden_size,
            dtype=dtype,
        )


def test_restorer_maps_rows_and_fences_reused_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    peer_output = torch.full((2, 2, 6, 3), -1, dtype=torch.float16)
    allocation = SimpleNamespace(
        local_view=peer_output[0],
        close=lambda: events.append("allocation_close"),
    )

    class FakeFence:
        def __init__(self, group: object, device: torch.device) -> None:
            assert group == "pcp-group"
            assert device == torch.device("cpu")

        def __call__(self) -> None:
            events.append("fence")

        def close(self) -> None:
            events.append("fence_close")

    def fake_create_peer_view(
        shape: tuple[int, int],
        *,
        dtype: torch.dtype,
        group: object,
        require_native_atomics: bool,
        device: torch.device,
    ) -> object:
        assert shape == (2, 6, 3)
        assert dtype == torch.float16
        assert group == "pcp-group"
        assert not require_native_atomics
        assert device == torch.device("cpu")
        return allocation

    def fake_direct_restore(
        hidden_states: torch.Tensor,
        outputs: torch.Tensor,
        global_row_indices: torch.Tensor,
    ) -> None:
        events.append("store")
        for local_row, global_row in enumerate(global_row_indices.tolist()):
            if 0 <= global_row < outputs.shape[1]:
                outputs[:, global_row] = hidden_states[local_row]

    monkeypatch.setattr(
        hidden_restore, "create_rank_major_peer_view", fake_create_peer_view
    )
    monkeypatch.setattr(
        hidden_restore,
        "make_rank_major_tensor_view",
        lambda actual_allocation, local: (
            peer_output
            if actual_allocation is allocation
            and local.data_ptr() == allocation.local_view.data_ptr()
            else pytest.fail("unexpected peer-view inputs")
        ),
    )
    monkeypatch.setattr(hidden_restore, "PeerMemoryFence", FakeFence)
    monkeypatch.setattr(
        hidden_restore, "direct_hidden_state_restore", fake_direct_restore
    )

    restorer = PCPHiddenStateRestorer(
        group="pcp-group",  # type: ignore[arg-type]
        device=torch.device("cpu"),
        max_num_tokens=6,
        hidden_size=3,
        dtype=torch.float16,
    )
    assert restorer.local_output.data_ptr() == peer_output[0].data_ptr()
    assert restorer.peer_output.data_ptr() == peer_output[:, 0].data_ptr()

    first_hidden = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float16)
    first = restorer.restore(
        first_hidden,
        torch.tensor([2, 0, -1]),
        num_global_tokens=4,
    )
    assert first.shape == (4, 3)
    torch.testing.assert_close(first[0], first_hidden[1])
    torch.testing.assert_close(first[2], first_hidden[0])
    torch.testing.assert_close(peer_output[1, 0, 0], first_hidden[1])
    torch.testing.assert_close(peer_output[1, 0, 2], first_hidden[0])
    assert events == ["store", "fence"]

    second_hidden = torch.tensor([[10, 11, 12]], dtype=torch.float16)
    second = restorer.restore(
        second_hidden,
        torch.tensor([1], dtype=torch.int32),
        num_global_tokens=2,
    )
    torch.testing.assert_close(second[1], second_hidden[0])
    assert events == ["store", "fence", "store", "fence"]

    restorer.close()
    restorer.close()
    assert events[-2:] == ["fence_close", "allocation_close"]
    with pytest.raises(RuntimeError, match="is closed"):
        restorer.restore(
            second_hidden,
            torch.tensor([1]),
            num_global_tokens=2,
        )
    with pytest.raises(RuntimeError, match="is closed"):
        _ = restorer.peer_output


def test_restorer_rejects_output_beyond_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer_output = torch.empty((2, 2, 3, 2), dtype=torch.bfloat16)
    allocation = SimpleNamespace(local_view=peer_output[0], close=lambda: None)
    monkeypatch.setattr(
        hidden_restore,
        "create_rank_major_peer_view",
        lambda *args, **kwargs: allocation,
    )
    monkeypatch.setattr(
        hidden_restore,
        "make_rank_major_tensor_view",
        lambda *args, **kwargs: peer_output,
    )
    monkeypatch.setattr(
        hidden_restore,
        "PeerMemoryFence",
        lambda *args, **kwargs: SimpleNamespace(close=lambda: None),
    )
    restorer = PCPHiddenStateRestorer(
        group=None,  # type: ignore[arg-type]
        device=torch.device("cpu"),
        max_num_tokens=3,
        hidden_size=2,
        dtype=torch.bfloat16,
    )

    with pytest.raises(ValueError, match="exceeds the direct output capacity"):
        restorer.restore(
            torch.empty((0, 2), dtype=torch.bfloat16),
            torch.empty(0, dtype=torch.int64),
            num_global_tokens=4,
        )


def test_manager_builds_exactly_one_hidden_publisher_per_global_row(
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
        hidden_state_restorer=object(),  # type: ignore[arg-type]
    )
    manager._build_batch_layout(
        num_scheduled_tokens=np.array([4, 1], dtype=np.int32),
        num_computed_tokens=np.array([0, 4], dtype=np.int32),
        is_prefilling=np.array([True, False]),
        query_start_loc_np=np.array([0, 4, 5], dtype=np.int32),
    )

    assert manager._hidden_publish_idx is not None
    torch.testing.assert_close(
        manager._hidden_publish_idx,
        torch.tensor([3, 4, 0, 1, 2, -1]),
    )
    published = manager._hidden_publish_idx[manager._hidden_publish_idx >= 0]
    torch.testing.assert_close(torch.sort(published).values, torch.arange(5))


def test_collective_manager_does_not_build_direct_publish_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pcp_manager,
        "async_copy_to_gpu",
        lambda array, *, device: torch.from_numpy(array.copy()).to(device),
    )
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
    )
    manager._build_batch_layout(
        num_scheduled_tokens=np.array([4], dtype=np.int32),
        num_computed_tokens=np.array([0], dtype=np.int32),
        is_prefilling=np.array([True]),
        query_start_loc_np=np.array([0, 4], dtype=np.int32),
    )

    assert manager._hidden_publish_idx is None


def test_manager_routes_only_rank_local_publish_map() -> None:
    class FakeRestorer:
        def __init__(self) -> None:
            self.call: tuple[torch.Tensor, torch.Tensor, int] | None = None

        def restore(
            self,
            hidden_states: torch.Tensor,
            global_row_indices: torch.Tensor,
            *,
            num_global_tokens: int,
        ) -> torch.Tensor:
            self.call = (
                hidden_states,
                global_row_indices,
                num_global_tokens,
            )
            return torch.empty((num_global_tokens, hidden_states.shape[1]))

        def close(self) -> None:
            pass

    restorer = FakeRestorer()
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=1,
        device=torch.device("cpu"),
        hidden_state_restorer=restorer,  # type: ignore[arg-type]
    )
    manager._hidden_restore_idx = torch.arange(5)
    manager._hidden_publish_idx = torch.tensor([3, 4, 0, 1, 2, -1])
    hidden_states = torch.empty((3, 7), dtype=torch.bfloat16)

    output = manager.restore_hidden_states(hidden_states)

    assert output.shape == (5, 7)
    assert restorer.call is not None
    assert restorer.call[0].data_ptr() == hidden_states.data_ptr()
    torch.testing.assert_close(restorer.call[1], torch.tensor([1, 2, -1]))
    assert restorer.call[2] == 5
