# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.v1.worker.gpu.dp_utils as dp_utils
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.spec_decode.eagle import speculator as eagle_speculator


def _make_manager(
    candidates: list[BatchExecutionDescriptor],
) -> CudaGraphManager:
    manager = object.__new__(CudaGraphManager)
    manager._graphs_captured = True
    max_num_tokens = max(desc.num_tokens for desc in candidates)
    manager._candidates = [[] for _ in range(max_num_tokens + 1)]
    manager._candidates[max_num_tokens] = candidates
    return manager


def _live_mm_inputs() -> tuple[list[torch.Tensor], torch.Tensor]:
    return [torch.empty(1, 1)], torch.ones(1, dtype=torch.bool)


def test_eagle_prefill_text_only_batch_keeps_full_cudagraph_enabled():
    full_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=4,
        num_reqs=1,
        uniform_token_count=4,
    )
    piecewise_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE,
        num_tokens=4,
        num_reqs=None,
    )
    manager = _make_manager([full_desc, piecewise_desc])

    invalid_modes = eagle_speculator._get_prefill_invalid_cudagraph_modes(None)

    assert invalid_modes is None
    assert manager.dispatch(1, 4, 4, invalid_modes=invalid_modes) == full_desc


def test_eagle_prefill_live_mm_batch_skips_full_and_uses_piecewise():
    full_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=4,
        num_reqs=1,
        uniform_token_count=4,
    )
    piecewise_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE,
        num_tokens=4,
        num_reqs=None,
    )
    manager = _make_manager([full_desc, piecewise_desc])

    invalid_modes = eagle_speculator._get_prefill_invalid_cudagraph_modes(
        _live_mm_inputs()
    )

    assert invalid_modes == {CUDAGraphMode.FULL}
    assert (
        manager.dispatch(
            1,
            4,
            4,
            invalid_modes=invalid_modes,
        )
        == piecewise_desc
    )


def test_eagle_prefill_live_mm_batch_falls_back_to_none_without_piecewise():
    full_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=4,
        num_reqs=1,
        uniform_token_count=4,
    )
    manager = _make_manager([full_desc])

    invalid_modes = eagle_speculator._get_prefill_invalid_cudagraph_modes(
        _live_mm_inputs()
    )
    desc = manager.dispatch(
        1,
        4,
        4,
        invalid_modes=invalid_modes,
    )

    assert invalid_modes == {CUDAGraphMode.FULL}
    assert desc == BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE,
        num_tokens=4,
        num_reqs=1,
    )


def test_eagle_prefill_invalid_modes_only_for_live_mm_inputs():
    assert eagle_speculator._get_prefill_invalid_cudagraph_modes(None) is None
    assert (
        eagle_speculator._get_prefill_invalid_cudagraph_modes(
            ([], torch.zeros(0, dtype=torch.bool))
        )
        is None
    )

    assert eagle_speculator._get_prefill_invalid_cudagraph_modes(_live_mm_inputs()) == {
        CUDAGraphMode.FULL
    }


def test_dispatch_cg_and_sync_dp_forwards_invalid_modes():
    class RecordingManager:
        invalid_modes = None

        def dispatch(
            self,
            num_reqs: int,
            num_tokens: int,
            uniform_token_count: int | None,
            invalid_modes: set[CUDAGraphMode] | None = None,
        ) -> BatchExecutionDescriptor:
            self.invalid_modes = invalid_modes
            return BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
            )

    manager = RecordingManager()
    desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
        manager,  # type: ignore[arg-type]
        num_reqs=1,
        num_tokens=4,
        uniform_token_count=4,
        dp_size=1,
        dp_rank=0,
        invalid_modes={CUDAGraphMode.FULL},
    )

    assert manager.invalid_modes == {CUDAGraphMode.FULL}
    assert desc.cg_mode == CUDAGraphMode.NONE
    assert num_tokens_across_dp is None


def test_dp_sync_does_not_redispatch_higher_mode(monkeypatch):
    full_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=4,
        num_reqs=1,
        uniform_token_count=4,
    )
    piecewise_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE,
        num_tokens=4,
        num_reqs=None,
    )
    manager = _make_manager([full_desc, piecewise_desc])

    class FakeDpGroup:
        cpu_group = object()

    def fake_all_reduce(tensor: torch.Tensor, group: object) -> None:
        del group
        tensor[:, :] = torch.tensor(
            [
                [4, 4],
                [CUDAGraphMode.PIECEWISE.value, CUDAGraphMode.FULL.value],
                [4, 4],
            ],
            dtype=tensor.dtype,
            device=tensor.device,
        )

    monkeypatch.setattr(dp_utils, "get_dp_group", lambda: FakeDpGroup())
    monkeypatch.setattr(dp_utils.dist, "all_reduce", fake_all_reduce)

    desc, _ = dp_utils.sync_cudagraph_and_dp_padding(
        manager,
        full_desc,
        num_tokens=4,
        num_reqs=1,
        uniform_token_count=4,
        dp_size=2,
        dp_rank=1,
    )

    assert desc == piecewise_desc
