# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MoE host bank primitives.

The StagingBank tests are pure indexing arithmetic and run on CPU. The
HostBank tests need a live CUDA context (pinning does) and are skipped
without one.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.uva_bank import HostBank, StagingBank

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA context to pin memory"
)


def test_staging_bank_stage_returns_compacted_prefix():
    cold = {"w": torch.arange(40, dtype=torch.float32).reshape(8, 5)}
    bank = StagingBank({"w": (5,)}, {"w": torch.float32}, max_rows_per_step=4)
    needed = torch.tensor([6, 1, 3])

    staged = bank.stage(cold, needed, non_blocking=False)["w"]

    assert staged.shape == (3, 5)
    torch.testing.assert_close(staged, cold["w"].index_select(0, needed))


def test_staging_bank_reuses_its_buffer():
    cold = {"w": torch.arange(40, dtype=torch.float32).reshape(8, 5)}
    bank = StagingBank({"w": (5,)}, {"w": torch.float32}, max_rows_per_step=4)

    first = bank.stage(cold, torch.tensor([0, 1]), non_blocking=False)["w"]
    second = bank.stage(cold, torch.tensor([2, 3]), non_blocking=False)["w"]

    assert first.data_ptr() == second.data_ptr()


def test_staging_bank_rejects_over_budget():
    cold = {"w": torch.zeros(8, 5)}
    bank = StagingBank({"w": (5,)}, {"w": torch.float32}, max_rows_per_step=2)

    with pytest.raises(ValueError, match="exceeds max_rows_per_step"):
        bank.stage(cold, torch.tensor([0, 1, 2]), non_blocking=False)


def test_remap_points_at_staged_rows():
    # 6 global experts; 0, 3, 5 are cold, sitting in cold rows 0, 1, 2.
    cold_map = torch.tensor([0, -1, -1, 1, -1, 2])
    needed = torch.tensor([2, 0])  # stage cold rows 2 and 0, in that order

    remapped = StagingBank.remap_for_staged(cold_map, needed, num_cold_rows=3)

    # expert 5 -> cold row 2 -> staged row 0; expert 0 -> cold row 0 -> row 1
    assert remapped.tolist() == [1, -1, -1, -1, -1, 0]


def test_remap_marks_unstaged_cold_rows_as_absent():
    cold_map = torch.tensor([0, 1, 2, -1])
    needed = torch.tensor([1])

    remapped = StagingBank.remap_for_staged(cold_map, needed, num_cold_rows=3)

    assert remapped.tolist() == [-1, 0, -1, -1]


def test_remap_infers_row_count_when_not_given():
    cold_map = torch.tensor([0, -1, 1])
    needed = torch.tensor([0, 1])

    inferred = StagingBank.remap_for_staged(cold_map, needed)
    explicit = StagingBank.remap_for_staged(cold_map, needed, num_cold_rows=2)

    assert inferred.tolist() == explicit.tolist() == [0, -1, 1]


def test_remap_preserves_dtype_and_shape():
    cold_map = torch.tensor([0, -1, 1], dtype=torch.int32)
    remapped = StagingBank.remap_for_staged(
        cold_map, torch.tensor([0, 1]), num_cold_rows=2
    )
    assert remapped.dtype == torch.int32
    assert remapped.shape == cold_map.shape


@requires_cuda
def test_pin_is_idempotent():
    already = torch.arange(8, dtype=torch.float32).pin_memory()
    assert HostBank.pin(already) is already


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_device_view_matches_source(dtype):
    src = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).to(dtype)

    view = HostBank.as_device_tensor(src)

    assert view.is_cuda
    assert view.shape == src.shape
    assert view.dtype == dtype
    torch.testing.assert_close(view.cpu(), src)


@requires_cuda
def test_device_view_allocates_no_device_memory():
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()

    view = HostBank.as_device_tensor(torch.zeros(1024, 1024, dtype=torch.float32))
    torch.cuda.synchronize()

    assert torch.cuda.memory_allocated() == before
    assert view.numel() == 1024 * 1024


@requires_cuda
def test_device_view_keeps_host_buffer_alive():
    view = HostBank.as_device_tensor(torch.arange(4, dtype=torch.float32))
    assert view._vllm_uva_host_keepalive.is_pinned()
