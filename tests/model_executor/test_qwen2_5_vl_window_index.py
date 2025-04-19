# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VisionAttentionScheduler)


@pytest.mark.parametrize("window_size, patch_size, spatial_merge_size", [
    (112, 14, 2),
    (128, 16, 2),
])
def test_qwen2_5_vl_get_window_indices_correctness(window_size, patch_size,
                                                   spatial_merge_size):
    scheduler = Qwen2_5_VisionAttentionScheduler(
        spatial_merge_size=spatial_merge_size,
        window_size=window_size,
        patch_size=patch_size,
        max_position_embeddings=32768,
        device=torch.device("cpu"),
    )

    get_assertion_msg = lambda grid_thw: f"mismatch at grid_thw={grid_thw}"

    for t in range(1, 3):
        for h in range(1, 50):
            for w in range(1, 50):
                grid_thw = torch.tensor(
                    [[t, h * spatial_merge_size, w * spatial_merge_size]],
                    dtype=torch.int64,
                )

                (
                    window_indices_torch,
                    reverse_indices_torch,
                    seqlens_full_torch,
                    seqlens_window_torch,
                    cu_seqlens_full_torch,
                    cu_seqlens_window_torch,
                ) = scheduler.generate_by_torch(grid_thw)

                (
                    window_indices_numba,
                    reverse_indices_numba,
                    seqlens_full_numba,
                    seqlens_window_numba,
                    cu_seqlens_full_numba,
                    cu_seqlens_window_numba,
                ) = scheduler.generate_by_torch_with_numba(grid_thw)

                assert window_indices_torch.dtype == \
                    window_indices_numba.dtype, get_assertion_msg(grid_thw)
                assert reverse_indices_torch.dtype == \
                    reverse_indices_numba.dtype, get_assertion_msg(grid_thw)
                assert seqlens_full_torch.dtype == \
                    seqlens_full_numba.dtype, get_assertion_msg(grid_thw)
                assert seqlens_window_torch.dtype == \
                    seqlens_window_numba.dtype, get_assertion_msg(grid_thw)
                assert cu_seqlens_full_torch.dtype == \
                    cu_seqlens_full_numba.dtype, get_assertion_msg(grid_thw)
                assert cu_seqlens_window_torch.dtype == \
                    cu_seqlens_window_numba.dtype, get_assertion_msg(grid_thw)

                assert window_indices_torch.shape == \
                    window_indices_numba.shape, get_assertion_msg(grid_thw)
                assert reverse_indices_torch.shape == \
                    reverse_indices_numba.shape, get_assertion_msg(grid_thw)
                assert seqlens_full_torch.shape == \
                    seqlens_full_numba.shape, get_assertion_msg(grid_thw)
                assert seqlens_window_torch.shape == \
                    seqlens_window_numba.shape, get_assertion_msg(grid_thw)
                assert cu_seqlens_full_torch.shape == \
                    cu_seqlens_full_numba.shape, get_assertion_msg(grid_thw)
                assert cu_seqlens_window_torch.shape == \
                    cu_seqlens_window_numba.shape, get_assertion_msg(grid_thw)

                assert torch.equal(window_indices_torch,
                                   window_indices_numba), \
                       get_assertion_msg(grid_thw)
                assert torch.equal(reverse_indices_torch,
                                   reverse_indices_numba), \
                       get_assertion_msg(grid_thw)
                assert torch.equal(seqlens_full_torch,
                                   seqlens_full_numba), \
                       get_assertion_msg(grid_thw)
                assert torch.equal(seqlens_window_torch,
                                   seqlens_window_numba), \
                       get_assertion_msg(grid_thw)
                assert torch.equal(cu_seqlens_full_torch,
                                   cu_seqlens_full_numba), \
                       get_assertion_msg(grid_thw)
                assert torch.equal(cu_seqlens_window_torch,
                                    cu_seqlens_window_numba), \
                       get_assertion_msg(grid_thw)


def _grid_thw_generator(t_range, h_range, w_range, spatial_merge_size):
    for t in t_range:
        for h in h_range:
            for w in w_range:
                yield torch.tensor(
                    [[t, h * spatial_merge_size, w * spatial_merge_size]],
                    dtype=torch.int64,
                )


@pytest.mark.parametrize("window_size, patch_size, spatial_merge_size", [
    (112, 14, 2),
    (128, 16, 2),
])
def test_qwen2_5_vl_get_window_indices_multi_items_correctness(
        window_size, patch_size, spatial_merge_size):
    scheduler = Qwen2_5_VisionAttentionScheduler(
        spatial_merge_size=spatial_merge_size,
        window_size=window_size,
        patch_size=patch_size,
        max_position_embeddings=32768,
        device=torch.device("cpu"),
    )

    get_assertion_msg = lambda grid_thw: f"mismatch at grid_thw={grid_thw}"

    for grid_thw1 in _grid_thw_generator(
            range(1, 3),
            range(1, 18, 3),
            range(1, 18, 3),
            spatial_merge_size,
    ):
        for grid_thw2 in _grid_thw_generator(
                range(1, 3),
                range(1, 18, 3),
                range(1, 18, 3),
                spatial_merge_size,
        ):
            grid_thw = torch.cat([grid_thw1, grid_thw2])

            (
                window_indices_torch,
                reverse_indices_torch,
                seqlens_full_torch,
                seqlens_window_torch,
                cu_seqlens_full_torch,
                cu_seqlens_window_torch,
            ) = scheduler.generate_by_torch(grid_thw)

            (
                window_indices_numba,
                reverse_indices_numba,
                seqlens_full_numba,
                seqlens_window_numba,
                cu_seqlens_full_numba,
                cu_seqlens_window_numba,
            ) = scheduler.generate_by_torch_with_numba(grid_thw)

            assert window_indices_torch.dtype == \
                window_indices_numba.dtype, get_assertion_msg(grid_thw)
            assert reverse_indices_torch.dtype == \
                reverse_indices_numba.dtype, get_assertion_msg(grid_thw)
            assert seqlens_full_torch.dtype == \
                seqlens_full_numba.dtype, get_assertion_msg(grid_thw)
            assert seqlens_window_torch.dtype == \
                seqlens_window_numba.dtype, get_assertion_msg(grid_thw)
            assert cu_seqlens_full_torch.dtype == \
                cu_seqlens_full_numba.dtype, get_assertion_msg(grid_thw)
            assert cu_seqlens_window_torch.dtype == \
                cu_seqlens_window_numba.dtype, get_assertion_msg(grid_thw)

            assert window_indices_torch.shape == \
                window_indices_numba.shape, get_assertion_msg(grid_thw)
            assert reverse_indices_torch.shape == \
                reverse_indices_numba.shape, get_assertion_msg(grid_thw)
            assert seqlens_full_torch.shape == \
                seqlens_full_numba.shape, get_assertion_msg(grid_thw)
            assert seqlens_window_torch.shape == \
                seqlens_window_numba.shape, get_assertion_msg(grid_thw)
            assert cu_seqlens_full_torch.shape == \
                cu_seqlens_full_numba.shape, get_assertion_msg(grid_thw)
            assert cu_seqlens_window_torch.shape == \
                cu_seqlens_window_numba.shape, get_assertion_msg(grid_thw)

            assert torch.equal(window_indices_torch,
                                window_indices_numba), \
                    get_assertion_msg(grid_thw)
            assert torch.equal(reverse_indices_torch,
                                reverse_indices_numba), \
                    get_assertion_msg(grid_thw)
            assert torch.equal(seqlens_full_torch,
                                seqlens_full_numba), \
                    get_assertion_msg(grid_thw)
            assert torch.equal(seqlens_window_torch,
                                seqlens_window_numba), \
                    get_assertion_msg(grid_thw)
            assert torch.equal(cu_seqlens_full_torch,
                                cu_seqlens_full_numba), \
                    get_assertion_msg(grid_thw)
            assert torch.equal(cu_seqlens_window_torch,
                                cu_seqlens_window_numba), \
                    get_assertion_msg(grid_thw)
