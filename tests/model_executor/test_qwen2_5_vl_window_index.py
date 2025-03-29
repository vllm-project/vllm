from unittest.mock import Mock

import torch
import torch.nn.functional as F
import pytest

from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VisionTransformer

@pytest.mark.parametrize("window_size, patch_size, spatial_merge_size", [
    (112, 14, 2),
    (128, 16, 2),
])
def test_qwen2_5_vl_get_window_indices_correctness(dist_init, window_size, patch_size, spatial_merge_size):
    vision_config = Mock(**{
        "depth": 32,
        "hidden_act": "silu",
        "hidden_size": 1280,
        "intermediate_size": 3420,
        "num_heads": 16,
        "in_channels": 3,
        "out_hidden_size": 3584,
        "patch_size": patch_size,
        "spatial_merge_size": spatial_merge_size,
        "spatial_patch_size": patch_size,
        "window_size": window_size,
        "fullatt_block_indexes": [
            7,
            15,
            23,
            31
        ],
        "tokens_per_second": 2,
        "temporal_patch_size": 2
    })

    vit = Qwen2_5_VisionTransformer(
        vision_config=vision_config
    )

    for t in range(1, 3):
        for h in range(1, 100):
            for w in range(1, 100):
                grid_thw = torch.tensor(
                    [[t, h * spatial_merge_size, w * spatial_merge_size]],
                    dtype=torch.int64,
                )
                
                # torch impl
                def torch_impl():
                    # windows attention
                    window_indices, cu_seqlens_window = vit.get_window_index_torch(grid_thw)
                    cu_seqlens_window = torch.tensor(
                        cu_seqlens_window,
                        dtype=torch.int32)
                    cu_seqlens_window = torch.unique_consecutive(cu_seqlens_window)

                    # compute cu_seqlens
                    seqlens_full = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                                         grid_thw[:, 0])
                    cu_seqlens_full = seqlens_full.cumsum(dim=0, dtype=torch.int32)
                    cu_seqlens_full = F.pad(cu_seqlens_full, (1, 0), "constant", 0)

                    seqlens_window = cu_seqlens_window[1:] - cu_seqlens_window[:-1]
                    seqlens_window = seqlens_window.to(torch.int64)
                    
                    # adapter
                    reverse_indices = torch.argsort(window_indices)

                    return (
                        window_indices, reverse_indices,
                        seqlens_full, seqlens_window, 
                        cu_seqlens_full, cu_seqlens_window,
                    )

                # numba impl
                def numba_impl():
                    return vit.get_window_index_and_seqlens(grid_thw, "cpu")
                
                (
                    window_indices_torch, reverse_indices_torch,
                    seqlens_full_torch, seqlens_window_torch,
                    cu_seqlens_full_torch, cu_seqlens_window_torch,
                ) = torch_impl()

                (
                    window_indices_numba, reverse_indices_numba,
                    seqlens_full_numba, seqlens_window_numba, 
                    cu_seqlens_full_numba, cu_seqlens_window_numba,
                ) = numba_impl()

                get_assertion_msg = lambda: f"mismatch at grid_thw={grid_thw}"

                assert window_indices_torch.dtype == window_indices_numba.dtype, get_assertion_msg()
                assert reverse_indices_torch.dtype == reverse_indices_numba.dtype, get_assertion_msg()
                assert seqlens_full_torch.dtype == seqlens_full_numba.dtype, get_assertion_msg()
                assert seqlens_window_torch.dtype == seqlens_window_numba.dtype, get_assertion_msg()
                assert cu_seqlens_full_torch.dtype == cu_seqlens_full_numba.dtype, get_assertion_msg()
                assert cu_seqlens_window_torch.dtype == cu_seqlens_window_numba.dtype, get_assertion_msg()

                assert window_indices_torch.shape == window_indices_numba.shape, get_assertion_msg()
                assert reverse_indices_torch.shape == reverse_indices_numba.shape, get_assertion_msg()
                assert seqlens_full_torch.shape == seqlens_full_numba.shape, get_assertion_msg()
                assert seqlens_window_torch.shape == seqlens_window_numba.shape, get_assertion_msg()
                assert cu_seqlens_full_torch.shape == cu_seqlens_full_numba.shape, get_assertion_msg()
                assert cu_seqlens_window_torch.shape == cu_seqlens_window_numba.shape, get_assertion_msg()

                assert torch.equal(window_indices_torch, window_indices_numba), get_assertion_msg()
                assert torch.equal(reverse_indices_torch, reverse_indices_numba), get_assertion_msg()
                assert torch.equal(seqlens_full_torch, seqlens_full_numba), get_assertion_msg()
                assert torch.equal(seqlens_window_torch, seqlens_window_numba), get_assertion_msg()
                assert torch.equal(cu_seqlens_full_torch, cu_seqlens_full_numba), get_assertion_msg()
                assert torch.equal(cu_seqlens_window_torch, cu_seqlens_window_numba), get_assertion_msg()
