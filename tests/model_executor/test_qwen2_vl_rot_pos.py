# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm.model_executor.models.qwen2_vl import Qwen2VLRotPosGenerator


@pytest.mark.parametrize("spatial_merge_size", [2, 3])
@pytest.mark.parametrize("impl_name, device", [
    ("generate_torch_fused", "cuda"),
    ("generate_numba", "cpu"),
])
def test_qwen2_vl_rot_pos_correctness(
    spatial_merge_size,
    impl_name,
    device,
):
    rot_pos_generator = Qwen2VLRotPosGenerator(
        spatial_merge_size=spatial_merge_size,
        max_position_embeddings=32768,
        device=torch.device(device),
    )

    get_assertion_msg = lambda grid_thw: f"mismatch at grid_thw={grid_thw}"

    for t in range(1, 3):
        for h in range(1, 32):
            for w in range(1, 32):
                for grid_thw in [
                        torch.tensor([
                            [
                                t, h * spatial_merge_size,
                                w * spatial_merge_size
                            ],
                        ],
                                     dtype=torch.int64),
                        torch.tensor([
                            [
                                t, h * spatial_merge_size,
                                w * spatial_merge_size
                            ],
                        ] * 2,
                                     dtype=torch.int64),
                        torch.tensor([
                            [
                                t, h * spatial_merge_size,
                                w * spatial_merge_size
                            ],
                            [
                                1, 2 * spatial_merge_size,
                                2 * spatial_merge_size
                            ],
                        ],
                                     dtype=torch.int64),
                ]:

                    rot_pos_torch = rot_pos_generator.generate_torch(grid_thw)
                    accel_impl = getattr(rot_pos_generator, impl_name)
                    rot_pos_accel = accel_impl(grid_thw)

                    assert rot_pos_accel.device.type == device, \
                        get_assertion_msg(grid_thw)
                    assert rot_pos_torch.dtype == rot_pos_accel.dtype, \
                        get_assertion_msg(grid_thw)
                    assert rot_pos_torch.shape == rot_pos_accel.shape, \
                        get_assertion_msg(grid_thw)

                    rot_pos_accel = rot_pos_accel.cpu()
                    assert torch.equal(rot_pos_torch, rot_pos_accel), \
                        get_assertion_msg(grid_thw)
