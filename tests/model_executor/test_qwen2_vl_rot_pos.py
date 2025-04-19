# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm.model_executor.models.qwen2_vl import Qwen2VLViTRotaryPosGenerator


@pytest.mark.parametrize("spatial_merge_size", [2, 3])
@pytest.mark.parametrize("impl_name, device", [
    ("generate_by_torch_fused", "cuda"),
    ("generate_by_numba", "cpu"),
])
def test_qwen2_vl_rot_pos_correctness(
    spatial_merge_size,
    impl_name,
    device,
):
    rot_pos_generator = Qwen2VLViTRotaryPosGenerator(
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
                    groundtruth = rot_pos_generator.generate_by_torch(grid_thw)
                    testing_impl = getattr(rot_pos_generator, impl_name)
                    actual = testing_impl(grid_thw)

                    assert actual.device.type == device, \
                        get_assertion_msg(grid_thw)
                    assert groundtruth.dtype == actual.dtype, \
                        get_assertion_msg(grid_thw)
                    assert groundtruth.shape == actual.shape, \
                        get_assertion_msg(grid_thw)

                    actual = actual.cpu()
                    assert torch.equal(groundtruth, actual), \
                        get_assertion_msg(grid_thw)
