# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HYV4 VL implements `--mm-encoder-tp-mode data` end to end: the wrapper
advertises the capability, and the eager encoder path shards whole media
items across the TP group and restores per-item embeddings in the original
order on every rank."""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

from tests.utils import ensure_current_vllm_config, multi_gpu_test
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.models.interfaces import (
    supports_multimodal_encoder_tp_data,
)
from vllm.models.hy_v4.common.vision import run_dp_sharded_hyv4_vision_tower
from vllm.platforms import current_platform
from vllm.transformers_utils.processors.hy_v4 import vision_token_count
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.cpu_test

OUT_HIDDEN = 32


def _vision_config() -> SimpleNamespace:
    # The vision-config fields that vision_token_count reads.
    return SimpleNamespace(
        out_hidden_size=OUT_HIDDEN,
        spatial_merge_size=2,
        spatial_patch_size=1,
        temporal_patch_size=2,
        cat_extra_token=1,
        use_image_newline_token=True,
    )


class ItemTaggingVisionTower(torch.nn.Module):
    """Stand-in for `HYV4VisionTransformer` that tags rows by item content.

    Emits `vision_token_count` rows per item, so variable output lengths,
    video temporal pooling and extra/newline tokens are all exercised. Rows
    are tagged with the mean of the item's input patches, so the test also
    catches a rank encoding the wrong item's pixels, not just a misordered
    gather. The tower's parameter dtype (bf16, what `dtype` reports)
    intentionally differs from its input data dtype (fp32): empty ranks must
    all-gather placeholders in the parameter dtype, not the pixel dtype, or
    the collective hangs.
    """

    def __init__(self, modality: str = "image"):
        super().__init__()
        self.config = _vision_config()
        self.modality = modality
        # Gives `.device` something to report on empty ranks.
        self._anchor = torch.nn.Parameter(torch.zeros(1))

    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16

    @property
    def device(self) -> torch.device:
        return self._anchor.device

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: list[list[int]]
    ) -> list[torch.Tensor]:
        out = []
        offset = 0
        for grid in grid_thw:
            patches = math.prod(grid)
            tag = pixel_values[offset : offset + patches].mean().item()
            offset += patches
            rows = vision_token_count(grid, self.config, self.modality)
            out.append(
                torch.full(
                    (rows, OUT_HIDDEN),
                    tag,
                    device=pixel_values.device,
                    dtype=torch.bfloat16,
                )
            )
        return out


def test_models_advertise_encoder_tp_data():
    """Without the flag, `--mm-encoder-tp-mode data` falls back to weights."""
    from vllm.models.hy_v4.common.vl_model import HYV4VLForConditionalGeneration
    from vllm.models.hy_v4_flash.common.vl_model import (
        HYV4FlashVLForConditionalGeneration,
    )

    assert supports_multimodal_encoder_tp_data(HYV4VLForConditionalGeneration)
    assert supports_multimodal_encoder_tp_data(HYV4FlashVLForConditionalGeneration)


def _check_sharded_tower(modality: str, grid_thw: list[list[int]]) -> None:
    """Every rank gets every item's embeddings, in the original order."""
    config = _vision_config()
    tower = ItemTaggingVisionTower(modality)
    # Item i's patches all carry the value i + 1; the tower tags its rows
    # with that value, so content and order are both checked.
    pixel_values = torch.cat(
        [torch.full((math.prod(g), 768), float(i + 1)) for i, g in enumerate(grid_thw)]
    )

    with torch.inference_mode():
        embeds = run_dp_sharded_hyv4_vision_tower(
            tower, pixel_values, grid_thw, modality
        )

    expected_rows = [vision_token_count(g, config, modality) for g in grid_thw]
    assert [e.shape[0] for e in embeds] == expected_rows
    for item, embed in enumerate(embeds):
        assert embed.dtype == torch.bfloat16, "empty-rank placeholder dtype"
        assert torch.all(embed == item + 1), f"item {item} came back misordered"


def _run_worker(local_rank, world_size, master_port, cases):
    set_random_seed(0)
    device = f"{current_platform.device_type}:{local_rank}"
    torch.accelerator.set_device_index(device)
    torch.set_default_device(device)

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(master_port),
        }
    )
    init_distributed_environment()
    with ensure_current_vllm_config():
        initialize_model_parallel(tensor_model_parallel_size=world_size)

    assert get_tensor_model_parallel_world_size() == world_size
    for modality, grid_thw in cases:
        _check_sharded_tower(modality, grid_thw)


@multi_gpu_test(num_gpus=2)
def test_dp_sharded_tower_empty_rank_and_reorder():
    """One item on two ranks leaves rank 1 empty; three uneven items must
    come back in the original order despite the size-based assignment."""
    mp.spawn(
        _run_worker,
        args=(
            2,
            get_open_port(),
            [
                ("image", [[1, 12, 8]]),
                ("image", [[1, 16, 4], [1, 6, 10], [1, 4, 4]]),
            ],
        ),
        nprocs=2,
    )


@multi_gpu_test(num_gpus=4)
def test_dp_sharded_tower_uneven_video_and_empty_ranks():
    """Three images on four ranks and two frame-pooled videos on four ranks:
    empty ranks, non-divisible assignment and variable output lengths."""
    mp.spawn(
        _run_worker,
        args=(
            4,
            get_open_port(),
            [
                ("image", [[1, 12, 8], [1, 20, 6], [1, 8, 8]]),
                ("video", [[4, 8, 8], [2, 6, 6]]),
            ],
        ),
        nprocs=4,
    )
