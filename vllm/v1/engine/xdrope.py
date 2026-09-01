# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

from vllm.exceptions import VLLMValidationError
from vllm.multimodal.inputs import MultiModalFeatureSpec

if TYPE_CHECKING:
    from vllm.config import ModelConfig


def _as_grid_thw(value: Any) -> tuple[int, int, int]:
    grid = value.tolist() if hasattr(value, "tolist") else value
    if (
        not isinstance(grid, (list, tuple))
        or len(grid) != 3
        or any(isinstance(dim, bool) or not isinstance(dim, int) for dim in grid)
        or any(dim <= 0 for dim in grid)
    ):
        raise VLLMValidationError(
            "XD-RoPE image_grid_thw entries must contain three positive integers."
        )
    return grid[0], grid[1], grid[2]


def validate_xdrope_input(
    model_config: "ModelConfig",
    prompt_token_ids: list[int] | None,
    mm_features: list[MultiModalFeatureSpec],
    prompt_is_token_ids: list[bool] | None = None,
    allow_unresolved_features: bool = False,
) -> None:
    """Validate XD-RoPE markers and image geometry before worker execution."""
    if getattr(model_config, "uses_xdrope_dim", 0) <= 0 or prompt_token_ids is None:
        return

    hf_config = model_config.hf_config
    image_start_token_id = getattr(hf_config, "image_start_token_id", None)
    vision_config = getattr(hf_config, "vision_config", None)
    spatial_merge_size = getattr(vision_config, "spatial_merge_size", None)
    if image_start_token_id is None or not isinstance(spatial_merge_size, int):
        return
    if spatial_merge_size <= 0:
        raise VLLMValidationError(
            "XD-RoPE spatial_merge_size must be a positive integer."
        )

    marker_positions = [
        index
        for index, token_id in enumerate(prompt_token_ids)
        if token_id == image_start_token_id
        and (
            prompt_is_token_ids is None
            or len(prompt_is_token_ids) != len(prompt_token_ids)
            or prompt_is_token_ids[index]
        )
    ]
    has_unresolved_image = any(
        feature.data is None and feature.modality == "image" for feature in mm_features
    )
    if has_unresolved_image and allow_unresolved_features:
        return
    kwargs = MultiModalFeatureSpec.gather_kwargs(mm_features, {"image_grid_thw"})
    image_grids = kwargs.get("image_grid_thw", [])
    if len(marker_positions) != len(image_grids):
        raise VLLMValidationError(
            "XD-RoPE image marker count does not match image_grid_thw count."
        )

    prompt_len = len(prompt_token_ids)
    for marker_position, raw_grid in zip(marker_positions, image_grids):
        _, height, width = _as_grid_thw(raw_grid)
        merged_height = height // spatial_merge_size
        merged_width = width // spatial_merge_size
        token_count = (merged_width + 1) * merged_height
        if marker_position + 2 + token_count > prompt_len:
            raise VLLMValidationError(
                "XD-RoPE image geometry extends past the end of the prompt."
            )
