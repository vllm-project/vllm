# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.platforms import current_platform

from ._vit_cudagraph import (
    CORE_MODEL_CONFIGS,
    params_with_marks,
    run_vit_cudagraph_image,
    run_vit_cudagraph_video,
)


@pytest.mark.parametrize("model_id", params_with_marks(CORE_MODEL_CONFIGS))
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Skip if not cuda or rocm"
)
def test_vit_cudagraph_image(model_id, vllm_runner, image_assets):
    run_vit_cudagraph_image(model_id, vllm_runner, image_assets)


@pytest.mark.parametrize("model_id", params_with_marks(CORE_MODEL_CONFIGS))
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Skip if not cuda or rocm"
)
def test_vit_cudagraph_video(model_id, vllm_runner, video_assets):
    run_vit_cudagraph_video(model_id, vllm_runner, video_assets)
