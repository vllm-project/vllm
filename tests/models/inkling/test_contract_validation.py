# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.config.compilation import CompilationConfig, CUDAGraphMode
from vllm.models.inkling.common.mm_preprocess import InklingMultiModalDataParser
from vllm.models.inkling.common.towers import plan_out_scales
from vllm.models.inkling.configs import (
    InklingAudioConfig,
    InklingModelConfig,
    InklingVisionConfig,
)
from vllm.models.inkling.nvidia.sconv_swa_attn import (
    InklingSconvMetadataBuilder,
)
from vllm.v1.attention.backend import AttentionCGSupport


def test_vision_scale_plan_matches_released_config():
    assert plan_out_scales(2, 40, 4) == [
        (1, 1, 1, 3),
        (1, 5, 5, 128),
        (1, 10, 10, 320),
        (1, 40, 40, 4800),
        (2, 40, 40, 9600),
    ]


def test_vision_scale_plan_breaks_assignment_ties_in_order():
    reductions = [np.prod(scale[:-1]) for scale in plan_out_scales(2, 52, 4)]

    assert reductions == sorted(set(reductions))


@pytest.mark.parametrize(
    ("config_cls", "kwargs", "missing"),
    [
        (InklingAudioConfig, {"decoder_dmodel": 16}, "n_mel_bins"),
        (InklingVisionConfig, {"decoder_dmodel": 16}, "vision_encoder_type"),
    ],
)
def test_enabled_tower_requires_architecture_fields(config_cls, kwargs, missing):
    with pytest.raises(ValueError, match=missing):
        config_cls(**kwargs)


def test_inkling_raw_2d_audio_is_rejected_as_ambiguous():
    parser = InklingMultiModalDataParser(target_sr=16_000, target_channels=1)
    with pytest.raises(ValueError, match="ambiguous channel layout"):
        parser._parse_audio_data(np.zeros((2, 100), dtype=np.float32))


def test_inkling_mtp_chain_norm_is_disabled_by_default():
    assert InklingModelConfig().chain_hidden_post_norm is False


def test_inkling_supports_piecewise_cudagraphs():
    support = InklingSconvMetadataBuilder.get_cudagraph_support
    assert support(None, None) == AttentionCGSupport.UNIFORM_BATCH

    compilation_config = CompilationConfig(
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        splitting_ops=[],
    )
    resolved_mode = compilation_config.resolve_cudagraph_mode_and_sizes(
        AttentionCGSupport.UNIFORM_BATCH,
        "InklingSconvBackend",
    )

    assert resolved_mode == CUDAGraphMode.PIECEWISE
