# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity tests for the DeepSeek-V4 vision tower against the official
reference implementation."""

import importlib.util
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.common.vision import (
    DeepseekV4Aligner,
    DeepseekV4ViT,
)

REF_VISION_PATH = "/tmp/dsv4vis/vision.py"


def _load_reference_vision():
    spec = importlib.util.spec_from_file_location("dsv4vis_ref", REF_VISION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref_vision = _load_reference_vision()


def _make_config() -> SimpleNamespace:
    return SimpleNamespace(
        vision_n_layers=2,
        vision_dim=64,
        vision_n_heads=4,
        vision_inter_dim=88,
        vision_patch_size=14,
        vision_rope_theta=10000.0,
        vision_downsample_ratio=3,
        hidden_size=96,
    )


def _ref_args(config: SimpleNamespace) -> SimpleNamespace:
    args = SimpleNamespace(**vars(config))
    args.dim = config.hidden_size
    return args


@pytest.mark.parametrize("n_vit_h,n_vit_w", [(6, 6), (7, 5), (4, 10)])
def test_vit_parity(n_vit_h: int, n_vit_w: int):
    torch.manual_seed(0)
    config = _make_config()
    ours = DeepseekV4ViT(config)
    ref = ref_vision.ViT(_ref_args(config))
    ref.load_state_dict(ours.state_dict())
    ours.eval()
    ref.eval()

    n_tokens = n_vit_h * n_vit_w
    patches = torch.randn(
        n_tokens, 3, config.vision_patch_size, config.vision_patch_size
    )
    with torch.no_grad():
        out_ours = ours(patches, n_vit_h, n_vit_w)
        out_ref = ref(patches, n_vit_h, n_vit_w)
    torch.testing.assert_close(out_ours, out_ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("n_vit_h,n_vit_w", [(6, 6), (7, 5), (4, 10)])
def test_aligner_parity(n_vit_h: int, n_vit_w: int):
    torch.manual_seed(0)
    config = _make_config()
    ours = DeepseekV4Aligner(config)
    ref = ref_vision.Aligner(_ref_args(config))
    ref.load_state_dict(ours.state_dict())
    ours.eval()
    ref.eval()

    n_tokens = n_vit_h * n_vit_w
    x = torch.randn(n_tokens, config.vision_dim)
    with torch.no_grad():
        out_ours = ours(x, n_vit_h, n_vit_w)
        out_ref = ref(x, n_vit_h, n_vit_w)
    assert out_ours.shape == out_ref.shape
    torch.testing.assert_close(out_ours, out_ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("n_vit_h,n_vit_w", [(7, 5), (4, 10)])
def test_aligner_output_grid(n_vit_h: int, n_vit_w: int):
    torch.manual_seed(0)
    config = _make_config()
    aligner = DeepseekV4Aligner(config)
    r = config.vision_downsample_ratio
    n_llm_h = -(-n_vit_h // r)
    n_llm_w = -(-n_vit_w // r)
    x = torch.randn(n_vit_h * n_vit_w, config.vision_dim)
    with torch.no_grad():
        out = aligner(x, n_vit_h, n_vit_w)
    assert out.shape == (n_llm_h * n_llm_w, config.hidden_size)
