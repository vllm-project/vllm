# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.utils import dispatch_cpu_unquantized_gemm
from vllm.model_executor.models.locate_anything import (
    LocateAnythingMultiModalProjector,
)
from vllm.transformers_utils.configs.locate_anything import LocateAnythingConfig

# CPU-only unit tests; core_model puts them in the per-PR CI run that
# sweeps tests/models/multimodal with `-m core_model`.
pytestmark = pytest.mark.core_model


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    with set_current_vllm_config(VllmConfig()):
        yield


def _config():
    return LocateAnythingConfig(
        vision_config=dict(hidden_size=1152, merge_kernel_size=[2, 2]),
        text_config=dict(hidden_size=2048),
    )


def _projector():
    proj = LocateAnythingMultiModalProjector(_config())
    dispatch_cpu_unquantized_gemm(proj.linear_1, remove_weight=False)
    dispatch_cpu_unquantized_gemm(proj.linear_2, remove_weight=False)
    return proj


def test_output_shape_after_2x2_merge():
    proj = _projector()
    feats = torch.randn(16, 1152)
    out = proj(feats)
    assert out.shape == (4, 2048)


def test_layernorm_is_over_merged_dim():
    proj = _projector()
    assert proj.pre_norm.normalized_shape == (4608,)


def test_named_submodules_exist():
    proj = _projector()
    names = dict(proj.named_modules())
    assert "pre_norm" in names
    assert "linear_1" in names
    assert "linear_2" in names


def test_forward_applies_norm_after_merge():
    proj = _projector()
    # vLLM linear layers ship uninitialized weights (filled at load time); give
    # them finite values so the numerical comparison below is meaningful.
    torch.manual_seed(0)
    with torch.no_grad():
        for lin in (proj.linear_1, proj.linear_2):
            lin.weight.normal_()
            if lin.bias is not None:
                lin.bias.normal_()
    feats = torch.randn(16, 1152)

    out = proj(feats)

    # Reference pins the InternVL-style ordering: 2x2 merge (flatten to 4608)
    # FIRST, THEN LayerNorm over the merged 4608-d feature, then linear/GELU/
    # linear. Asserting on shape alone would not catch a regression that moves
    # the norm before the merge (Kimi-VL style, over 1152) — this does.
    merged = feats.view(-1, 4608)
    normed = proj.pre_norm(merged)
    hidden, _ = proj.linear_1(normed)
    hidden = proj.act(hidden)
    expected, _ = proj.linear_2(hidden)

    assert torch.allclose(out, expected, atol=1e-5)
