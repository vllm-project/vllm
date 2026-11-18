# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The loader reaches the language model's post-load hook inside VL wrappers."""

import importlib

import pytest
import torch
from torch import nn

from vllm.model_executor.model_loader.weight_utils import finalize_model_weights
from vllm.model_executor.models.utils import WeightsMapper
from vllm.models.deepseek_v4.common.ops.vision import (
    apply_rotary,
    vision_qk_rotary,
    vision_spatial_merge,
)
from vllm.models.deepseek_v4.common.vision import (
    DeepseekV4RMSNorm,
    build_packed_merge_metadata,
    get_vision_cos_sin,
)


class _FakeLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tensor_a = nn.Parameter(torch.zeros(1))
        self.tensor_c = nn.Parameter(torch.zeros(1))
        self.finalized_values: list[tuple[float, float]] = []

    def process_weights_after_loading(self) -> None:
        self.finalized_values.append((self.tensor_a.item(), self.tensor_c.item()))


@pytest.mark.parametrize(
    "module,cls_name",
    [
        (
            "vllm.models.deepseek_v4.common.vl_model",
            "DeepseekV4ForConditionalGeneration",
        ),
        ("vllm.models.deepseek_v41.nvidia.vl_model", "DeepseekV41ForCausalLM"),
        ("vllm.models.deepseek_v41.amd.vl_model", "DeepseekV41ForCausalLM"),
    ],
    ids=["v4", "v41-nvidia", "v41-amd"],
)
def test_vl_wrapper_runs_language_model_hook(module, cls_name) -> None:
    wrapper_cls = getattr(importlib.import_module(module), cls_name)

    model = object.__new__(wrapper_cls)
    nn.Module.__init__(model)
    model.language_model = _FakeLanguageModel()
    model.vision = nn.Module()
    model.vision.tensor_b = nn.Parameter(torch.zeros(1))
    model.hf_to_vllm_mapper = WeightsMapper()

    def interleaved_weights():
        yield "language_model.tensor_a", torch.tensor([1.0])
        assert model.language_model.tensor_a.item() == 1.0
        yield "vision.tensor_b", torch.tensor([2.0])
        assert model.vision.tensor_b.item() == 2.0
        yield "language_model.tensor_c", torch.tensor([3.0])

    loaded = model.load_weights(interleaved_weights())

    assert loaded == {
        "language_model.tensor_a",
        "vision.tensor_b",
        "language_model.tensor_c",
    }
    assert model.language_model.finalized_values == []

    finalize_model_weights(model)

    assert model.language_model.finalized_values == [(1.0, 3.0)]

    # A reload reruns the hook: no one-shot guard may skip it.
    finalize_model_weights(model)
    assert model.language_model.finalized_values == [(1.0, 3.0)] * 2


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(0, 1024), (7, 128), (83, 1024), (1566, 1024)])
@torch.inference_mode()
def test_vision_rms_norm_preserves_reference_rounding(dtype, weight_dtype, shape):
    """Fusing pointwise ops must preserve the ViT's FP32 mean and weight multiply."""
    torch.manual_seed(0)
    rows, dim = shape
    layer = DeepseekV4RMSNorm(dim).to(device="cuda", dtype=weight_dtype)
    layer.weight.normal_(mean=1.0, std=0.2)
    # An offset view also exercises unaligned input loads.
    x = torch.randn(rows * dim + 1, device="cuda", dtype=dtype)[1:].view(shape)
    if rows:
        x[0].zero_()
        x[1].mul_(1e-6)
        x[2].mul_(100)
    ref = x.float()
    ref = ref * torch.rsqrt(ref.square().mean(-1, keepdim=True) + layer.eps)
    ref = (layer.weight * ref).to(dtype)
    torch.testing.assert_close(layer(x), ref, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("heads", [2, 4, 16])
@pytest.mark.parametrize("grids", [[(1, 7)], [(3, 5), (2, 9)], [(18, 87)]])
@torch.inference_mode()
def test_vision_qk_rotary_preserves_packed_positions(dtype, heads, grids):
    """Q/K fusion keeps per-image 2D positions and leaves the V slice untouched."""
    torch.manual_seed(0)
    n = sum(h * w for h, w in grids)
    dim = 64
    storage = torch.randn(n * 3 * heads * dim + 1, device="cuda", dtype=dtype)
    qkv = storage[1:].view(n, 3 * heads * dim)
    original = qkv.clone()
    tables = [get_vision_cos_sin(h, w, dim // 2, 10000.0) for h, w in grids]
    cos, sin = (torch.cat(parts).cuda() for parts in zip(*tables))
    q, k, _ = (t.view(n, heads, dim) for t in qkv.chunk(3, -1))
    expected = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
    actual = vision_qk_rotary(qkv, cos, sin, heads, dim)
    for out, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(out, ref, atol=0, rtol=0)
    torch.testing.assert_close(qkv, original, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("grids", [[(2, 2)], [(3, 5), (7, 9)], [(18, 87)]])
@torch.inference_mode()
def test_vision_spatial_merge_preserves_channel_order_and_padding(dtype, grids):
    """The packed merge must match per-image padding and channel-major unfold."""
    torch.manual_seed(0)
    dim, ratio = 128, 3
    sizes = [h * w for h, w in grids]
    x = torch.randn(sum(sizes), dim, device="cuda", dtype=dtype)
    metadata = build_packed_merge_metadata(grids, ratio, device=x.device, dtype=dtype)
    expected = []
    for item, (h, w) in zip(x.split(sizes), grids, strict=True):
        item = item.view(h, w, dim).permute(2, 0, 1)
        item = torch.nn.functional.pad(item, (0, -w % ratio, 0, -h % ratio))
        expected.append(
            torch.nn.functional.unfold(item.unsqueeze(0), ratio, stride=ratio)
            .squeeze(0)
            .T
        )
    actual = vision_spatial_merge(x, metadata["merge_idx"], metadata["merge_mask"])
    torch.testing.assert_close(actual, torch.cat(expected), atol=0, rtol=0)
