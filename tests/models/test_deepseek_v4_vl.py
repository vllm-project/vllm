# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The loader reaches the language model's post-load hook inside VL wrappers."""

import importlib
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

import vllm.utils.gpu_sync_debug as gpu_sync_debug
from tests.models.registry import HF_EXAMPLE_MODELS
from tests.models.utils import dummy_hf_overrides
from vllm import SamplingParams
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
from vllm.models.deepseek_v41.common.mm_preprocess import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_START,
)
from vllm.models.deepseek_v41.common.vl_cudagraph import (
    DeepseekV4VLEncoderCudaGraphMixin,
)
from vllm.platforms import current_platform


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


class _SpanHost(DeepseekV4VLEncoderCudaGraphMixin, nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(vision_downsample_ratio=3)
        self.image_start = nn.Parameter(torch.randn(hidden_size, dtype=torch.float32))
        self.image_newline = nn.Parameter(torch.randn(hidden_size, dtype=torch.float32))
        self.image_end = nn.Parameter(torch.randn(hidden_size, dtype=torch.float32))


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Requires CUDA or ROCm"
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_deepseek_v41_build_image_span_no_gpu_sync(monkeypatch, dtype) -> None:
    """Eager and CUDA-graph span assembly must match reference without GPU sync."""
    torch.manual_seed(0)
    device = torch.device(current_platform.device_type)
    dim = 64
    host = _SpanHost(dim).to(device)

    types_0 = [IMAGE_START, IMAGE, IMAGE, IMAGE_NEW_LINE, IMAGE_END]
    types_1 = [IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE, IMAGE_NEW_LINE, IMAGE_END]
    # Unpinned CPU tensor sliced per image, matching `mm_preprocess` (`keep_on_cpu`).
    packed_types = torch.tensor(types_0 + types_1, dtype=torch.int64)
    assert not packed_types.is_pinned()
    device_types_0 = packed_types[: len(types_0)].to(device)

    rows_0 = torch.randn(2, dim, device=device, dtype=dtype)
    rows_1 = torch.randn(2, dim, device=device, dtype=dtype)
    padded_aligner_out = torch.cat(
        [rows_0, rows_1, torch.full((3, dim), float("nan"), device=device, dtype=dtype)]
    )

    start = host.image_start.to(dtype)
    newline = host.image_newline.to(dtype)
    end = host.image_end.to(dtype)
    expected_0 = torch.stack([start, rows_0[0], rows_0[1], newline, end])
    expected_1 = torch.stack([start, rows_1[0], newline, rows_1[1], newline, end])

    # Warm up allocator and any lazy kernel init outside the checked region.
    _ = host._build_image_span(rows_0, packed_types[: len(types_0)])

    monkeypatch.setattr(gpu_sync_debug, "_SYNC_CHECK_MODE", "error")
    monkeypatch.setattr(gpu_sync_debug, "_sync_check_enabled", False)
    gpu_sync_debug.enable_gpu_sync_check()

    @gpu_sync_debug.with_gpu_sync_check
    def run_checked():
        span_0 = host._build_image_span(rows_0, packed_types[: len(types_0)])
        span_1 = host._build_image_span(rows_1, packed_types[len(types_0) :])
        span_dev = host._build_image_span(rows_0, device_types_0)
        dest: list[torch.Tensor | None] = [None, None]
        host.postprocess_encoder_output(
            outputs={"default": padded_aligner_out},
            indices=[0, 1],
            per_item_out_tokens=[len(types_0), len(types_1)],
            dest=dest,
            batch_mm_kwargs={
                "vit_grid": [[3, 6], [6, 3]],
                "llm_grid": [[1, 2], [2, 1]],
                "types": packed_types,
            },
        )
        return span_0, span_1, span_dev, dest

    span_0, span_1, span_dev, dest = run_checked()
    torch.testing.assert_close(span_0, expected_0, atol=0, rtol=0)
    torch.testing.assert_close(span_1, expected_1, atol=0, rtol=0)
    torch.testing.assert_close(span_dev, expected_0, atol=0, rtol=0)
    assert dest[0] is not None and dest[1] is not None
    torch.testing.assert_close(dest[0], expected_0, atol=0, rtol=0)
    torch.testing.assert_close(dest[1], expected_1, atol=0, rtol=0)


@pytest.mark.skipif(
    not (
        current_platform.is_rocm()
        or (current_platform.is_cuda() and current_platform.has_device_capability(90))
    ),
    reason="Requires ROCm or CUDA capability >= 9.0",
)
def test_deepseek_v41_eager_image_no_gpu_sync(monkeypatch, vllm_runner) -> None:
    """Eager vision encoding (`cudagraph_mm_encoder=False`) must not sync."""
    monkeypatch.setenv("VLLM_GPU_SYNC_CHECK", "error")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    arch = "DeepseekV41ForCausalLM"
    model_info = HF_EXAMPLE_MODELS.get_hf_info(arch)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    prompt = (
        "<｜begin▁of▁sentence｜><｜User｜><｜deepseek_image｜>\n\n"
        "What is in this image?<｜Assistant｜></think>"
    )
    images = [
        Image.new("RGB", (224, 224), color=(128, 64, 32)),
        Image.new("RGB", (336, 224), color=(32, 128, 64)),
    ]

    with vllm_runner(
        model_info.default,
        dtype="bfloat16",
        max_model_len=2048,
        max_num_seqs=2,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
        load_format="dummy",
        attention_backend=(
            "ROCM_FLASHMLA_SPARSE_DSV41"
            if current_platform.is_rocm()
            else "FLASHMLA_SPARSE_DSV41"
        ),
        hf_overrides=partial(
            dummy_hf_overrides,
            model_arch=arch,
            exist_overrides={"vision_n_layers": 1},
        ),
    ) as runner:
        outputs = runner.llm.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": img}} for img in images],
            SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True),
        )

    assert [len(output.outputs[0].token_ids) for output in outputs] == [4, 4]
