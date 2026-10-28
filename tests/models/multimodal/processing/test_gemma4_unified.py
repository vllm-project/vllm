# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping

import pytest
import torch
from PIL import Image as PILImage
from transformers.models.gemma4 import Gemma4AudioFeatureExtractor

from vllm.exceptions import VLLMValidationError
from vllm.model_executor.models.gemma4_mm import (
    Gemma4ImagePixelInputs,
    _dummy_audio_num_samples,
)
from vllm.model_executor.models.gemma4_unified import (
    Gemma4UnifiedForConditionalGeneration,
    Gemma4UnifiedProcessingInfo,
    Gemma4UnifiedVisionEmbedder,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.utils.gpu_sync_debug import gpu_sync_allowed

from ....conftest import ImageTestAssets
from ...utils import build_model_context

# The Unified model ID for testing purposes
GEMMA4_UNIFIED_MODEL_ID = "google/gemma-4-12B-it"


def test_gemma4_unified_image_schema_accepts_variable_patch_counts():
    Gemma4ImagePixelInputs(
        pixel_values=[
            torch.randn(10080, 768),
            torch.randn(2520, 768),
        ],
        pixel_position_ids=[
            torch.zeros(10080, 2, dtype=torch.long),
            torch.zeros(2520, 2, dtype=torch.long),
        ],
    )


def test_gemma4_unified_image_batching_keeps_variable_patch_counts_unstacked():
    field = MultiModalFieldConfig.batched("image").field
    elems = field.build_elems(
        "image",
        "pixel_values",
        [torch.randn(10080, 768), torch.randn(2520, 768)],
    )

    reduced = field.reduce_data(list(elems))

    assert isinstance(reduced, list)
    assert [tensor.shape for tensor in reduced] == [
        torch.Size([10080, 768]),
        torch.Size([2520, 768]),
    ]


@pytest.mark.parametrize(
    "image_width,image_height,max_soft_tokens",
    [
        (900, 3, 280),
        (3, 900, 280),
        (900, 3, 70),
        (4000, 2, 1120),
    ],
)
@pytest.mark.parametrize("model_id", [GEMMA4_UNIFIED_MODEL_ID])
def test_compute_num_soft_tokens_does_not_exceed_max_soft_tokens(
    model_id: str,
    image_width: int,
    image_height: int,
    max_soft_tokens: int,
):
    """Verify ``_compute_num_soft_tokens`` caps output at ``max_soft_tokens``."""
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs={"do_pan_and_scan": True},
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    assert isinstance(processor.info, Gemma4UnifiedProcessingInfo)

    num_soft_tokens = processor.info._compute_num_soft_tokens(
        image_width=image_width,
        image_height=image_height,
        max_soft_tokens=max_soft_tokens,
    )

    assert num_soft_tokens <= max_soft_tokens, (
        f"_compute_num_soft_tokens returned {num_soft_tokens} for "
        f"image_width={image_width}, image_height={image_height}, "
        f"max_soft_tokens={max_soft_tokens} — exceeds the cap."
    )


@pytest.mark.parametrize(
    ("mm_processor_kwargs", "expected_image_tokens"),
    [
        ({}, 280),
        ({"max_soft_tokens": 70}, 70),
        ({"max_soft_tokens": 280}, 280),
        ({"max_soft_tokens": 1120}, 1120),
        ({"images_kwargs": {"max_soft_tokens": 560}}, 560),
        ({"images_kwargs": None}, 280),
        ({"images_kwargs": "not-a-dict"}, 280),
    ],
)
@pytest.mark.parametrize("model_id", [GEMMA4_UNIFIED_MODEL_ID])
def test_get_mm_max_tokens_per_item_respects_configured_max_soft_tokens(
    model_id: str,
    mm_processor_kwargs: dict[str, object],
    expected_image_tokens: int,
):
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs,
        limit_mm_per_prompt={"image": 1, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    tokens = processor.info.get_mm_max_tokens_per_item(
        seq_len=ctx.model_config.max_model_len,
        mm_counts={"image": 1, "video": 1},
    )

    assert tokens is not None
    assert tokens["image"] == expected_image_tokens
    assert tokens["video"] == 32 * (70 + 2 + 6)


@pytest.mark.parametrize(
    ("limit_mm_per_prompt", "expected_video_tokens"),
    [
        ({"video": 1}, 32 * (70 + 2 + 6)),
        ({"video": {"count": 1}}, 32 * (70 + 2 + 6)),
        ({"video": {"count": 1, "num_frames": 1}}, 1 * (70 + 2 + 6)),
        ({"video": {"count": 1, "num_frames": 8}}, 8 * (70 + 2 + 6)),
        ({"video": {"count": 1, "num_frames": 32}}, 32 * (70 + 2 + 6)),
        ({"video": {"count": 1, "num_frames": 40}}, 32 * (70 + 2 + 6)),
    ],
)
@pytest.mark.parametrize("model_id", [GEMMA4_UNIFIED_MODEL_ID])
def test_get_mm_max_tokens_per_item_respects_configured_video_num_frames(
    model_id: str,
    limit_mm_per_prompt: Mapping[str, int | Mapping[str, int]],
    expected_video_tokens: int,
):
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt=limit_mm_per_prompt,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    tokens = processor.info.get_mm_max_tokens_per_item(
        seq_len=ctx.model_config.max_model_len,
        mm_counts={"video": 1},
    )

    assert tokens is not None
    assert tokens["image"] == 280
    assert tokens["video"] == expected_video_tokens


@pytest.mark.parametrize("model_id", [GEMMA4_UNIFIED_MODEL_ID])
def test_get_prompt_updates_respects_nested_max_soft_tokens(model_id: str):
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs={"images_kwargs": {"max_soft_tokens": 560}},
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    assert isinstance(processor.info, Gemma4UnifiedProcessingInfo)
    image = PILImage.new("RGB", (1000, 1000), color="white")
    image_size = image.size
    mm_items = processor.info.parse_mm_data({"image": image})

    prompt_update = processor._get_prompt_updates(mm_items, {}, {})[0]
    replacement = prompt_update.resolve(0).content.full
    expected = processor.info.get_image_repl(
        image_width=image_size[0],
        image_height=image_size[1],
        processor=processor.info.get_hf_processor(),
        max_soft_tokens=560,
    ).full

    assert replacement == expected


@pytest.mark.parametrize("model_id", [GEMMA4_UNIFIED_MODEL_ID])
def test_limit_mm_per_prompt(
    image_assets: ImageTestAssets,
    model_id: str,
):
    """Test that limit_mm_per_prompt restricts multiple images correctly."""
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs={},
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    prompt = "<image><image>"
    images = [asset.pil_image for asset in image_assets][:2]
    if len(images) < 2:
        images = [images[0], images[0]]

    mm_data = {"image": images}

    with pytest.raises(VLLMValidationError, match="At most 1 image"):
        processor(
            prompt,
            mm_items=processor.info.parse_mm_data(mm_data),
            hf_processor_mm_kwargs={},
        )


def test_dummy_audio_length_falls_back_for_the_unified_extractor():
    """The unified audio feature extractor does not define ``fft_length``.

    ``Gemma4DummyInputsBuilder.get_dummy_mm_data`` read that attribute
    unconditionally, so memory profiling raised ``AttributeError`` for every
    unified Gemma 4 deployment in which audio is the modality with the largest
    per-item token count -- which is the case as soon as ``audio_seq_length``
    is raised above the video budget.
    """

    class _UnifiedFeatureExtractor:
        sampling_rate = 16000

    class _UnifiedProcessor:
        feature_extractor = _UnifiedFeatureExtractor()
        audio_seq_length = 3000
        audio_ms_per_token = 40

    # 3000 tokens * 40 ms = 120 s at 16 kHz
    assert _dummy_audio_num_samples(_UnifiedProcessor()) == 120 * 16000


def test_dummy_audio_length_still_uses_fft_length_for_the_tower_extractor():
    """The mel/tower extractor keeps its existing behaviour."""

    class _TowerFeatureExtractor(Gemma4AudioFeatureExtractor):
        def __init__(self):  # noqa: D107 - bypass the real feature-extractor init
            self.fft_length = 512
            self.sampling_rate = 16000

    class _TowerProcessor:
        feature_extractor = _TowerFeatureExtractor()
        audio_seq_length = 750
        audio_ms_per_token = 40

    assert _dummy_audio_num_samples(_TowerProcessor()) == 512


# ---------------------------------------------------------------------------
# Gemma4 Unified Vision Embedder & Multimodal Processing Unit Tests
# ---------------------------------------------------------------------------


def _reference_factorized_posemb(
    pos_embedding: torch.Tensor,
    positions_xy: torch.Tensor,
) -> torch.Tensor:
    """Reference loop implementation for factorized 2D positional embeddings."""
    max_pos = pos_embedding.shape[0]
    D = pos_embedding.shape[-1]
    orig_shape = positions_xy.shape
    pos_flat = positions_xy.reshape(-1, 2)
    out = torch.zeros(
        pos_flat.shape[0], D, dtype=pos_embedding.dtype, device=pos_embedding.device
    )
    for i in range(pos_flat.shape[0]):
        px = int(pos_flat[i, 0].item())
        py = int(pos_flat[i, 1].item())
        if px >= 0:
            cx = min(max(px, 0), max_pos - 1)
            out[i] += pos_embedding[cx, 0]
        if py >= 0:
            cy = min(max(py, 0), max_pos - 1)
            out[i] += pos_embedding[cy, 1]
    return out.reshape(*orig_shape[:-1], D)


def _build_test_embedder(
    embed_dim: int = 48,
    max_pos: int = 100,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Gemma4UnifiedVisionEmbedder:
    embedder = object.__new__(Gemma4UnifiedVisionEmbedder)
    torch.nn.Module.__init__(embedder)
    embedder.mm_embed_dim = embed_dim
    embedder.pos_embedding = torch.nn.Parameter(
        torch.randn(max_pos, 2, embed_dim, dtype=dtype, device=device)
    )
    embedder.patch_ln1 = torch.nn.LayerNorm(embed_dim).to(device=device, dtype=dtype)
    lin = torch.nn.Linear(embed_dim, embed_dim, dtype=dtype, device=device)
    embedder.patch_dense = lambda x: (lin(x), None)
    embedder.patch_ln2 = torch.nn.LayerNorm(embed_dim).to(device=device, dtype=dtype)
    embedder.pos_norm = torch.nn.LayerNorm(embed_dim).to(device=device, dtype=dtype)
    return embedder


class _MockEmbedVision(torch.nn.Module):
    def __init__(
        self, embed_dim: int, dtype: torch.dtype = torch.float32, device: str = "cpu"
    ):
        super().__init__()
        self.embedding_projection = torch.nn.Linear(
            embed_dim, embed_dim, bias=False, dtype=dtype, device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding_projection(x)


class _MockUnifiedModel(Gemma4UnifiedForConditionalGeneration):
    # Initialize only the modules needed by the borrowed vision methods.
    def __init__(
        self,
        embed_dim: int = 48,
        max_pos: int = 100,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        torch.nn.Module.__init__(self)
        self.vision_embedder = _build_test_embedder(
            embed_dim=embed_dim, max_pos=max_pos, dtype=dtype, device=device
        )
        self.embed_vision = _MockEmbedVision(
            embed_dim=embed_dim, dtype=dtype, device=device
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(16, 2), (2, 8, 2), (1, 1, 2)])
def test_factorized_posemb_parity_with_reference(
    dtype: torch.dtype, shape: tuple[int, ...]
):
    """Verify clean vectorized _factorized_posemb matches reference loop across
    dtypes.
    """
    embedder = _build_test_embedder(embed_dim=48, max_pos=100, dtype=dtype)

    positions = torch.randint(-3, 100, shape)
    if len(shape) == 2:
        positions[0] = torch.tensor([-1, -1])
        if shape[0] > 1:
            positions[1] = torch.tensor([-1, 20])
        if shape[0] > 2:
            positions[2] = torch.tensor([30, -1])
        if shape[0] > 3:
            positions[3] = torch.tensor([-10, -5])
    else:
        positions[0, 0] = torch.tensor([-1, -1])
        if shape[0] > 1 and shape[1] > 1:
            positions[0, 1] = torch.tensor([-1, 15])
            positions[1, 0] = torch.tensor([25, -1])
            positions[1, 1] = torch.tensor([-99, -42])

    res = embedder._factorized_posemb(positions)
    ref = _reference_factorized_posemb(embedder.pos_embedding, positions)

    assert res.dtype == dtype, f"Expected dtype {dtype}, got {res.dtype}"
    assert torch.equal(res, ref), f"Bit mismatch for dtype {dtype}, shape {shape}"


@pytest.mark.parametrize("shape", [(0, 2), (1, 0, 2), (0, 5, 2)])
def test_factorized_posemb_empty(shape: tuple[int, ...]):
    """Verify empty batches in _factorized_posemb return empty tensors without error."""
    embedder = _build_test_embedder(embed_dim=48, max_pos=100)
    positions = torch.empty(shape, dtype=torch.long)
    res = embedder._factorized_posemb(positions)
    assert res.shape == (*shape[:-1], 48)
    assert res.numel() == 0


def test_vision_chunk_budget_calculation():
    """Verify _vision_chunk budget calculation and corner cases."""
    assert Gemma4UnifiedForConditionalGeneration._vision_chunk(0, 1000, 1000) == 1
    assert Gemma4UnifiedForConditionalGeneration._vision_chunk(-5, 1000, 1000) == 1

    total_bytes = 100 * (1024**3)
    free_bytes = 80 * (1024**3)
    patches = 280
    patch_dim = 6912
    mm_embed_dim = 3840
    chunk = Gemma4UnifiedForConditionalGeneration._vision_chunk(
        patches, free_bytes, total_bytes, patch_dim, mm_embed_dim
    )
    cost = patches * (2 * patch_dim + 4 * mm_embed_dim) * 2
    budget = min(free_bytes // 2, total_bytes // 10)
    assert chunk == max(1, budget // cost)
    assert chunk > 1

    # Tiny free memory returns at least 1
    assert Gemma4UnifiedForConditionalGeneration._vision_chunk(280, 100, 1000) == 1


def test_process_image_input_batching_and_chunking():
    """Verify _process_image_input with 3D tensors, varlen lists, and chunking
    fallback.
    """
    from unittest.mock import patch

    model = _MockUnifiedModel()  # type: ignore[abstract]

    # 1. 3D stacked image tensor
    pv = torch.randn(2, 5, 48)
    pp = torch.randint(0, 100, (2, 5, 2))
    pp[0, 4] = -1  # 1 padded patch in img 0
    pp[1, 3:] = -1  # 2 padded patches in img 1
    img_input = {"pixel_values": pv, "pixel_position_ids": pp}
    out = model._process_image_input(img_input)
    assert len(out) == 2
    assert out[0].shape == (4, 48)
    assert out[1].shape == (3, 48)

    # 2. list[torch.Tensor] variable patch counts
    pv_list = [torch.randn(6, 48), torch.randn(3, 48)]
    pp_list = [torch.randint(0, 100, (6, 2)), torch.randint(0, 100, (3, 2))]
    pp_list[0][5] = -1
    img_input_var = {"pixel_values": pv_list, "pixel_position_ids": pp_list}
    out_var = model._process_image_input(img_input_var)
    assert len(out_var) == 2
    assert out_var[0].shape == (5, 48)
    assert out_var[1].shape == (3, 48)

    # 3. Empty list
    assert (
        model._process_image_input({"pixel_values": [], "pixel_position_ids": []}) == []
    )

    # 4. Multi-chunk fallback when max_batch is small
    with patch.object(model, "_vision_chunk", return_value=1) as mock_chunk:
        out_chunked = model._process_image_input(img_input)
        mock_chunk.assert_called_once()
        assert len(out_chunked) == 2
        assert torch.equal(out[0], out_chunked[0])
        assert torch.equal(out[1], out_chunked[1])


def test_process_video_input_batching():
    """Verify _process_video_input with frame counts and lists of tensors."""
    model = _MockUnifiedModel()  # type: ignore[abstract]

    # 2 videos: 3 frames and 2 frames, each frame has 4 patches
    pv_vid = torch.randn(5, 4, 48)
    pp_vid = torch.randint(0, 100, (5, 4, 2))
    pp_vid[0, 3] = -1
    fc = torch.tensor([3, 2])
    vid_input = {
        "pixel_values_videos": pv_vid,
        "pixel_position_ids_videos": pp_vid,
        "video_frame_counts": fc,
    }
    out_vid = model._process_video_input(vid_input)
    assert len(out_vid) == 2
    assert out_vid[0].shape == (3 * 4 - 1, 48)
    assert out_vid[1].shape == (2 * 4, 48)

    # Video with list of tensors and list of frame counts
    vid_input_list = {
        "pixel_values_videos": [pv_vid[:3], pv_vid[3:]],
        "pixel_position_ids_videos": [pp_vid[:3], pp_vid[3:]],
        "video_frame_counts": [3, 2],
    }
    out_vid_list = model._process_video_input(vid_input_list)
    assert len(out_vid_list) == 2
    assert torch.equal(out_vid[0], out_vid_list[0])
    assert torch.equal(out_vid[1], out_vid_list[1])

    # Empty video
    empty_vid = {
        "pixel_values_videos": torch.empty((0, 4, 48)),
        "pixel_position_ids_videos": torch.empty((0, 4, 2), dtype=torch.long),
        "video_frame_counts": [],
    }
    assert model._process_video_input(empty_vid) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_execution_with_gpu_sync_allowed():
    """Verify vision processing on CUDA under gpu_sync_allowed."""
    device = "cuda:0"
    model = _MockUnifiedModel(device=device, dtype=torch.bfloat16)  # type: ignore[abstract]

    pv = torch.randn(2, 5, 48, device=device, dtype=torch.bfloat16)
    pp = torch.randint(0, 100, (2, 5, 2), device=device)
    pp[0, 4] = -1
    img_input = {"pixel_values": pv, "pixel_position_ids": pp}

    with gpu_sync_allowed():
        out = model._process_image_input(img_input)

    assert len(out) == 2
    assert out[0].device.type == "cuda"
    assert out[0].dtype == torch.bfloat16
