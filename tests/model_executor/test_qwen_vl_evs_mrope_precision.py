# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for M-RoPE position precision during multimodal pruning.

When video token pruning (EVS / VidCom2) is enabled, M-RoPE position metadata
travels temporarily as trailing columns on the floating-point multimodal
embedding tensor between ``embed_multimodal`` and ``recompute_mrope_positions``.
In ``bfloat16`` (8 significand bits) and ``float16`` (11 significand bits),
concatenating raw ``int64`` positions directly onto the embedding tensor rounded
positions above 256 / 2048.
"""

import dataclasses
from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.models.cohere_compass import (
    CohereCompassForConditionalGeneration,
)
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
)
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.multimodal.video_prune.evs import (
    compute_mrope_for_media,
    pack_mrope_positions,
    unpack_mrope_positions,
)
from vllm.multimodal.video_prune.evs import (
    compute_retention_mask as evs_compute_retention_mask,
)
from vllm.multimodal.video_prune.vidcom2 import (
    compute_retention_mask as vidcom2_compute_retention_mask,
)

IMAGE_TOKEN_ID = 999
VIDEO_TOKEN_ID = 888
VISION_START_TOKEN_ID = 777
VISION_END_TOKEN_ID = 778


@pytest.fixture(autouse=True, scope="module")
def _force_cpu_default_device():
    original = torch.get_default_device()
    torch.set_default_device("cpu")
    yield
    torch.set_default_device(original)


@dataclass(slots=True)
class _DummyVisionConfig:
    spatial_merge_size: int = 2
    tokens_per_second: float = 25.0


@dataclass(slots=True)
class _DummyConfig:
    image_token_id: int = IMAGE_TOKEN_ID
    video_token_id: int = VIDEO_TOKEN_ID
    vision_start_token_id: int = VISION_START_TOKEN_ID
    vision_end_token_id: int = VISION_END_TOKEN_ID
    vision_config: _DummyVisionConfig = dataclasses.field(
        default_factory=_DummyVisionConfig
    )


class _DummyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        # Deterministic 6-token timestamp encoding (e.g. "<12.5 seconds>")
        return [10 + (ord(ch) % 50) for ch in text[:6]]


class _DummyVisual(nn.Module):
    def __init__(self, spatial_merge_size: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.dtype = dtype


class _DummyLanguageModel(nn.Module):
    def __init__(self, hidden_size: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dtype = dtype

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        vals = (input_ids.to(torch.float32).unsqueeze(-1) * 0.01).expand(
            -1, self.hidden_size
        )
        return vals.to(dtype=self.dtype).clone()


def _make_qwen3_vl_stub(
    *,
    dtype: torch.dtype,
    pruning_method: str = "evs",
    pruning_rate: float = 0.5,
    spatial_merge_size: int = 2,
    hidden_size: int = 8,
    deepstack_levels: int = 0,
) -> Qwen3VLForConditionalGeneration:
    model = Qwen3VLForConditionalGeneration.__new__(Qwen3VLForConditionalGeneration)
    nn.Module.__init__(model)
    cfg = _DummyConfig()
    cfg.vision_config.spatial_merge_size = spatial_merge_size
    model.config = cfg
    model.visual = _DummyVisual(spatial_merge_size=spatial_merge_size, dtype=dtype)
    model.language_model = _DummyLanguageModel(hidden_size=hidden_size, dtype=dtype)
    model._tokenizer = _DummyTokenizer()
    model.is_multimodal_pruning_enabled = True
    model.video_pruning_rate = pruning_rate
    model.video_pruning_method = pruning_method
    model.use_deepstack = deepstack_levels > 0
    model.deepstack_num_level = deepstack_levels
    model.visual_dim = hidden_size
    model.multiscale_dim = hidden_size * deepstack_levels
    return model


def _make_qwen2_5_vl_stub(
    *,
    dtype: torch.dtype,
    pruning_rate: float = 0.5,
    spatial_merge_size: int = 2,
    tokens_per_second: float = 25.0,
) -> Qwen2_5_VLForConditionalGeneration:
    model = Qwen2_5_VLForConditionalGeneration.__new__(
        Qwen2_5_VLForConditionalGeneration
    )
    nn.Module.__init__(model)
    cfg = _DummyConfig()
    cfg.vision_config.spatial_merge_size = spatial_merge_size
    cfg.vision_config.tokens_per_second = tokens_per_second
    model.config = cfg
    model.visual = _DummyVisual(spatial_merge_size=spatial_merge_size, dtype=dtype)
    model.is_multimodal_pruning_enabled = True
    model.video_pruning_rate = pruning_rate
    return model


def _make_cohere_compass_stub(
    *,
    dtype: torch.dtype,
    spatial_merge_size: int = 2,
) -> CohereCompassForConditionalGeneration:
    model = CohereCompassForConditionalGeneration.__new__(
        CohereCompassForConditionalGeneration
    )
    nn.Module.__init__(model)
    cfg = _DummyConfig()
    cfg.vision_config.spatial_merge_size = spatial_merge_size
    model.config = cfg
    model.visual = _DummyVisual(spatial_merge_size=spatial_merge_size, dtype=dtype)
    model.is_multimodal_pruning_enabled = True
    return model


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("num_channels", [4, 5])
def test_pack_unpack_mrope_positions_exact_boundaries(
    dtype: torch.dtype, num_channels: int
) -> None:
    # Include bf16/fp16 rounding thresholds, 0x7F81 (bf16 sNaN bit-pattern if
    # raw int16 bitcasts were used), fp16 max (65504), and up to 16,777,215.
    critical_values = [
        0,
        1,
        255,
        256,
        257,
        259,
        511,
        512,
        513,
        1023,
        1024,
        1025,
        2047,
        2048,
        2049,
        32641,
        65504,
        65535,
        65536,
        131073,
        1_000_000,
        16_777_215,
    ]
    if torch.finfo(dtype).bits < 32:
        critical_values.append((1 << 32) - 1)

    base_col = torch.tensor(critical_values, dtype=torch.long)
    positions = torch.stack(
        [torch.roll(base_col, shifts=c) for c in range(num_channels)], dim=1
    )
    hidden = torch.randn((len(critical_values), 12), dtype=dtype)

    packed = pack_mrope_positions(positions, dtype)
    assert packed.dtype == dtype
    assert torch.isfinite(packed).all()

    combined = torch.cat([hidden, packed], dim=1)
    cleaned, unpacked = unpack_mrope_positions(combined, num_channels=num_channels)

    assert cleaned.dtype == dtype
    assert torch.equal(cleaned, hidden)
    assert unpacked.dtype == torch.long
    assert unpacked.shape == (num_channels, len(critical_values))
    assert torch.equal(unpacked, positions.permute(1, 0))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_pack_unpack_contiguous_million_range_zero_error(dtype: torch.dtype) -> None:
    n = 1_000_001
    t_col = torch.arange(n, dtype=torch.long)
    positions = torch.stack(
        [
            t_col,
            (t_col * 3) % 1_000_000,
            (t_col * 7) % 1_000_000,
            t_col & 1,
            (t_col >> 1) & 1,
        ],
        dim=1,
    )
    hidden = torch.zeros((n, 4), dtype=dtype)
    combined = torch.cat([hidden, pack_mrope_positions(positions, dtype)], dim=1)

    cleaned, unpacked = unpack_mrope_positions(combined, num_channels=5)
    assert cleaned.shape == (n, 4)
    max_err = (unpacked - positions.permute(1, 0)).abs().max().item()
    assert max_err == 0


def test_pack_unpack_empty_and_unsupported_dtype() -> None:
    empty_hidden = torch.empty((0, 8), dtype=torch.bfloat16)
    empty_pos = torch.empty((0, 5), dtype=torch.long)
    combined = torch.cat(
        [empty_hidden, pack_mrope_positions(empty_pos, torch.bfloat16)], dim=1
    )
    cleaned, unpacked = unpack_mrope_positions(combined, num_channels=5)
    assert cleaned.shape == (0, 8)
    assert unpacked.shape == (5, 0)
    assert unpacked.dtype == torch.long

    with pytest.raises(ValueError, match="Unsupported embedding dtype"):
        pack_mrope_positions(torch.zeros((2, 4), dtype=torch.long), torch.float8_e4m3fn)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("pruning_method", ["evs", "vidcom2"])
@pytest.mark.parametrize("deepstack_levels", [0, 2])
@pytest.mark.parametrize("num_frames", [64, 128])
def test_qwen3_vl_video_pruning_mrope_precision(
    dtype: torch.dtype,
    pruning_method: str,
    deepstack_levels: int,
    num_frames: int,
) -> None:
    merge_size = 2
    h, w = 8, 8
    hidden_size = 8
    pruning_rate = 0.5
    model = _make_qwen3_vl_stub(
        dtype=dtype,
        pruning_method=pruning_method,
        pruning_rate=pruning_rate,
        spatial_merge_size=merge_size,
        hidden_size=hidden_size,
        deepstack_levels=deepstack_levels,
    )

    tokens_per_frame = (h // merge_size) * (w // merge_size)
    timestamps = [0.5 * i for i in range(num_frames)]
    embed_width = hidden_size * (1 + deepstack_levels)
    gen = torch.Generator().manual_seed(1234)
    raw_video_embeds = torch.randn(
        (num_frames * tokens_per_frame, embed_width), dtype=dtype, generator=gen
    )

    mm_out = model.embed_multimodal(
        video_embeds=raw_video_embeds,
        video_grid_thw=torch.tensor([[num_frames, h, w]]),
        timestamps=[timestamps],
    )
    assert mm_out is not None and len(mm_out) == 1
    assert mm_out[0].dtype == dtype

    mask_fn = (
        vidcom2_compute_retention_mask
        if pruning_method == "vidcom2"
        else evs_compute_retention_mask
    )
    retention_mask = mask_fn(
        raw_video_embeds,
        (num_frames, h, w),
        spatial_merge_size=merge_size,
        q=pruning_rate,
    )
    num_retained = int(retention_mask.sum().item())

    # Build the lumped placeholder token sequence produced by Qwen3-VL processor
    repl_pruned = Qwen3VLMultiModalProcessor.get_video_repl(
        tokens_per_frame=[num_retained] + [0] * (num_frames - 1),
        timestamps=timestamps,
        tokenizer=model._tokenizer,
        vision_start_token_id=VISION_START_TOKEN_ID,
        vision_end_token_id=VISION_END_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
    ).full
    prefix = [1, 2, 3, 4, 5]
    suffix = [6, 7, 8, 9]
    prompt_ids = prefix + repl_pruned + suffix

    init_positions = torch.zeros((3, len(prompt_ids)), dtype=torch.long)
    init_positions[:, : len(prefix)] = (
        torch.arange(len(prefix), dtype=torch.long).unsqueeze(0).expand(3, -1)
    )

    cleaned_mm, actual_positions, actual_delta = model.recompute_mrope_positions(
        input_ids=prompt_ids,
        multimodal_embeddings=mm_out,
        mrope_positions=init_positions,
        num_computed_tokens=0,
    )
    assert len(cleaned_mm) == 1
    assert cleaned_mm[0].shape == (len(repl_pruned), embed_width)
    assert cleaned_mm[0].dtype == dtype

    # Ground-truth M-RoPE positions from unpruned sequence masked by retention_mask
    repl_unpruned = Qwen3VLMultiModalProcessor.get_video_repl(
        tokens_per_frame=[tokens_per_frame] * num_frames,
        timestamps=timestamps,
        tokenizer=model._tokenizer,
        vision_start_token_id=VISION_START_TOKEN_ID,
        vision_end_token_id=VISION_END_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
    ).full
    unpruned_prompt = prefix + repl_unpruned + suffix
    mm_feature = MultiModalFeatureSpec(
        data=MultiModalKwargsItem(
            {
                "video_grid_thw": MultiModalFieldElem(
                    data=torch.tensor([num_frames, h, w]),
                    field=None,
                )
            }
        ),
        modality="video",
        identifier="DUMMY",
        mm_position=PlaceholderRange(offset=len(prefix), length=len(repl_unpruned)),
    )
    gt_unpruned, _ = model.get_mrope_input_positions(unpruned_prompt, [mm_feature])

    unpruned_tensor = torch.tensor(unpruned_prompt, dtype=torch.long)
    keep = torch.ones_like(unpruned_tensor, dtype=torch.bool)
    video_positions = (unpruned_tensor == VIDEO_TOKEN_ID).nonzero(as_tuple=True)[0]
    keep[video_positions] = retention_mask

    expected_positions = gt_unpruned[:, keep]
    expected_delta = int(expected_positions.max().item() + 1 - len(prompt_ids))

    # Sanity-check that positions in this test genuinely exceed bf16's 256 limit.
    assert expected_positions.max().item() > 512
    assert torch.equal(actual_positions, expected_positions)
    assert actual_delta == expected_delta


def test_chunked_prefill_bf16_qwen2_5_vl_intra_media_and_qwen3_vl_multi_video() -> None:
    # 1. Qwen2.5-VL intra-video chunked prefill in bfloat16 (t_index up to 3150)
    merge_size = 2
    num_frames, h, w = 64, 8, 16
    hidden_size = 8
    q25 = _make_qwen2_5_vl_stub(
        dtype=torch.bfloat16,
        pruning_rate=0.5,
        spatial_merge_size=merge_size,
        tokens_per_second=25.0,
    )
    tpf = (h // merge_size) * (w // merge_size)
    gen = torch.Generator().manual_seed(77)
    raw_v25 = torch.randn(
        (num_frames * tpf, hidden_size), dtype=torch.bfloat16, generator=gen
    )
    mm_v25 = q25.embed_multimodal(
        video_embeds=raw_v25,
        video_grid_thw=torch.tensor([[num_frames, h, w]]),
        second_per_grid_ts=torch.tensor([2.0]),
    )
    full_mm25 = mm_v25[0]
    num_retained25 = full_mm25.shape[0]
    prefix = [1, 2, 3, 4]
    suffix = [5, 6, 7]
    prompt25 = (
        prefix
        + [VISION_START_TOKEN_ID]
        + [VIDEO_TOKEN_ID] * num_retained25
        + [VISION_END_TOKEN_ID]
        + suffix
    )
    init25 = torch.zeros((3, len(prompt25)), dtype=torch.long)
    init25[:, : len(prefix) + 1] = (
        torch.arange(len(prefix) + 1, dtype=torch.long).unsqueeze(0).expand(3, -1)
    )
    _, full_pos25, full_delta25 = q25.recompute_mrope_positions(
        input_ids=prompt25,
        multimodal_embeddings=[full_mm25],
        mrope_positions=init25.clone(),
        num_computed_tokens=0,
    )

    running_pos25 = init25.clone()
    chunk_size = 97
    cleaned_chunks25: list[torch.Tensor] = []
    chunk_delta25 = 0
    for start in range(0, num_retained25, chunk_size):
        end = min(start + chunk_size, num_retained25)
        mm_chunk = full_mm25[start:end]
        num_computed = 0 if start == 0 else len(prefix) + 1 + start
        cleaned, running_pos25, chunk_delta25 = q25.recompute_mrope_positions(
            input_ids=prompt25,
            multimodal_embeddings=[mm_chunk],
            mrope_positions=running_pos25,
            num_computed_tokens=num_computed,
        )
        cleaned_chunks25.extend(cleaned)

    assert torch.cat(cleaned_chunks25, dim=0).shape == (num_retained25, hidden_size)
    assert torch.equal(running_pos25, full_pos25)
    assert chunk_delta25 == full_delta25

    # 2. Qwen3-VL multi-video chunked prefill in bfloat16 (two 48-frame videos
    # processed in separate prefill chunks, reaching positions > 1,000).
    q3 = _make_qwen3_vl_stub(
        dtype=torch.bfloat16,
        pruning_method="evs",
        pruning_rate=0.5,
        spatial_merge_size=merge_size,
        hidden_size=hidden_size,
    )
    nf3 = 48
    ts3 = [0.5 * i for i in range(nf3)]
    raw_v3_a = torch.randn(
        (nf3 * tpf, hidden_size), dtype=torch.bfloat16, generator=gen
    )
    raw_v3_b = torch.randn(
        (nf3 * tpf, hidden_size), dtype=torch.bfloat16, generator=gen
    )
    mm_v3 = q3.embed_multimodal(
        video_embeds=torch.cat([raw_v3_a, raw_v3_b], dim=0),
        video_grid_thw=torch.tensor([[nf3, h, w], [nf3, h, w]]),
        timestamps=[ts3, ts3],
    )
    assert mm_v3 is not None and len(mm_v3) == 2

    ret_mask_a = evs_compute_retention_mask(raw_v3_a, (nf3, h, w), merge_size, 0.5)
    ret_mask_b = evs_compute_retention_mask(raw_v3_b, (nf3, h, w), merge_size, 0.5)
    ret_a = int(ret_mask_a.sum().item())
    ret_b = int(ret_mask_b.sum().item())
    repl_a = Qwen3VLMultiModalProcessor.get_video_repl(
        tokens_per_frame=[ret_a] + [0] * (nf3 - 1),
        timestamps=ts3,
        tokenizer=q3._tokenizer,
        vision_start_token_id=VISION_START_TOKEN_ID,
        vision_end_token_id=VISION_END_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
    ).full
    repl_b = Qwen3VLMultiModalProcessor.get_video_repl(
        tokens_per_frame=[ret_b] + [0] * (nf3 - 1),
        timestamps=ts3,
        tokenizer=q3._tokenizer,
        vision_start_token_id=VISION_START_TOKEN_ID,
        vision_end_token_id=VISION_END_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
    ).full
    mid_text = [21, 22, 23]
    prompt3 = prefix + repl_a + mid_text + repl_b + suffix
    init3 = torch.zeros((3, len(prompt3)), dtype=torch.long)
    init3[:, : len(prefix)] = (
        torch.arange(len(prefix), dtype=torch.long).unsqueeze(0).expand(3, -1)
    )

    # Compare bfloat16 multi-video chunked prefill against exact float64
    # reference using the identical retention masks.
    q3_fp64 = _make_qwen3_vl_stub(
        dtype=torch.float64,
        pruning_method="evs",
        pruning_rate=0.5,
        spatial_merge_size=merge_size,
        hidden_size=hidden_size,
    )
    rows, cols = h // merge_size, w // merge_size
    mm_v3_fp64 = (
        q3_fp64._create_final_video_embeddings(
            video_embeddings=raw_v3_a[ret_mask_a].to(torch.float64),
            num_tokens_per_frame=ret_mask_a.reshape(nf3, rows, cols)
            .sum(dim=(1, 2))
            .tolist(),
            timestamps=ts3,
            video_grid_thw=[nf3, h, w],
            retention_mask=ret_mask_a,
        ),
        q3_fp64._create_final_video_embeddings(
            video_embeddings=raw_v3_b[ret_mask_b].to(torch.float64),
            num_tokens_per_frame=ret_mask_b.reshape(nf3, rows, cols)
            .sum(dim=(1, 2))
            .tolist(),
            timestamps=ts3,
            video_grid_thw=[nf3, h, w],
            retention_mask=ret_mask_b,
        ),
    )

    _, ref_chunk1, _ = q3_fp64.recompute_mrope_positions(
        input_ids=prompt3,
        multimodal_embeddings=[mm_v3_fp64[0]],
        mrope_positions=init3.clone(),
        num_computed_tokens=0,
    )
    _, expected_pos3, expected_delta3 = q3_fp64.recompute_mrope_positions(
        input_ids=prompt3,
        multimodal_embeddings=[mm_v3_fp64[1]],
        mrope_positions=ref_chunk1,
        num_computed_tokens=len(prefix) + len(repl_a) + len(mid_text),
    )
    assert expected_pos3.max().item() > 1500

    # Chunk 1: prefix + video A + mid_text; Chunk 2: video B + suffix in bfloat16
    _, after_chunk1, _ = q3.recompute_mrope_positions(
        input_ids=prompt3,
        multimodal_embeddings=[mm_v3[0]],
        mrope_positions=init3.clone(),
        num_computed_tokens=0,
    )
    _, after_chunk2, chunk_delta3 = q3.recompute_mrope_positions(
        input_ids=prompt3,
        multimodal_embeddings=[mm_v3[1]],
        mrope_positions=after_chunk1,
        num_computed_tokens=len(prefix) + len(repl_a) + len(mid_text),
    )
    assert torch.equal(after_chunk2, expected_pos3)
    assert chunk_delta3 == expected_delta3


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_qwen3_vl_and_cohere_compass_high_res_image_mrope_precision(
    dtype: torch.dtype,
) -> None:
    merge_size = 2
    # Grid after merge is 260 x 302 -> both h_index and w_index exceed 256.
    grid_t, grid_h, grid_w = 1, 520, 604
    llm_h, llm_w = grid_h // merge_size, grid_w // merge_size
    num_image_tokens = llm_h * llm_w
    hidden_size = 8
    raw_image_embeds = torch.randn((num_image_tokens, hidden_size), dtype=dtype)

    prefix = [1, 2, 3]
    suffix = [4, 5, 6]
    prompt_ids = (
        prefix
        + [VISION_START_TOKEN_ID]
        + [IMAGE_TOKEN_ID] * num_image_tokens
        + [VISION_END_TOKEN_ID]
        + suffix
    )

    gt_media = compute_mrope_for_media(
        torch.tensor([grid_t, grid_h, grid_w]), merge_size
    )
    base = len(prefix) + 1  # position right after VISION_START_TOKEN_ID
    expected_img_pos = gt_media[:, :3].permute(1, 0) + base
    expected_after_offset = int(gt_media[:, :3].max().item() + base + 1)

    for model in (
        _make_qwen3_vl_stub(dtype=dtype, spatial_merge_size=merge_size),
        _make_cohere_compass_stub(dtype=dtype, spatial_merge_size=merge_size),
    ):
        mm_out = model.embed_multimodal(
            image_embeds=raw_image_embeds,
            image_grid_thw=torch.tensor([[grid_t, grid_h, grid_w]]),
        )
        assert mm_out is not None and len(mm_out) == 1

        init_positions = torch.zeros((3, len(prompt_ids)), dtype=torch.long)
        init_positions[:, : len(prefix) + 1] = (
            torch.arange(len(prefix) + 1, dtype=torch.long).unsqueeze(0).expand(3, -1)
        )
        cleaned, actual_pos, _ = model.recompute_mrope_positions(
            input_ids=prompt_ids,
            multimodal_embeddings=mm_out,
            mrope_positions=init_positions,
            num_computed_tokens=0,
        )
        assert torch.equal(cleaned[0], raw_image_embeds)
        img_slice = slice(len(prefix) + 1, len(prefix) + 1 + num_image_tokens)
        assert torch.equal(actual_pos[:, img_slice], expected_img_pos)
        expected_tail = torch.arange(
            expected_after_offset,
            expected_after_offset + 1 + len(suffix),
            dtype=torch.long,
        ).expand(3, -1)
        assert torch.equal(
            actual_pos[:, len(prefix) + 1 + num_image_tokens :], expected_tail
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_qwen2_5_vl_video_and_image_evs_mrope_precision(dtype: torch.dtype) -> None:
    merge_size = 2
    tokens_per_second = 25.0
    model = _make_qwen2_5_vl_stub(
        dtype=dtype,
        pruning_rate=0.5,
        spatial_merge_size=merge_size,
        tokens_per_second=tokens_per_second,
    )

    # 1. Video with t_index up to 63 * 50 = 3150 (> 2048 > 256) and llm_grid_w = 16
    num_frames, h, w = 64, 8, 32
    tpf = (h // merge_size) * (w // merge_size)
    hidden_size = 8
    gen = torch.Generator().manual_seed(99)
    raw_video_embeds = torch.randn(
        (num_frames * tpf, hidden_size), dtype=dtype, generator=gen
    )
    mm_video = model.embed_multimodal(
        video_embeds=raw_video_embeds,
        video_grid_thw=torch.tensor([[num_frames, h, w]]),
        second_per_grid_ts=torch.tensor([2.0]),
    )
    retention_mask = evs_compute_retention_mask(
        raw_video_embeds, (num_frames, h, w), spatial_merge_size=merge_size, q=0.5
    )
    num_retained = int(retention_mask.sum().item())

    prefix = [1, 2, 3]
    suffix = [4, 5, 6]
    video_prompt = (
        prefix
        + [VISION_START_TOKEN_ID]
        + [VIDEO_TOKEN_ID] * num_retained
        + [VISION_END_TOKEN_ID]
        + suffix
    )
    init_video_pos = torch.zeros((3, len(video_prompt)), dtype=torch.long)
    init_video_pos[:, : len(prefix) + 1] = (
        torch.arange(len(prefix) + 1, dtype=torch.long).unsqueeze(0).expand(3, -1)
    )
    cleaned_vid, actual_vid_pos, actual_vid_delta = model.recompute_mrope_positions(
        input_ids=video_prompt,
        multimodal_embeddings=mm_video,
        mrope_positions=init_video_pos,
        num_computed_tokens=0,
    )
    assert torch.equal(cleaned_vid[0], raw_video_embeds[retention_mask])

    gt_video_media = compute_mrope_for_media(
        torch.tensor([num_frames, h, w]),
        merge_size,
        tokens_per_second=tokens_per_second,
        video_second_per_grid=2,
    )[retention_mask]
    base = len(prefix) + 1
    expected_vid_slice = gt_video_media[:, :3].permute(1, 0) + base
    assert expected_vid_slice.max().item() > 3000
    assert torch.equal(
        actual_vid_pos[:, base : base + num_retained], expected_vid_slice
    )
    expected_vid_offset = int(gt_video_media[0, 3].item() + base)
    expected_vid_tail = torch.arange(
        expected_vid_offset,
        expected_vid_offset + 1 + len(suffix),
        dtype=torch.long,
    ).expand(3, -1)
    assert torch.equal(actual_vid_pos[:, base + num_retained :], expected_vid_tail)
    assert actual_vid_delta == int(actual_vid_pos.max().item() + 1 - len(video_prompt))

    # 2. Image with llm_grid_w = 302 (which rounded to 304 in raw bfloat16,
    # shifting all trailing text positions by +2) and llm_grid_h = 260 (> 256).
    img_h, img_w = 520, 604
    num_img_tokens = (img_h // merge_size) * (img_w // merge_size)
    raw_img_embeds = torch.randn((num_img_tokens, hidden_size), dtype=dtype)
    mm_img = model.embed_multimodal(
        image_embeds=raw_img_embeds,
        image_grid_thw=torch.tensor([[1, img_h, img_w]]),
    )
    img_prompt = (
        prefix
        + [VISION_START_TOKEN_ID]
        + [IMAGE_TOKEN_ID] * num_img_tokens
        + [VISION_END_TOKEN_ID]
        + suffix
    )
    init_img_pos = torch.zeros((3, len(img_prompt)), dtype=torch.long)
    init_img_pos[:, : len(prefix) + 1] = (
        torch.arange(len(prefix) + 1, dtype=torch.long).unsqueeze(0).expand(3, -1)
    )
    cleaned_img, actual_img_pos, _ = model.recompute_mrope_positions(
        input_ids=img_prompt,
        multimodal_embeddings=mm_img,
        mrope_positions=init_img_pos,
        num_computed_tokens=0,
    )
    assert torch.equal(cleaned_img[0], raw_img_embeds)
    gt_img_media = compute_mrope_for_media(torch.tensor([1, img_h, img_w]), merge_size)
    assert torch.equal(
        actual_img_pos[:, base : base + num_img_tokens],
        gt_img_media[:, :3].permute(1, 0) + base,
    )
    expected_img_offset = int(gt_img_media[0, 3].item() + base)  # 302 + 4 = 306
    expected_img_tail = torch.arange(
        expected_img_offset,
        expected_img_offset + 1 + len(suffix),
        dtype=torch.long,
    ).expand(3, -1)
    assert torch.equal(actual_img_pos[:, base + num_img_tokens :], expected_img_tail)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cuda_pack_unpack_no_host_sync_and_end_to_end(dtype: torch.dtype) -> None:
    device = torch.device("cuda")
    n = 100_000
    t_col = torch.arange(n, device=device, dtype=torch.long) * 10
    positions = torch.stack(
        [t_col, t_col % 513, t_col % 777, t_col & 1, (t_col >> 1) & 1], dim=1
    )
    hidden = torch.randn((n, 16), device=device, dtype=dtype)

    torch.cuda.synchronize()
    prev_sync_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        packed = pack_mrope_positions(positions, dtype)
        combined = torch.cat([hidden, packed], dim=1)
        cleaned, unpacked = unpack_mrope_positions(combined, num_channels=5)
    finally:
        torch.cuda.set_sync_debug_mode(prev_sync_mode)

    assert torch.equal(cleaned, hidden)
    assert torch.equal(unpacked, positions.permute(1, 0))
