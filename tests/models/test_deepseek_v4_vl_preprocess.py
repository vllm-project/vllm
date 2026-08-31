# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity tests for the DeepSeek-V4 VL multimodal preprocessing
(``vllm/models/deepseek_v4/common/mm_preprocess.py``) against the official
reference implementation (``image_processor.py`` from the HF repo)."""

import importlib.util
import io
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from vllm.models.deepseek_v4.common import mm_preprocess as ours
from vllm.models.deepseek_v4.common.mm_preprocess import (
    COMPRESS_PAD_TO,
    IMAGE,
    IMAGE_END,
    IMAGE_PAD,
    IMAGE_START,
    DeepseekV4VLMultiModalProcessor,
    DeepseekV4VLProcessor,
)
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.processing import PromptReplacement, PromptUpdateDetails
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config

REF_IMAGE_PROCESSOR_PATH = "/tmp/dsv4vis/image_processor.py"


def _load_reference_image_processor():
    spec = importlib.util.spec_from_file_location(
        "dsv4vis_ref_image_processor", REF_IMAGE_PROCESSOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pytestmark = pytest.mark.skipif(
    not os.path.exists(REF_IMAGE_PROCESSOR_PATH),
    reason="reference image_processor.py not available",
)

if os.path.exists(REF_IMAGE_PROCESSOR_PATH):
    ref = _load_reference_image_processor()
else:
    ref = None

PATCH_SIZE = 14
DOWNSAMPLE_RATIO = 3
MAX_N_TOKEN = 384

REF_ARGS = SimpleNamespace(
    vision_patch_size=PATCH_SIZE,
    vision_downsample_ratio=DOWNSAMPLE_RATIO,
    vision_max_n_token=MAX_N_TOKEN,
    vision_min_pixels=147456,
    vision_max_wh_ratio=8,
)

OURS_KWARGS = dict(
    patch_size=PATCH_SIZE,
    downsample_ratio=DOWNSAMPLE_RATIO,
    max_n_token=MAX_N_TOKEN,
    min_pixels=147456,
    max_wh_ratio=8,
)


def _run(fn, *args):
    """Run ``fn`` capturing AssertionError so failure parity is also checked."""
    try:
        return ("ok", fn(*args))
    except AssertionError:
        return ("assert", None)


@pytest.mark.parametrize(
    "height,width",
    [
        (h, w)
        for h in (14, 15, 28, 41, 100, 410, 756)
        for w in (14, 30, 42, 100, 512, 756)
    ],
)
def test_grid_tokens_parity(height: int, width: int):
    expected = ref.grid_tokens(height, width, PATCH_SIZE, DOWNSAMPLE_RATIO)
    actual = ours.grid_tokens(height, width, PATCH_SIZE, DOWNSAMPLE_RATIO)
    assert actual == expected


@pytest.mark.parametrize("max_n_token", [6, 32, 128, MAX_N_TOKEN, 1024])
@pytest.mark.parametrize(
    "height,width", [(50, 3000), (3000, 50), (137, 400), (756, 756), (100, 100)]
)
def test_solve_resize_ratio_parity(height: int, width: int, max_n_token: int):
    expected = _run(
        ref.solve_resize_ratio,
        height,
        width,
        PATCH_SIZE,
        DOWNSAMPLE_RATIO,
        max_n_token,
    )
    actual = _run(
        ours.solve_resize_ratio,
        height,
        width,
        PATCH_SIZE,
        DOWNSAMPLE_RATIO,
        max_n_token,
    )
    assert actual == expected


@pytest.mark.parametrize("max_n_token", [32, 128, MAX_N_TOKEN, 1024])
@pytest.mark.parametrize(
    "height,width", [(50, 3000), (3000, 50), (137, 400), (756, 756)]
)
def test_safe_resize_parity(height: int, width: int, max_n_token: int):
    best_h = -(-height // PATCH_SIZE) * PATCH_SIZE
    best_w = -(-width // PATCH_SIZE) * PATCH_SIZE
    expected = ref.safe_resize(
        height, width, best_h, best_w, PATCH_SIZE, DOWNSAMPLE_RATIO, max_n_token
    )
    actual = ours.safe_resize(
        height, width, best_h, best_w, PATCH_SIZE, DOWNSAMPLE_RATIO, max_n_token
    )
    assert actual == expected


@pytest.mark.parametrize("start_pos", range(9))
@pytest.mark.parametrize(
    "n_llm_h,n_llm_w", [(h, w) for h in range(1, 9) for w in range(1, 9)]
)
def test_build_image_block_parity(n_llm_h: int, n_llm_w: int, start_pos: int):
    ref_types, ref_perm = ref.build_image_block(n_llm_h, n_llm_w, start_pos)
    types, perm = ours.build_image_block(n_llm_h, n_llm_w, start_pos)
    assert torch.equal(types, ref_types)
    assert torch.equal(perm, ref_perm)


@pytest.mark.parametrize(
    "width,height",
    [(800, 600), (100, 2000), (2000, 100), (50, 50), (13, 7), (384, 384)],
)
def test_load_image_parity(width: int, height: int):
    rng = np.random.default_rng(width * 10000 + height)
    array = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    image = Image.fromarray(array)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    record = {"data": buf.getvalue()}

    ref_out = ref.load_image(record, REF_ARGS)
    our_out = ours.load_image(image, **OURS_KWARGS)

    assert our_out[1:] == ref_out[1:]
    assert torch.equal(our_out[0], ref_out[0])


@pytest.mark.parametrize("start_pos", range(8))
@pytest.mark.parametrize("n_llm_h,n_llm_w", [(1, 1), (2, 3), (3, 2), (5, 4)])
def test_block_semantics(n_llm_h: int, n_llm_w: int, start_pos: int):
    types, perm = ours.build_image_block(n_llm_h, n_llm_w, start_pos)

    # The grid occupies the pad-free part of the block.
    _, _, num_tokens = ref.grid_tokens(
        n_llm_h * PATCH_SIZE * DOWNSAMPLE_RATIO,
        n_llm_w * PATCH_SIZE * DOWNSAMPLE_RATIO,
        PATCH_SIZE,
        DOWNSAMPLE_RATIO,
    )
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    assert len(types) == num_tokens + compress_pad
    assert (types == IMAGE_PAD).sum() >= compress_pad
    assert (types[:compress_pad] == IMAGE_PAD).all()
    assert types[compress_pad] == IMAGE_START
    assert types[-1] == IMAGE_END
    assert (types == IMAGE).sum() == n_llm_h * n_llm_w
    assert len(perm) == n_llm_h * n_llm_w
    # ``perm`` selects every IMAGE slot exactly once.
    assert torch.equal(perm.sort().values, torch.arange(n_llm_h * n_llm_w))

    # The is_embed mask must mark exactly the IMAGE positions.
    is_embed = types == IMAGE
    assert is_embed.sum() == n_llm_h * n_llm_w
    assert not is_embed[:compress_pad].any()


@pytest.mark.parametrize("n_llm_h,n_llm_w", [(1, 1), (3, 2), (4, 5)])
def test_pad_free_block(n_llm_h: int, n_llm_w: int):
    types, perm = ours.build_image_block_pad_free(n_llm_h, n_llm_w)
    ref_types, ref_perm = ours.build_image_block(n_llm_h, n_llm_w, start_pos=0)
    compress_pad = COMPRESS_PAD_TO - 1
    assert torch.equal(types, ref_types[compress_pad:])
    assert torch.equal(perm, ref_perm)
    assert types[0] == IMAGE_START and types[-1] == IMAGE_END


def test_processor_output():
    config = DeepseekV4Config(vocab_size=129280)
    processor = DeepseekV4VLProcessor(config)
    images = [
        Image.new("RGB", (800, 600), color=(10, 20, 30)),
        Image.new("RGB", (100, 2000), color=(200, 100, 50)),
    ]
    out = processor(images=images)

    vit_grid = out["vit_grid"]
    llm_grid = out["llm_grid"]
    assert vit_grid.shape == llm_grid.shape == (2, 2)
    assert out["patches"].shape[0] == vit_grid.prod(-1).sum()
    assert out["patches"].dtype == torch.bfloat16
    assert out["patches"].shape[1:] == (3, PATCH_SIZE, PATCH_SIZE)
    assert out["perm"].shape[0] == llm_grid.prod(-1).sum()

    # Per-image pad-free types: [IMAGE_START, ..., IMAGE_END], with
    # n_llm_h * n_llm_w IMAGE entries.
    offset = 0
    for i in range(2):
        n_llm_h, n_llm_w = llm_grid[i].tolist()
        _, _, num_tokens = ref.grid_tokens(
            n_llm_h * PATCH_SIZE * DOWNSAMPLE_RATIO,
            n_llm_w * PATCH_SIZE * DOWNSAMPLE_RATIO,
            PATCH_SIZE,
            DOWNSAMPLE_RATIO,
        )
        types = out["types"][offset : offset + num_tokens]
        offset += num_tokens
        assert types[0] == IMAGE_START and types[-1] == IMAGE_END
        assert (types == IMAGE).sum() == n_llm_h * n_llm_w
    assert offset == out["types"].shape[0]

    assert processor(images=[]) == {}


class _StubInfo:
    """Minimum ``ProcessingInfo`` surface for the placeholder-splicing test."""

    def __init__(self, vocab_size: int) -> None:
        self._config = SimpleNamespace(vocab_size=vocab_size)

    def get_hf_config(self):
        return self._config

    def get_data_parser(self):
        return MultiModalDataParser()


def test_apply_token_matches_adds_compress_pad():
    vocab_size = 1000
    image_token_id = 7
    n_llm_h, n_llm_w = 3, 2

    processor = DeepseekV4VLMultiModalProcessor(_StubInfo(vocab_size), None)

    types, _ = ours.build_image_block_pad_free(n_llm_h, n_llm_w)
    full = (vocab_size + types).tolist()
    update = PromptReplacement(
        modality="image",
        target=[image_token_id],
        replacement=PromptUpdateDetails.select_token_id(full, vocab_size + IMAGE),
    )
    prompt = [11, 12, image_token_id, 13, 14, 15, image_token_id, 16]
    mm_prompt_updates = {"image": [[update.resolve(0)], [update.resolve(1)]]}

    new_token_ids, match_result, placeholders = (
        processor._apply_token_matches_with_placeholders(prompt, mm_prompt_updates)
    )
    assert match_result == {"image": [0, 0]}

    # Reference construction: pads computed from the final position of each
    # block via ``build_image_block``'s ``start_pos``.
    ref_ids = [11, 12]
    ref_types, _ = ref.build_image_block(n_llm_h, n_llm_w, len(ref_ids))
    ref_ids += (vocab_size + ref_types).tolist()
    ref_ids += [13, 14, 15]
    ref_types, _ = ref.build_image_block(n_llm_h, n_llm_w, len(ref_ids))
    ref_ids += (vocab_size + ref_types).tolist()
    ref_ids += [16]
    assert new_token_ids == ref_ids

    image_placeholders = placeholders["image"]
    assert len(image_placeholders) == 2
    for placeholder in image_placeholders:
        compress_pad = COMPRESS_PAD_TO - 1 - placeholder.start_idx % COMPRESS_PAD_TO
        assert len(placeholder.tokens) == len(full) + compress_pad
        assert placeholder.tokens[:compress_pad] == [vocab_size + IMAGE_PAD] * (
            compress_pad
        )
        assert placeholder.tokens[compress_pad:] == full
        assert (
            new_token_ids[
                placeholder.start_idx : placeholder.start_idx + len(placeholder.tokens)
            ]
            == placeholder.tokens
        )
        is_embed = placeholder.is_embed
        assert is_embed.sum() == n_llm_h * n_llm_w
        assert not is_embed[:compress_pad].any()
        assert is_embed.tolist() == [
            token == vocab_size + IMAGE for token in placeholder.tokens
        ]
