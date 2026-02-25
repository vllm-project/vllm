# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression test for https://github.com/vllm-project/vllm/issues/29863

When a Qwen3-VL prompt ends with punctuation (e.g., `.` or `:`) immediately
before a video placeholder, the HF processor expands the placeholder into
timestamp markers like `<0.3 seconds><|vision_start|>...`.  The `<` in the
timestamp text merges with the preceding punctuation (e.g., `.<` becomes a
single token), breaking exact token matching in `_find_mm_placeholders`.

This test exercises `_find_mm_placeholders` directly with token IDs that
contain the boundary merge, verifying the Qwen3VL-specific override
correctly splits merged tokens before matching.
"""

import pytest

from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor,
    PromptUpdateDetails,
    ResolvedPromptUpdate,
    UpdateMode,
)

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
NUM_PADS = 4

pytestmark = pytest.mark.cpu_test


# ── fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def processor():
    model_config = ModelConfig(
        model=MODEL,
        limit_mm_per_prompt={"video": 1},
    )
    return MULTIMODAL_REGISTRY.create_processor(model_config)


@pytest.fixture(scope="module")
def tokenizer(processor):
    return processor.info.get_tokenizer()


@pytest.fixture(scope="module")
def vision_ids(processor):
    """Return (vision_start_id, video_pad_id, vision_end_id)."""
    cfg = processor.info.get_hf_config()
    return cfg.vision_start_token_id, cfg.video_token_id, cfg.vision_end_token_id


# ── helpers ──────────────────────────────────────────────────────────


def _make_video_update(tokenizer, vision_ids):
    """Build video replacement tokens and the corresponding prompt update."""
    vision_start_id, video_pad_id, vision_end_id = vision_ids
    timestamp_tokens = tokenizer.encode("<0.0 seconds>", add_special_tokens=False)
    video_replacement = (
        timestamp_tokens
        + [vision_start_id]
        + [video_pad_id] * NUM_PADS
        + [vision_end_id]
    )
    content = PromptUpdateDetails.select_token_id(video_replacement, video_pad_id)
    update = ResolvedPromptUpdate(
        modality="video",
        item_idx=0,
        mode=UpdateMode.REPLACE,
        target="<|vision_start|><|video_pad|><|vision_end|>",
        content=content,
    )
    return timestamp_tokens, video_replacement, {"video": [[update]]}


# ── tests ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("merge_pattern", [".<", ":<"], ids=["dot", "colon"])
def test_boundary_merge(processor, tokenizer, vision_ids, merge_pattern):
    """
    When punctuation merges with `<` (e.g. `.<` or `:<` become one token),
    the Qwen3VL override should still find the video placeholder.
    """
    vision_start_id, video_pad_id, vision_end_id = vision_ids

    # Verify the merge actually happens with this tokenizer
    merged_ids = tokenizer.encode(merge_pattern, add_special_tokens=False)
    assert len(merged_ids) == 1, (
        f"Expected '{merge_pattern}' to merge into 1 token, got {len(merged_ids)}"
    )
    merged_token = merged_ids[0]

    timestamp_tokens, video_replacement, mm_prompt_updates = _make_video_update(
        tokenizer, vision_ids
    )

    # Simulate HF processor output: "details" + merged_token + rest of
    # timestamp + vision_start + pad*N + vision_end
    prompt_ids = (
        tokenizer.encode("details", add_special_tokens=False)
        + [merged_token]
        + timestamp_tokens[1:]
        + [vision_start_id]
        + [video_pad_id] * NUM_PADS
        + [vision_end_id]
    )

    # Qwen3VL override should find the placeholder
    result = processor._find_mm_placeholders(prompt_ids, mm_prompt_updates)
    assert "video" in result
    assert len(result["video"]) == 1
    ph = result["video"][0]
    assert ph.item_idx == 0
    assert ph.modality == "video"
    assert len(ph.tokens) == len(video_replacement)

    # Base class should fail — proves the override is needed
    base_result = BaseMultiModalProcessor._find_mm_placeholders(
        processor, prompt_ids, mm_prompt_updates
    )
    assert base_result.get("video", []) == [], (
        "Base _find_mm_placeholders should NOT find the merged placeholder"
    )


def test_no_merge(processor, tokenizer, vision_ids):
    """When there's no merge (e.g., newline before `<`), should still work."""
    vision_start_id, video_pad_id, vision_end_id = vision_ids

    timestamp_tokens, _, mm_prompt_updates = _make_video_update(tokenizer, vision_ids)

    # No merge: newline before '<' keeps it as a separate token
    prompt_ids = (
        tokenizer.encode("details\n", add_special_tokens=False)
        + timestamp_tokens
        + [vision_start_id]
        + [video_pad_id] * NUM_PADS
        + [vision_end_id]
    )

    result = processor._find_mm_placeholders(prompt_ids, mm_prompt_updates)
    assert "video" in result
    assert len(result["video"]) == 1
    ph = result["video"][0]
    assert ph.item_idx == 0
    assert ph.modality == "video"
