# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for multi-image support in benchmark dataset samplers.

These tests are pure-Python and never instantiate an engine, model, or
network call — they monkeypatch ``self.data`` on the dataset instances and
exercise only the sampler logic.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image

from vllm.benchmarks.datasets import (
    CustomMMDataset,
    SampleRequest,
    VisionArenaDataset,
)
from vllm.benchmarks.datasets.datasets import BenchmarkDataset

# These tests are pure-Python and never initialize torch / a device
# allocator, so the autouse global-cleanup fixture in tests/conftest.py
# (which exists for tests that load models or allocate GPU memory) is
# unnecessary here. Skipping it follows the convention documented on
# `should_do_global_cleanup_after_test` in tests/conftest.py and gives a
# ~10x speedup for non-GPU unit tests.
pytestmark = [pytest.mark.benchmark, pytest.mark.skip_global_cleanup]


class _ConcreteBenchmarkDataset(BenchmarkDataset):
    """Concrete BenchmarkDataset subclass for unit-testing helpers."""

    def sample(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError


def _tiny_image() -> Image.Image:
    return Image.new("RGB", (8, 8), color=(255, 0, 0))


def _fake_tokenizer() -> Any:
    """Tokenizer stub matching the two interfaces the samplers use."""
    tok = MagicMock()
    tok.encode.side_effect = lambda s: [0] * max(len(s), 1)
    tok.return_value = MagicMock(input_ids=[0, 0, 0])
    return tok


def _make_vision_arena_dataset(items: list[dict]) -> VisionArenaDataset:
    """Build a VisionArenaDataset without triggering HF download."""
    inst = VisionArenaDataset.__new__(VisionArenaDataset)
    # Required attrs that __init__ would normally set:
    inst.dataset_path = "lmarena-ai/VisionArena-Chat"
    inst.hf_name = "lmarena-ai/VisionArena-Chat"
    inst.dataset_split = "train"
    inst.dataset_subset = None
    inst.load_stream = False
    inst.trust_remote_code = False
    inst.random_seed = 0
    inst.disable_shuffle = True
    inst.data = items
    return inst


def _make_custom_mm_dataset(items: list[dict]) -> CustomMMDataset:
    inst = CustomMMDataset.__new__(CustomMMDataset)
    inst.dataset_path = "test.jsonl"
    inst.random_seed = 0
    inst.disable_shuffle = True
    inst.data = items
    return inst


def _va_item(prompt: str, num_images: int) -> dict:
    """Build a VisionArena-Chat-shaped item with N images."""
    return {
        "conversation": [[{"content": prompt, "role": "user"}]],
        "images": [_tiny_image() for _ in range(num_images)],
    }


def _custom_mm_item(prompt: str, num_images: int) -> dict:
    return {
        "prompt": prompt,
        # CustomMMDataset accepts string paths/URLs; use data: URLs to skip I/O.
        "image_files": ["data:image/jpeg;base64,/9j/AAA=" for _ in range(num_images)],
    }


@pytest.mark.benchmark
def test_vision_arena_single_image_yields_dict() -> None:
    """Default cap=1 + single-image item → mm_content is a dict (back-compat)."""
    ds = _make_vision_arena_dataset([_va_item("hello", 1)])
    out = ds.sample(tokenizer=_fake_tokenizer(), num_requests=1, output_len=8)
    assert len(out) == 1
    assert isinstance(out[0].multi_modal_data, dict)
    assert out[0].multi_modal_data["type"] == "image_url"


@pytest.mark.benchmark
def test_vision_arena_multi_image_capped_to_one_by_default() -> None:
    """Without limit_mm_per_prompt, a 3-image item still produces a single dict."""
    ds = _make_vision_arena_dataset([_va_item("hello", 3)])
    out = ds.sample(tokenizer=_fake_tokenizer(), num_requests=1, output_len=8)
    assert isinstance(out[0].multi_modal_data, dict)


@pytest.mark.benchmark
def test_vision_arena_multi_image_emits_list_when_cap_raised() -> None:
    """With limit_mm_per_prompt={'image': 4}, a 3-image item → list of 3 dicts."""
    ds = _make_vision_arena_dataset([_va_item("hello", 3)])
    out = ds.sample(
        tokenizer=_fake_tokenizer(),
        num_requests=1,
        output_len=8,
        limit_mm_per_prompt={"image": 4},
    )
    mm = out[0].multi_modal_data
    assert isinstance(mm, list)
    assert len(mm) == 3
    assert all(isinstance(item, dict) for item in mm)
    assert all(item["type"] == "image_url" for item in mm)


@pytest.mark.benchmark
def test_vision_arena_cap_truncates_excess_images() -> None:
    """Cap of 2 against a 5-image item yields a list of exactly 2."""
    ds = _make_vision_arena_dataset([_va_item("hello", 5)])
    out = ds.sample(
        tokenizer=_fake_tokenizer(),
        num_requests=1,
        output_len=8,
        limit_mm_per_prompt={"image": 2},
    )
    mm = out[0].multi_modal_data
    assert isinstance(mm, list)
    assert len(mm) == 2


@pytest.mark.benchmark
def test_vision_arena_cap_one_collapses_to_dict() -> None:
    """Even with explicit cap=1, the result is a dict (not a 1-element list)."""
    ds = _make_vision_arena_dataset([_va_item("hello", 3)])
    out = ds.sample(
        tokenizer=_fake_tokenizer(),
        num_requests=1,
        output_len=8,
        limit_mm_per_prompt={"image": 1},
    )
    assert isinstance(out[0].multi_modal_data, dict)


@pytest.mark.benchmark
def test_vision_arena_cap_zero_yields_no_mm_content() -> None:
    """A cap of 0 disables image attachment; mm_content is None."""
    ds = _make_vision_arena_dataset([_va_item("hello", 3)])
    out = ds.sample(
        tokenizer=_fake_tokenizer(),
        num_requests=1,
        output_len=8,
        limit_mm_per_prompt={"image": 0},
    )
    assert out[0].multi_modal_data is None


@pytest.mark.benchmark
def test_custom_mm_multi_image_emits_list() -> None:
    """CustomMMDataset emits list[dict] for >1 image when cap allows it."""
    ds = _make_custom_mm_dataset([_custom_mm_item("describe the images", 3)])
    out = ds.sample(
        tokenizer=_fake_tokenizer(),
        num_requests=1,
        output_len=8,
        limit_mm_per_prompt={"image": 8},
    )
    mm = out[0].multi_modal_data
    assert isinstance(mm, list)
    assert len(mm) == 3


@pytest.mark.benchmark
def test_custom_mm_default_cap_preserves_single_image_shape() -> None:
    """CustomMMDataset default cap=1 collapses multi-image input to a dict."""
    ds = _make_custom_mm_dataset([_custom_mm_item("describe", 4)])
    out = ds.sample(tokenizer=_fake_tokenizer(), num_requests=1, output_len=8)
    assert isinstance(out[0].multi_modal_data, dict)


@pytest.mark.benchmark
def test_sample_request_type_contract_is_respected() -> None:
    """Every emitted multi_modal_data is None, dict, or list[dict]."""
    ds = _make_vision_arena_dataset(
        [
            _va_item("p1", 1),
            _va_item("p2", 2),
            _va_item("p3", 4),
        ]
    )
    out = ds.sample(
        tokenizer=_fake_tokenizer(),
        num_requests=3,
        output_len=8,
        limit_mm_per_prompt={"image": 3},
    )
    for req in out:
        mm = req.multi_modal_data
        assert (
            mm is None
            or isinstance(mm, dict)
            or (isinstance(mm, list) and all(isinstance(x, dict) for x in mm))
        )


@pytest.mark.benchmark
def test_chat_transformation_round_trip_with_list() -> None:
    """apply_multimodal_chat_transformation correctly unfolds a list of mm dicts."""
    base = _ConcreteBenchmarkDataset.__new__(_ConcreteBenchmarkDataset)
    mm_list = [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,A"}},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,B"}},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,C"}},
    ]
    chat = base.apply_multimodal_chat_transformation("describe the images", mm_list)
    assert isinstance(chat, list)
    assert len(chat) == 1
    assert chat[0]["role"] == "user"
    content = chat[0]["content"]
    # 1 text block + 3 image blocks, in order.
    assert len(content) == 4
    assert content[0] == {"text": "describe the images", "type": "text"}
    assert content[1:] == mm_list


@pytest.mark.benchmark
def test_throughput_prompt_builder_accepts_list_form() -> None:
    """Mirror the throughput.py prompt-build assertion for the list-form path.

    The on-engine code at vllm/benchmarks/throughput.py does:
        if request.multi_modal_data:
            assert isinstance(request.multi_modal_data, (dict, list))
            prompt["multi_modal_data"] = request.multi_modal_data
    This test re-runs that check against a synthetic list-form SampleRequest.
    """
    req = SampleRequest(
        prompt="x",
        prompt_len=1,
        expected_output_len=1,
        multi_modal_data=[
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,A"}},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,B"}},
        ],
    )
    if req.multi_modal_data:
        assert isinstance(req.multi_modal_data, (dict, list))
    # Confirm dict form still passes.
    req_dict = SampleRequest(
        prompt="x",
        prompt_len=1,
        expected_output_len=1,
        multi_modal_data={
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,A"},
        },
    )
    if req_dict.multi_modal_data:
        assert isinstance(req_dict.multi_modal_data, (dict, list))
