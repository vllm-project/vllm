# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random
from typing import Any, NamedTuple, cast

import numpy as np
import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.benchmarks.datasets import (
    RandomDataset,
    RandomMultiModalDataset,
    SampleRequest,
)


@pytest.fixture(scope="session")
def hf_tokenizer() -> PreTrainedTokenizerBase:
    # Use a small, commonly available tokenizer
    return AutoTokenizer.from_pretrained("gpt2")


class Params(NamedTuple):
    num_requests: int
    prefix_len: int
    range_ratio: float
    input_len: int
    output_len: int


@pytest.fixture(scope="session")
def random_dataset_params() -> Params:
    return Params(
        num_requests=16, prefix_len=7, range_ratio=0.3, input_len=50, output_len=20
    )


def _fingerprint_sample(req: SampleRequest) -> tuple[str, int, int]:
    """Project a SampleRequest into a comparable tuple."""
    return (req.prompt, req.prompt_len, req.expected_output_len)


def _collect_samples(
    dataset: RandomDataset,
    tokenizer: PreTrainedTokenizerBase,
    num_requests: int = 16,
    prefix_len: int = 7,
    range_ratio: float = 0.3,
    input_len: int = 50,
    output_len: int = 20,
) -> list[tuple[str, int, int]]:
    samples = dataset.sample(
        tokenizer=tokenizer,
        num_requests=num_requests,
        prefix_len=prefix_len,
        range_ratio=range_ratio,
        input_len=input_len,
        output_len=output_len,
    )
    return [_fingerprint_sample(s) for s in samples]


@pytest.mark.benchmark
def test_random_dataset_same_seed(
    hf_tokenizer: PreTrainedTokenizerBase, random_dataset_params: Params
) -> None:
    """Same seed should yield identical outputs, even if global RNGs change.

    This guards against accidental reliance on Python's random or np.random
    in RandomDataset after moving to numpy.default_rng.
    """
    p = random_dataset_params
    common_seed = 123
    dataset_a = RandomDataset(random_seed=common_seed)
    dataset_b = RandomDataset(random_seed=common_seed)
    a = _collect_samples(
        dataset_a,
        hf_tokenizer,
        num_requests=p.num_requests,
        prefix_len=p.prefix_len,
        range_ratio=p.range_ratio,
        input_len=p.input_len,
        output_len=p.output_len,
    )

    # Perturb global RNG state to ensure isolation
    random.seed(999)
    _ = [random.random() for _ in range(100)]
    np.random.seed(888)
    _ = [np.random.random() for _ in range(100)]

    b = _collect_samples(
        dataset_b,
        hf_tokenizer,
        num_requests=p.num_requests,
        prefix_len=p.prefix_len,
        range_ratio=p.range_ratio,
        input_len=p.input_len,
        output_len=p.output_len,
    )
    assert a == b


@pytest.mark.benchmark
def test_random_dataset_different_seeds(
    hf_tokenizer: PreTrainedTokenizerBase, random_dataset_params: Params
) -> None:
    """Different seeds should change outputs with overwhelming likelihood."""
    p = random_dataset_params
    seed_a = 0
    dataset_a = RandomDataset(random_seed=seed_a)
    a = _collect_samples(
        dataset_a,
        hf_tokenizer,
        num_requests=p.num_requests,
        prefix_len=p.prefix_len,
        range_ratio=p.range_ratio,
        input_len=p.input_len,
        output_len=p.output_len,
    )

    seed_b = 999
    dataset_b = RandomDataset(random_seed=seed_b)
    # Perturb global RNG with same seed as dataset_a to ensure isolation
    random.seed(seed_a)
    np.random.seed(seed_a)
    b = _collect_samples(
        dataset_b,
        hf_tokenizer,
        num_requests=p.num_requests,
        prefix_len=p.prefix_len,
        range_ratio=p.range_ratio,
        input_len=p.input_len,
        output_len=p.output_len,
    )
    assert a != b


# -----------------------------
# RandomMultiModalDataset tests
# -----------------------------


def _mm_fingerprint_sample(
    req: SampleRequest,
) -> tuple[str, int, int, int, list[str]]:
    """Create a compact fingerprint for multimodal samples.

    Includes:
    - prompt string
    - prompt_len
    - expected_output_len
    - count of multimodal items
    - per-item type and URL prefix (e.g., 'data:image/jpeg;base64,')
    """
    items = req.multi_modal_data or []
    item_prefixes: list[str] = []
    for it in items:
        if isinstance(it, dict) and it.get("type") == "image_url":
            url = it.get("image_url", {}).get("url", "")
            # Only keep a short identifying prefix to avoid huge strings
            item_prefixes.append(f"image:{url[:22]}")
        elif isinstance(it, dict) and it.get("type") == "video_url":
            url = it.get("video_url", {}).get("url", "")
            item_prefixes.append(f"video:{url[:22]}")
        else:
            item_prefixes.append("unknown:")
    return (
        req.prompt,
        req.prompt_len,
        req.expected_output_len,
        len(items),
        item_prefixes,
    )


def _collect_mm_samples(
    dataset: RandomMultiModalDataset,
    tokenizer: PreTrainedTokenizerBase,
    *,
    num_requests: int = 8,
    prefix_len: int = 3,
    range_ratio: float = 0.0,
    input_len: int = 20,
    output_len: int = 5,
    base_items_per_request: int = 2,
    num_mm_items_range_ratio: float = 0.0,
    limit_mm_per_prompt: dict[str, int] | None = None,
    bucket_config: dict[tuple[int, int, int], float] | None = None,
    enable_multimodal_chat: bool = False,
) -> list[SampleRequest]:
    if limit_mm_per_prompt is None:
        limit_mm_per_prompt = {"image": 5, "video": 0}
    if bucket_config is None:
        bucket_config = {(32, 32, 1): 0.5, (52, 64, 1): 0.5}
    return dataset.sample(
        tokenizer=tokenizer,
        num_requests=num_requests,
        prefix_len=prefix_len,
        range_ratio=range_ratio,
        input_len=input_len,
        output_len=output_len,
        base_items_per_request=base_items_per_request,
        num_mm_items_range_ratio=num_mm_items_range_ratio,
        limit_mm_per_prompt=limit_mm_per_prompt,
        bucket_config=bucket_config,
        enable_multimodal_chat=enable_multimodal_chat,
    )


@pytest.mark.benchmark
def test_random_mm_same_seed(hf_tokenizer: PreTrainedTokenizerBase) -> None:
    seed = 42
    ds_a = RandomMultiModalDataset(random_seed=seed)
    ds_b = RandomMultiModalDataset(random_seed=seed)
    a = _collect_mm_samples(ds_a, hf_tokenizer)
    b = _collect_mm_samples(ds_b, hf_tokenizer)
    fa = [_mm_fingerprint_sample(s) for s in a]
    fb = [_mm_fingerprint_sample(s) for s in b]
    assert fa == fb


@pytest.mark.benchmark
def test_random_mm_different_seeds(
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    ds_a = RandomMultiModalDataset(random_seed=0)
    ds_b = RandomMultiModalDataset(random_seed=999)
    a = _collect_mm_samples(ds_a, hf_tokenizer)
    b = _collect_mm_samples(ds_b, hf_tokenizer)
    fa = [_mm_fingerprint_sample(s) for s in a]
    fb = [_mm_fingerprint_sample(s) for s in b]
    assert fa != fb


@pytest.mark.benchmark
def test_random_mm_respects_limits(
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    ds = RandomMultiModalDataset(random_seed=0)
    # Requesting 3 items with a per-prompt limit of 1 should error per current
    # design (dataset refuses to silently clamp below the requested baseline).
    with pytest.raises(ValueError):
        _collect_mm_samples(
            ds,
            hf_tokenizer,
            num_requests=12,
            base_items_per_request=3,
            num_mm_items_range_ratio=0.0,
            limit_mm_per_prompt={"image": 1, "video": 0},
            bucket_config={(32, 32, 1): 1.0},
        )


@pytest.mark.benchmark
def test_random_mm_zero_prob_entries_are_removed(
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    ds = RandomMultiModalDataset(random_seed=0)
    # Second bucket has zero probability and should be ignored after
    # normalization
    samples = _collect_mm_samples(
        ds,
        hf_tokenizer,
        num_requests=6,
        base_items_per_request=2,
        num_mm_items_range_ratio=0.0,
        limit_mm_per_prompt={"image": 10, "video": 0},
        bucket_config={(32, 32, 1): 1.0, (52, 64, 1): 0.0},
    )
    for s in samples:
        assert isinstance(s.multi_modal_data, list)
        typed_mm = cast(list[dict[str, Any]], s.multi_modal_data)
        for it in typed_mm:
            assert it.get("type") == "image_url"


@pytest.mark.benchmark
def test_random_mm_zero_items(hf_tokenizer: PreTrainedTokenizerBase) -> None:
    ds = RandomMultiModalDataset(random_seed=0)
    samples = _collect_mm_samples(
        ds,
        hf_tokenizer,
        num_requests=5,
        base_items_per_request=0,
        num_mm_items_range_ratio=0.0,
        limit_mm_per_prompt={"image": 5, "video": 0},
        bucket_config={(32, 32, 1): 1.0},
    )
    for s in samples:
        assert s.multi_modal_data == []


@pytest.mark.benchmark
def test_random_mm_num_items_per_prompt(hf_tokenizer: PreTrainedTokenizerBase) -> None:
    ds = RandomMultiModalDataset(random_seed=0)
    # Fixed number of images per prompt
    # set num_mm_items_range_ratio to 0.0
    # TODO: modify video values when video sampling is implemented
    samples_fixed_items = _collect_mm_samples(
        ds,
        hf_tokenizer,
        num_requests=5,
        base_items_per_request=3,
        num_mm_items_range_ratio=0.0,
        limit_mm_per_prompt={"image": 3, "video": 0},
        bucket_config={(32, 32, 1): 1.0},
    )
    # Must have 5 requests each with 3 mm items per prompt
    assert len(samples_fixed_items) == 5
    for s in samples_fixed_items:
        mm_data = cast(list[dict[str, Any]], s.multi_modal_data)
        assert len(mm_data) == 3
        for it in mm_data:
            assert it.get("type") == "image_url"


@pytest.mark.benchmark
def test_random_mm_bucket_config_not_mutated(
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    ds = RandomMultiModalDataset(random_seed=0)
    # This bucket config is not normalized to sum to 1
    # and has more buckets than requested images
    original = {(32, 32, 1): 0.2, (52, 64, 1): 6, (25, 64, 1): 3}
    # Keep a snapshot to compare after sampling
    snapshot = dict(original)

    _ = _collect_mm_samples(
        ds,
        hf_tokenizer,
        num_requests=4,
        base_items_per_request=1,
        num_mm_items_range_ratio=0.0,
        limit_mm_per_prompt={"image": 1, "video": 0},
        bucket_config=original,
    )

    # Ensure the original dict content is unchanged
    assert original == snapshot

    # Vary number of mm items per prompt
    # set num_mm_items_range_ratio to 0.5
    samples_varying_items = _collect_mm_samples(
        ds,
        hf_tokenizer,
        num_requests=5,
        base_items_per_request=2,
        num_mm_items_range_ratio=0.5,
        limit_mm_per_prompt={"image": 4, "video": 0},
        bucket_config={(32, 32, 1): 1.0},
    )
    # Must have 5 requests each with less than 4 mm items per prompt
    # but at least 1 mm item per prompt
    assert len(samples_varying_items) == 5
    for s in samples_varying_items:
        mm_data = cast(list[dict[str, Any]], s.multi_modal_data)
        assert len(mm_data) <= 4
        assert len(mm_data) >= 1
        for it in mm_data:
            assert it.get("type") == "image_url"


@pytest.mark.benchmark
def test_random_mm_video_sampling(hf_tokenizer: PreTrainedTokenizerBase) -> None:
    """Test video sampling functionality in RandomMultiModalDataset."""
    ds = RandomMultiModalDataset(random_seed=42)

    # Test with video bucket configuration
    bucket_config = {
        (64, 64, 1): 0.3,  # Images
        (64, 64, 8): 0.7,  # Videos
    }

    limit_mm_per_prompt = {"image": 2, "video": 2}

    samples = _collect_mm_samples(
        ds,
        hf_tokenizer,
        num_requests=5,
        base_items_per_request=1,
        num_mm_items_range_ratio=0.0,
        limit_mm_per_prompt=limit_mm_per_prompt,
        bucket_config=bucket_config,
    )

    assert len(samples) == 5

    # Check that we have both images and videos
    video_count = 0
    image_count = 0

    for s in samples:
        mm_data = cast(list[dict[str, Any]], s.multi_modal_data)
        assert len(mm_data) == 1

        item = mm_data[0]
        if item.get("type") == "video_url":
            video_count += 1
            # Verify video URL format
            url = item.get("video_url", {}).get("url", "")
            assert url.startswith("data:video/mp4;base64,")
        elif item.get("type") == "image_url":
            image_count += 1
            # Verify image URL format
            url = item.get("image_url", {}).get("url", "")
            assert url.startswith("data:image/jpeg;base64,")

    # Should have some videos due to 0.7 probability
    assert video_count > 0
    assert image_count > 0


@pytest.mark.benchmark
def test_random_mm_video_only_sampling(hf_tokenizer: PreTrainedTokenizerBase) -> None:
    """Test sampling with only video buckets."""
    ds = RandomMultiModalDataset(random_seed=42)

    bucket_config = {
        (64, 64, 8): 1.0,  # Only videos
    }

    limit_mm_per_prompt = {"image": 0, "video": 1}

    samples = _collect_mm_samples(
        ds,
        hf_tokenizer,
        num_requests=3,
        base_items_per_request=1,
        num_mm_items_range_ratio=0.0,
        limit_mm_per_prompt=limit_mm_per_prompt,
        bucket_config=bucket_config,
    )

    assert len(samples) == 3

    for s in samples:
        mm_data = cast(list[dict[str, Any]], s.multi_modal_data)
        assert len(mm_data) == 1

        item = mm_data[0]
        assert item.get("type") == "video_url"
        url = item.get("video_url", {}).get("url", "")
        assert url.startswith("data:video/mp4;base64,")


@pytest.mark.benchmark
def test_random_mm_video_deterministic_sampling(
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Test that video sampling is deterministic with same seed."""
    seed = 123
    ds_a = RandomMultiModalDataset(random_seed=seed)
    ds_b = RandomMultiModalDataset(random_seed=seed)

    bucket_config = {
        (64, 64, 8): 1.0,  # Only videos
    }

    limit_mm_per_prompt = {"image": 0, "video": 1}

    a = _collect_mm_samples(
        ds_a,
        hf_tokenizer,
        num_requests=3,
        base_items_per_request=1,
        num_mm_items_range_ratio=0.0,
        limit_mm_per_prompt=limit_mm_per_prompt,
        bucket_config=bucket_config,
    )

    b = _collect_mm_samples(
        ds_b,
        hf_tokenizer,
        num_requests=3,
        base_items_per_request=1,
        num_mm_items_range_ratio=0.0,
        limit_mm_per_prompt=limit_mm_per_prompt,
        bucket_config=bucket_config,
    )

    fa = [_mm_fingerprint_sample(s) for s in a]
    fb = [_mm_fingerprint_sample(s) for s in b]
    assert fa == fb
