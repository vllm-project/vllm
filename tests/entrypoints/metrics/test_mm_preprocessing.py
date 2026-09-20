# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for multi-modal preprocessing Prometheus metrics."""

import pytest
from prometheus_client import REGISTRY, generate_latest
from prometheus_client.parser import text_string_to_metric_families

from vllm.entrypoints.metrics import mm_preprocessing


@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset module-level state before each test."""
    mm_preprocessing._metrics_initialized = False
    mm_preprocessing._media_download_latency = None
    mm_preprocessing._media_decode_latency = None
    mm_preprocessing._media_download_bytes = None
    mm_preprocessing._mm_resolve_items_latency = None
    mm_preprocessing._mm_preprocessing_total_latency = None
    yield
    # Clean up any registered metrics after the test
    for collector in list(REGISTRY._collector_to_names):
        if hasattr(collector, "_name") and "vllm:mm_" in str(
            getattr(collector, "_name", "")
        ):
            REGISTRY.unregister(collector)
    mm_preprocessing._metrics_initialized = False
    mm_preprocessing._media_download_latency = None
    mm_preprocessing._media_decode_latency = None
    mm_preprocessing._media_download_bytes = None
    mm_preprocessing._mm_resolve_items_latency = None
    mm_preprocessing._mm_preprocessing_total_latency = None


def _get_metric_count(metric_name: str) -> float:
    """Get the _count value of a histogram metric from the registry."""
    output = generate_latest(REGISTRY).decode("utf-8")
    for family in text_string_to_metric_families(output):
        if family.name == metric_name:
            for sample in family.samples:
                if sample.name == metric_name + "_count":
                    return sample.value
    return 0.0


def _get_metric_sum(metric_name: str) -> float:
    """Get the _sum value of a histogram metric from the registry."""
    output = generate_latest(REGISTRY).decode("utf-8")
    for family in text_string_to_metric_families(output):
        if family.name == metric_name:
            for sample in family.samples:
                if sample.name == metric_name + "_sum":
                    return sample.value
    return 0.0


def test_ensure_metrics_lazy_init():
    """Metrics should not be initialized until first observe call."""
    assert not mm_preprocessing._metrics_initialized
    assert mm_preprocessing._media_download_latency is None

    mm_preprocessing._ensure_metrics()

    assert mm_preprocessing._metrics_initialized
    assert mm_preprocessing._media_download_latency is not None
    assert mm_preprocessing._media_decode_latency is not None
    assert mm_preprocessing._media_download_bytes is not None
    assert mm_preprocessing._mm_resolve_items_latency is not None
    assert mm_preprocessing._mm_preprocessing_total_latency is not None


def test_observe_media_download():
    """observe_media_download should record latency and bytes."""
    mm_preprocessing.observe_media_download("ImageMediaIO", 0.5, 102400)

    assert _get_metric_count("vllm:mm_media_download_latency_seconds") == 1
    assert _get_metric_sum("vllm:mm_media_download_latency_seconds") == pytest.approx(
        0.5
    )
    assert _get_metric_count("vllm:mm_media_download_bytes") == 1
    assert _get_metric_sum("vllm:mm_media_download_bytes") == pytest.approx(102400)


def test_observe_media_decode():
    """observe_media_decode should record latency."""
    mm_preprocessing.observe_media_decode("VideoMediaIO", 0.3)

    assert _get_metric_count("vllm:mm_media_decode_latency_seconds") == 1
    assert _get_metric_sum("vllm:mm_media_decode_latency_seconds") == pytest.approx(0.3)


def test_observe_resolve_items():
    """observe_resolve_items should record latency."""
    mm_preprocessing.observe_resolve_items(1.2)

    assert _get_metric_count("vllm:mm_resolve_items_latency_seconds") == 1
    assert _get_metric_sum("vllm:mm_resolve_items_latency_seconds") == pytest.approx(
        1.2
    )


def test_observe_preprocessing_total():
    """observe_preprocessing_total should record latency."""
    mm_preprocessing.observe_preprocessing_total(0.8)

    assert _get_metric_count("vllm:mm_preprocessing_total_latency_seconds") == 1
    assert _get_metric_sum(
        "vllm:mm_preprocessing_total_latency_seconds"
    ) == pytest.approx(0.8)


def test_multiple_observations_accumulate():
    """Multiple observations should accumulate in the histogram."""
    for _ in range(5):
        mm_preprocessing.observe_media_decode("ImageMediaIO", 0.1)

    count = _get_metric_count("vllm:mm_media_decode_latency_seconds")
    assert count == 5


def test_different_media_types_tracked_separately():
    """Different media_type labels should be tracked separately."""
    mm_preprocessing.observe_media_download("ImageMediaIO", 0.1, 1000)
    mm_preprocessing.observe_media_download("VideoMediaIO", 0.2, 2000)
    mm_preprocessing.observe_media_download("AudioMediaIO", 0.3, 3000)

    output = generate_latest(REGISTRY).decode("utf-8")

    image_count = 0
    video_count = 0
    audio_count = 0
    for family in text_string_to_metric_families(output):
        if family.name == "vllm:mm_media_download_latency_seconds":
            for sample in family.samples:
                if sample.name == family.name + "_count":
                    if sample.labels.get("media_type") == "ImageMediaIO":
                        image_count = sample.value
                    elif sample.labels.get("media_type") == "VideoMediaIO":
                        video_count = sample.value
                    elif sample.labels.get("media_type") == "AudioMediaIO":
                        audio_count = sample.value

    assert image_count == 1
    assert video_count == 1
    assert audio_count == 1


def test_all_five_metrics_registered():
    """All five metric families should be registered after init."""
    mm_preprocessing._ensure_metrics()

    output = generate_latest(REGISTRY).decode("utf-8")
    metric_names = {f.name for f in text_string_to_metric_families(output)}

    assert "vllm:mm_media_download_latency_seconds" in metric_names
    assert "vllm:mm_media_decode_latency_seconds" in metric_names
    assert "vllm:mm_media_download_bytes" in metric_names
    assert "vllm:mm_resolve_items_latency_seconds" in metric_names
    assert "vllm:mm_preprocessing_total_latency_seconds" in metric_names


def test_ensure_metrics_thread_safe():
    """Concurrent _ensure_metrics calls must not raise duplicate registration."""
    from concurrent.futures import ThreadPoolExecutor

    mm_preprocessing._metrics_initialized = False
    mm_preprocessing._media_download_latency = None

    def _call():
        return mm_preprocessing._ensure_metrics()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: _call(), range(16)))

    assert all(results)
    assert mm_preprocessing._metrics_initialized
