# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the ``skip_pixel_values`` raw-byte payload.

``raw_images`` is positional against ``mm_hashes``, and the consumer has no
way to mix it with serialized tensors, so the producer must either cover
every item or decline the whole request. These tests pin that rule without
needing a model.
"""

import logging

import pybase64
import pytest

from vllm.entrypoints.scale_out.render import serving as render_serving
from vllm.entrypoints.scale_out.render.serving import _encode_raw_images


@pytest.fixture
def warnings(caplog):
    """Capture this module's warnings; the `vllm` logger does not propagate."""
    logger = logging.getLogger(render_serving.__name__)
    logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)


def test_encodes_every_item_when_all_bytes_are_retained():
    encoded = _encode_raw_images(
        {"image": ["h0", "h1"]},
        {"mm_raw_bytes": {"image": [b"\x89PNG-a", b"\x89PNG-b"]}},  # type: ignore[typeddict-item]
    )

    assert encoded is not None
    assert [pybase64.b64decode(e, validate=True) for e in encoded["image"]] == [
        b"\x89PNG-a",
        b"\x89PNG-b",
    ]


def test_declines_when_a_modality_never_retained_bytes(warnings):
    """Audio and video carry no source bytes, so a mixed request must fall
    back to shipping tensors for everything rather than half a payload — and
    say so, since the caller asked for the opposite."""
    encoded = _encode_raw_images(
        {"image": ["h0"], "audio": ["h1"]},
        {"mm_raw_bytes": {"image": [b"\x89PNG"]}},  # type: ignore[typeddict-item]
    )

    assert encoded is None
    assert "audio" in warnings.text


def test_declines_on_partial_coverage_within_a_modality():
    assert (
        _encode_raw_images(
            {"image": ["h0", "h1"]},
            {"mm_raw_bytes": {"image": [b"\x89PNG", None]}},  # type: ignore[typeddict-item]
        )
        is None
    )

    assert (
        _encode_raw_images(
            {"image": ["h0", "h1"]},
            {"mm_raw_bytes": {"image": [b"\x89PNG"]}},  # type: ignore[typeddict-item]
        )
        is None
    )


def test_declines_when_the_renderer_reported_nothing():
    assert _encode_raw_images({"image": ["h0"]}, {}) is None  # type: ignore[arg-type]
