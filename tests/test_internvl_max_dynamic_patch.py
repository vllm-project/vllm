# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from PIL import Image

from vllm.transformers_utils.processors.internvl import InternVLImageProcessor

pytestmark = pytest.mark.skip_global_cleanup


def _make_processor(
    *,
    image_size: int = 448,
    image_size_limit: int | None = None,
    max_dynamic_patch: int = 12,
    max_dynamic_patch_limit: int | None = None,
) -> InternVLImageProcessor:
    return InternVLImageProcessor(
        image_size=image_size,
        min_dynamic_patch=1,
        max_dynamic_patch=max_dynamic_patch,
        dynamic_image_size=True,
        use_thumbnail=True,
        image_size_limit=image_size_limit,
        max_dynamic_patch_limit=max_dynamic_patch_limit,
    )


def test_request_max_dynamic_patch_cannot_exceed_processor_limit():
    processor = _make_processor()

    assert processor.resolve_min_max_num(max_dynamic_patch=12) == (1, 13)
    assert processor.resolve_min_max_num(max_dynamic_patch=4) == (1, 5)

    with pytest.raises(ValueError, match="cannot exceed the configured limit"):
        processor.resolve_min_max_num(max_dynamic_patch=13)

    with pytest.raises(ValueError, match="cannot exceed the configured limit"):
        processor(Image.new("RGB", (1, 1)), max_dynamic_patch=13)


def test_trusted_limit_is_kept_separate_from_merged_request_value():
    with pytest.raises(ValueError, match="cannot exceed the configured limit"):
        _make_processor(max_dynamic_patch=17, max_dynamic_patch_limit=16)

    processor = _make_processor(
        max_dynamic_patch=16,
        max_dynamic_patch_limit=16,
    )
    assert processor.resolve_min_max_num(max_dynamic_patch=16) == (1, 17)


def test_request_image_size_cannot_exceed_processor_limit():
    with pytest.raises(ValueError, match="cannot exceed the configured limit"):
        _make_processor(image_size=896, image_size_limit=448)

    processor = _make_processor(image_size=448, image_size_limit=448)
    assert processor.image_size == 448
