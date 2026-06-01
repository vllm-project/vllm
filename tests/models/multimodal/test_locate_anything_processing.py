# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import types

from vllm.model_executor.models.locate_anything import (
    LocateAnythingProcessingInfo,
)


def _fake_info():
    info = LocateAnythingProcessingInfo.__new__(LocateAnythingProcessingInfo)
    fake_image_processor = types.SimpleNamespace(
        patch_size=14,
        merge_kernel_size=(2, 2),
        in_token_limit=4096,
    )
    fake_processor = types.SimpleNamespace(image_processor=fake_image_processor)
    info.get_hf_processor = lambda **kw: fake_processor
    return info


def test_num_image_tokens_basic():
    info = _fake_info()
    # 448x448 -> 32x32 patches -> /(2*2) -> 16x16 = 256 tokens
    n = info.get_num_image_tokens(image_width=448, image_height=448)
    assert n == 256


def test_num_image_tokens_pads_to_merge_multiple():
    info = _fake_info()
    # 430 not a multiple of 28 -> pad up to 448 -> 16x16 = 256
    n = info.get_num_image_tokens(image_width=430, image_height=430)
    assert n == 256


def test_num_image_tokens_exact_multiple_no_pad():
    info = _fake_info()
    # 420 already a multiple of 28 -> 15x15 = 225 (no padding)
    n = info.get_num_image_tokens(image_width=420, image_height=420)
    assert n == 225
