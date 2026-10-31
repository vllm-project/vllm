# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

import pytest

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU only")


@pytest.mark.parametrize("module", ["attention", "audio_encoder", "vision_attention"])
def test_dots3_uses_platform_varlen_func(module):
    from vllm._xpu_ops import xpu_ops

    mod = importlib.import_module(f"vllm.models.dots3_note.nvidia.{module}")
    assert mod.flash_attn_varlen_func is xpu_ops.flash_attn_varlen_func


def test_dots3_audio_selects_fa2():
    from vllm.models.dots3_note.nvidia import audio_encoder

    assert audio_encoder._FLASH_ATTN_VERSION == 2
