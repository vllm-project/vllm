# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.platforms import current_platform
from vllm.v1.attention.backends import fa_utils

pytestmark = pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU only")


@pytest.mark.parametrize("fa_version,expected", [(2, True), (3, False), (4, False)])
def test_xpu_is_fa_version_supported(fa_version, expected):
    assert fa_utils.is_fa_version_supported(fa_version) is expected


def test_xpu_selected_fa_version_is_supported():
    fa_version = fa_utils.get_flash_attn_version()
    assert fa_version is not None
    assert fa_utils.is_fa_version_supported(fa_version)
