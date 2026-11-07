# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")


@pytest.mark.parametrize(
    "arch, env, expected",
    [
        ("gfx942:sramecc+:xnack-", None, True),
        ("gfx950:sramecc+:xnack-", None, True),
        ("gfx90a:sramecc+:xnack-", None, True),
        ("gfx90a:sramecc+:xnack-", "0", False),
        ("gfx1100", None, False),
        ("gfx1100", "1", False),
    ],
)
def test_use_custom_allreduce_gfx90a(monkeypatch, arch, env, expected):
    from vllm.platforms import rocm

    monkeypatch.setattr(rocm, "_GCN_ARCH", arch)
    if env is None:
        monkeypatch.delenv("VLLM_ROCM_CUSTOM_AR_GFX90A", raising=False)
    else:
        monkeypatch.setenv("VLLM_ROCM_CUSTOM_AR_GFX90A", env)
    assert rocm.RocmPlatform.use_custom_allreduce() is expected
