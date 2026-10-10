# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Auto-selecting a Triton-based attention backend on a pre-SM80 GPU should
emit a one-time performance warning.
See https://github.com/vllm-project/vllm/issues/60821.
"""

from unittest.mock import patch

import pytest

from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.registry import AttentionBackendEnum

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-specific tests"
)

SM75 = DeviceCapability(major=7, minor=5)
SM80 = DeviceCapability(major=8, minor=0)


@pytest.mark.parametrize(
    "backend",
    [AttentionBackendEnum.TRITON_ATTN, AttentionBackendEnum.FLEX_ATTENTION],
)
def test_warns_for_triton_backends_on_pre_sm80(backend):
    from vllm.platforms import cuda

    with patch.object(cuda.logger, "warning_once") as warn:
        cuda._warn_if_slow_pre_sm80_backend(backend, SM75)
    warn.assert_called_once()
    msg, *args = warn.call_args.args
    assert "tensor cores" in msg
    assert backend.name in args


@pytest.mark.parametrize(
    ("backend", "capability"),
    [
        (AttentionBackendEnum.TRITON_ATTN, SM80),
        (AttentionBackendEnum.FLEX_ATTENTION, DeviceCapability(9, 0)),
        (AttentionBackendEnum.FLASHINFER, SM75),
        (AttentionBackendEnum.FLASH_ATTN, SM80),
    ],
)
def test_no_warning_otherwise(backend, capability):
    from vllm.platforms import cuda

    with patch.object(cuda.logger, "warning_once") as warn:
        cuda._warn_if_slow_pre_sm80_backend(backend, capability)
    warn.assert_not_called()
