# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest import mock

from vllm.v1.worker.xpu_worker import _should_warm_up_xccl


def test_single_device_skips_xccl_probe():
    with mock.patch(
        "vllm.v1.worker.xpu_worker.torch.distributed.is_xccl_available"
    ) as is_xccl_available:
        assert not _should_warm_up_xccl(1)

    is_xccl_available.assert_not_called()


def test_multi_device_warms_up_when_xccl_is_available():
    with mock.patch(
        "vllm.v1.worker.xpu_worker.torch.distributed.is_xccl_available",
        return_value=True,
    ) as is_xccl_available:
        assert _should_warm_up_xccl(2)

    is_xccl_available.assert_called_once_with()


def test_multi_device_skips_warmup_when_xccl_is_unavailable():
    with mock.patch(
        "vllm.v1.worker.xpu_worker.torch.distributed.is_xccl_available",
        return_value=False,
    ) as is_xccl_available:
        assert not _should_warm_up_xccl(2)

    is_xccl_available.assert_called_once_with()
