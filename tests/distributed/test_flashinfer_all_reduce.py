# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.device_communicators import flashinfer_all_reduce as fi_ar

MiB = 1024 * 1024


def _uninitialized_comm() -> fi_ar.FlashInferAllReduce:
    comm = fi_ar.FlashInferAllReduce.__new__(fi_ar.FlashInferAllReduce)
    comm.disabled = False
    comm.group = object()
    comm.device = torch.device("cuda:0")
    comm.world_size = 2
    comm.rank = 0
    comm.max_workspace_size = 8 * MiB
    comm.max_num_tokens = 0
    return comm


def _fake_tensor(shape, dtype) -> SimpleNamespace:
    element_size = torch.empty((), dtype=dtype).element_size()
    numel = shape[0] * shape[1]
    return SimpleNamespace(
        is_cuda=True,
        is_contiguous=lambda: True,
        shape=shape,
        dtype=dtype,
        nbytes=numel * element_size,
        element_size=lambda: element_size,
    )


@pytest.mark.parametrize(
    ("shape", "dtype", "expected"),
    [
        # Mamba2 with n_groups=1 all-reduces an fp32 (num_tokens, 1) partial sum.
        ((16, 1), torch.float32, False),
        ((16, 2), torch.float32, False),
        ((16, 1), torch.bfloat16, False),
        ((16, 4), torch.bfloat16, False),
        ((16, 4), torch.float32, True),
        ((16, 8), torch.bfloat16, True),
        ((16, 4096), torch.bfloat16, True),
    ],
)
def test_should_use_requires_16_byte_row_width(monkeypatch, shape, dtype, expected):
    comm = _uninitialized_comm()
    monkeypatch.setattr(
        fi_ar.FlashInferAllReduce, "_ensure_workspace", lambda *_, **__: True
    )
    monkeypatch.setattr(
        fi_ar,
        "get_fi_ar_workspace",
        lambda **_: SimpleNamespace(is_buffer_size_sufficient=lambda **__: True),
    )

    assert comm.should_use_fi_ar(_fake_tensor(shape, dtype)) is expected


def test_narrow_row_does_not_allocate_a_workspace(monkeypatch):
    # Workspace creation is collective, so a shape the kernel cannot accept must
    # be rejected before any rank starts allocating one.
    def _fail(*_, **__):
        raise AssertionError("workspace must not be created for a rejected shape")

    monkeypatch.setattr(fi_ar.FlashInferAllReduce, "_ensure_workspace", _fail)
    monkeypatch.setattr(fi_ar, "get_fi_ar_workspace", _fail)

    comm = _uninitialized_comm()
    assert comm.should_use_fi_ar(_fake_tensor((16, 1), torch.float32)) is False
