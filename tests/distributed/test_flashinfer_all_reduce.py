# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission-gate tests for :class:`FlashInferAllReduce`.

These exercise the capacity accounting only, so they run on CPU without CUDA or
flashinfer installed: both the workspace and the input tensor are stubbed.
"""

import pytest
import torch

from vllm.distributed.device_communicators import flashinfer_all_reduce
from vllm.distributed.device_communicators.flashinfer_all_reduce import (
    FlashInferAllReduce,
)
from vllm.utils.math_utils import round_up

# These tests stub the workspace and never allocate on device, so skip the
# global GPU teardown -- on non-CUDA hosts it drives torch.accelerator into an
# uninitialized allocator and errors during teardown.
pytestmark = pytest.mark.skip_global_cleanup

# The repro from the bug report: DeepSeek-V4 hidden size, TP8, bf16, and the 2MB
# per-rank budget FI_ALLREDUCE_FUSION_MAX_SIZE_MB gives TP8 on sm103.
HIDDEN_DIM = 7168
WORLD_SIZE = 8
DTYPE = torch.bfloat16
WORKSPACE_BUDGET = 2 * 1024 * 1024

# Byte counts quoted in the traceback, reproduced by _FakeMnnvlWorkspace below.
REPORTED_BUFFER_BYTES = 698368
REPORTED_REQUIRED_BYTES = 802816

# Tokens whose bf16 payload exceeds a single Lamport buffer but still fits the
# whole workspace budget -- the window that used to reach the kernel and abort.
OVERSIZED_TOKENS = 56
# Largest token count that genuinely fits one Lamport buffer here.
FITTING_TOKENS = 48


class _FakeMnnvlWorkspace:
    """Mimics flashinfer's MNNVL workspace capacity accounting.

    flashinfer sizes one Lamport buffer at roughly ``payload / 3`` and allocates
    three of them, so only about a third of the requested budget backs any single
    all-reduce. ``is_buffer_size_sufficient`` is the real API that the production
    code consults instead of reimplementing this arithmetic.
    """

    NUM_LAMPORT_BUFFERS = 3
    backend = "mnnvl"

    def __init__(self, max_token_num: int, hidden_dim: int, dtype: torch.dtype):
        payload = max_token_num * hidden_dim * dtype.itemsize
        self.buffer_size_bytes = round_up(payload // self.NUM_LAMPORT_BUFFERS, 1024)

    def is_buffer_size_sufficient(
        self,
        tp_size: int,
        num_tokens: int,
        hidden_dim: int,
        dtype: torch.dtype,
        use_oneshot=None,
    ) -> bool:
        return num_tokens * hidden_dim * dtype.itemsize <= self.buffer_size_bytes


class _FakeTensor:
    """Just the surface ``should_use_fi_ar`` inspects, so no GPU is needed."""

    def __init__(self, num_tokens: int, hidden_dim: int, dtype: torch.dtype):
        self.shape = (num_tokens, hidden_dim)
        self.dtype = dtype
        self.is_cuda = True

    def is_contiguous(self) -> bool:
        return True


@pytest.fixture
def comm(monkeypatch):
    """A FlashInferAllReduce wired to a stub MNNVL workspace.

    ``__init__`` is bypassed because it needs a live process group and a CUDA
    platform; the attributes it would set are filled in directly. The stub is
    cached like the real process-wide singleton, so the first caller's shape
    fixes the capacity for every later caller.
    """
    singleton: list[_FakeMnnvlWorkspace] = []

    def fake_get_workspace(world_size, rank, max_token_num, hidden_dim, dtype, group):
        if not singleton:
            singleton.append(_FakeMnnvlWorkspace(max_token_num, hidden_dim, dtype))
        return singleton[0]

    monkeypatch.setattr(
        flashinfer_all_reduce, "get_fi_ar_workspace", fake_get_workspace
    )

    obj = FlashInferAllReduce.__new__(FlashInferAllReduce)
    obj.disabled = False
    obj.group = None
    obj.world_size = WORLD_SIZE
    obj.rank = 0
    obj.device = "cuda:0"
    obj.max_workspace_size = WORKSPACE_BUDGET
    obj.max_num_tokens = 0
    obj.workspace = None
    return obj


def test_rejects_tensor_larger_than_one_lamport_buffer(comm):
    """The reported crash: 56 tokens fit the budget but not a Lamport buffer."""
    tensor = _FakeTensor(OVERSIZED_TOKENS, HIDDEN_DIM, DTYPE)

    assert comm.should_use_fi_ar(tensor) is False

    # The cheap budget bound admits it, so only the authoritative workspace check
    # can be what rejected it. Without that check this tensor reached the kernel.
    assert comm.max_num_tokens >= OVERSIZED_TOKENS


def test_accepts_tensor_that_fits_one_lamport_buffer(comm):
    tensor = _FakeTensor(FITTING_TOKENS, HIDDEN_DIM, DTYPE)

    assert comm.should_use_fi_ar(tensor) is True
    assert comm.workspace is not None


def test_reproduces_the_byte_counts_from_the_traceback(comm):
    oversized = _FakeTensor(OVERSIZED_TOKENS, HIDDEN_DIM, DTYPE)
    assert comm.should_use_fi_ar(oversized) is False

    # "Buffer: 698368 bytes, Required: 802816 bytes."
    assert comm.workspace.buffer_size_bytes == REPORTED_BUFFER_BYTES
    required = OVERSIZED_TOKENS * HIDDEN_DIM * DTYPE.itemsize
    assert required == REPORTED_REQUIRED_BYTES


def test_still_rejects_tensors_beyond_the_whole_budget(comm):
    """The cheap pre-check must keep short-circuiting huge prefill tensors."""
    huge = _FakeTensor(8192, HIDDEN_DIM, DTYPE)

    assert comm.should_use_fi_ar(huge) is False
    # Rejected before any workspace was created.
    assert comm.workspace is None


def test_gate_follows_payload_not_the_first_shape_seen(comm):
    """A drafter with a different hidden size must not inherit a stale limit."""
    assert comm.should_use_fi_ar(_FakeTensor(FITTING_TOKENS, HIDDEN_DIM, DTYPE)) is True

    # Same token count, twice the hidden size: twice the payload, so it no longer
    # fits, even though the cached token-count bound would still admit it.
    wide = _FakeTensor(FITTING_TOKENS, HIDDEN_DIM * 2, DTYPE)
    assert comm.max_num_tokens >= FITTING_TOKENS
    assert comm.should_use_fi_ar(wide) is False
