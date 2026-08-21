# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eligibility for the ROCm fused all-reduce + RMSNorm latent-MoE path.

``MoERunner._can_fuse_ar_rmsnorm`` decides whether the latent-MoE all-reduce
and its routed-output RMSNorm collapse into one aiter kernel. Getting it wrong
is silent: a false positive would drop a post-norm op (e.g. an unrecognised
transform) or run the kernel on an unsupported layout. These pin the predicate
device-free by mocking the platform and feeding a tensor stand-in, so the
boolean logic is tested without a GPU or an initialised aiter all-reduce.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.utils import ensure_current_vllm_config
from vllm.model_executor.layers.fused_moe.runner import moe_runner
from vllm.model_executor.layers.layernorm import RMSNorm

pytestmark = pytest.mark.cpu_test

HIDDEN = 64
LATENT = 32


class _FakeTensor:
    """Stand-in exposing only what the predicate reads (no GPU needed)."""

    def __init__(
        self,
        *,
        is_cuda: bool = True,
        dim: int = 2,
        contiguous: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.is_cuda = is_cuda
        self._dim = dim
        self._contiguous = contiguous
        self.dtype = dtype

    def dim(self) -> int:
        return self._dim

    def is_contiguous(self) -> bool:
        return self._contiguous


@pytest.fixture
def norm():
    # RMSNorm construction needs a current vLLM config for its custom-op setup.
    with ensure_current_vllm_config():
        yield RMSNorm(LATENT, eps=1e-5)


def _runner(norm, **overrides):
    """Bare MoERunner carrying only the attributes the predicate reads."""
    transform: SimpleNamespace | None = SimpleNamespace(
        norm=overrides.pop("transform_norm", norm),
        up_proj=overrides.pop("transform_up_proj", lambda x: (x, None)),
    )
    if overrides.pop("no_transform", False):
        transform = None

    runner = object.__new__(moe_runner.MoERunner)
    runner.routed_output_transform = transform
    runner.routed_scaling_factor = overrides.pop("routed_scaling_factor", 1.0)
    runner.moe_config = SimpleNamespace(
        tp_size=overrides.pop("tp_size", 8),
        ep_size=overrides.pop("ep_size", 1),
        is_sequence_parallel=overrides.pop("is_sequence_parallel", False),
    )
    assert not overrides, f"unexpected overrides: {overrides}"
    return runner


@pytest.fixture(autouse=True)
def _force_rocm_with_aiter(monkeypatch):
    """Default the environment gates to eligible; each test perturbs one."""
    monkeypatch.setattr(moe_runner.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(moe_runner, "_aiter_fused_ar_rmsnorm", object())


def test_eligible_under_the_kimi_k3_serving_shape(norm):
    runner = _runner(norm)
    assert runner._can_fuse_ar_rmsnorm(_FakeTensor(), False)


def test_disabled_when_aiter_op_missing(norm, monkeypatch):
    monkeypatch.setattr(moe_runner, "_aiter_fused_ar_rmsnorm", None)
    runner = _runner(norm)
    assert not runner._can_fuse_ar_rmsnorm(_FakeTensor(), False)


def test_disabled_off_rocm(norm, monkeypatch):
    monkeypatch.setattr(moe_runner.current_platform, "is_rocm", lambda: False)
    runner = _runner(norm)
    assert not runner._can_fuse_ar_rmsnorm(_FakeTensor(), False)


def test_disabled_without_transform(norm):
    runner = _runner(norm, no_transform=True)
    assert not runner._can_fuse_ar_rmsnorm(_FakeTensor(), False)


def test_disabled_when_norm_is_not_rmsnorm(norm):
    # A transform whose ``.norm`` is some other module must not be fused, or
    # its real normalization would be silently replaced.
    runner = _runner(norm, transform_norm=torch.nn.LayerNorm(LATENT))
    assert not runner._can_fuse_ar_rmsnorm(_FakeTensor(), False)


def test_disabled_when_up_proj_not_callable(norm):
    runner = _runner(norm, transform_up_proj=object())
    assert not runner._can_fuse_ar_rmsnorm(_FakeTensor(), False)


def test_disabled_with_routed_scaling_factor(norm):
    # The fused path assumes the routed-scale step stays a no-op.
    runner = _runner(norm, routed_scaling_factor=2.0)
    assert not runner._can_fuse_ar_rmsnorm(_FakeTensor(), False)


def test_eligible_under_expert_parallelism_without_tp(norm):
    # With EP the MoE reports tp_size == 1 but still all-reduces the routed
    # output across the EP ranks, so the fused path must engage on ep_size too.
    runner = _runner(norm, tp_size=1, ep_size=8)
    assert runner._can_fuse_ar_rmsnorm(_FakeTensor(), False)


def test_disabled_without_tp_or_ep(norm):
    runner = _runner(norm, tp_size=1, ep_size=1)
    assert not runner._can_fuse_ar_rmsnorm(_FakeTensor(), False)


def test_disabled_under_sequence_parallelism(norm):
    runner = _runner(norm, is_sequence_parallel=True)
    assert not runner._can_fuse_ar_rmsnorm(_FakeTensor(), False)


def test_disabled_when_already_reduced(norm):
    runner = _runner(norm)
    assert not runner._can_fuse_ar_rmsnorm(_FakeTensor(), True)


def test_disabled_for_non_cuda_tensor(norm):
    runner = _runner(norm)
    assert not runner._can_fuse_ar_rmsnorm(_FakeTensor(is_cuda=False), False)


def test_disabled_for_non_2d_tensor(norm):
    runner = _runner(norm)
    assert not runner._can_fuse_ar_rmsnorm(_FakeTensor(dim=3), False)


def test_disabled_for_non_contiguous_tensor(norm):
    runner = _runner(norm)
    assert not runner._can_fuse_ar_rmsnorm(_FakeTensor(contiguous=False), False)


def test_disabled_for_unsupported_dtype(norm):
    runner = _runner(norm)
    assert not runner._can_fuse_ar_rmsnorm(_FakeTensor(dtype=torch.float32), False)
