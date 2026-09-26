# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4 MXFP4 selector must say why every backend was rejected."""

import pytest

from vllm.model_executor.layers.fused_moe.oracle import mxfp4 as mxfp4_oracle
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    select_deepseek_v4_mxfp4_moe_backend,
)

from .utils import make_dummy_moe_config

# Pinned so the test does not inherit the host's platform priorities: on CPU
# and XPU the real list is a single backend, which would not exercise the
# aggregation at all.
CANDIDATES = [
    Mxfp4MoeBackend.TRITON_UNFUSED,
    Mxfp4MoeBackend.MARLIN,
]
# Reasons deliberately do not mention a backend, so an assertion on a reason
# cannot pass by accidentally matching the backend name.
REASONS = {
    Mxfp4MoeBackend.TRITON_UNFUSED: "first refusal",
    Mxfp4MoeBackend.MARLIN: "second refusal",
}


def _refuser(name: str, reason: str) -> type:
    return type(
        name,
        (),
        {"is_supported_config": staticmethod(lambda *a, **k: (False, reason))},
    )


@pytest.fixture
def pinned_candidates(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(mxfp4_oracle, "_get_priority_backends", lambda: CANDIDATES)
    return CANDIDATES


def _config():
    return make_dummy_moe_config(hidden_dim=256, intermediate_size=64)


def test_every_rejection_reaches_the_error(pinned_candidates, monkeypatch) -> None:
    """Without this the message says only that nothing worked.

    The generic MXFP4 selector already lists the candidates and each
    rejection; the DeepSeek-V4 path logged them at debug and dropped them, so
    a server that refuses to start gave no way to tell a missing kernel from,
    say, a batch-invariance gate without reading the source.
    """
    monkeypatch.setattr(
        mxfp4_oracle,
        "backend_to_kernel_cls",
        lambda backend: [_refuser(f"Kernel{backend.value}", REASONS[backend])],
    )

    with pytest.raises(NotImplementedError) as excinfo:
        select_deepseek_v4_mxfp4_moe_backend(_config())

    message = str(excinfo.value)
    assert f"Candidate backends were: {[b.value for b in CANDIDATES]}" in message
    for backend in CANDIDATES:
        entry = (
            f"backend: {backend.value}, "
            f"kernel: Kernel{backend.value}, "
            f"reason: {REASONS[backend]}"
        )
        assert entry in message, message


def test_each_kernel_class_of_a_backend_is_named(
    pinned_candidates, monkeypatch
) -> None:
    """One backend can offer several kernel classes that refuse differently."""
    monkeypatch.setattr(
        mxfp4_oracle,
        "backend_to_kernel_cls",
        lambda backend: [
            _refuser("MonolithicKernel", "monolithic refusal"),
            _refuser("ModularKernel", "modular refusal"),
        ],
    )

    with pytest.raises(NotImplementedError) as excinfo:
        select_deepseek_v4_mxfp4_moe_backend(_config())

    message = str(excinfo.value)
    for kernel, reason in (
        ("MonolithicKernel", "monolithic refusal"),
        ("ModularKernel", "modular refusal"),
    ):
        assert f"kernel: {kernel}, reason: {reason}" in message, message


def test_a_later_backend_can_still_win(pinned_candidates, monkeypatch) -> None:
    """Collecting reasons must not turn a recoverable rejection into a failure."""
    winner = type(
        "WinningKernel",
        (),
        {"is_supported_config": staticmethod(lambda *a, **k: (True, None))},
    )

    def kernels(backend):
        if backend is CANDIDATES[0]:
            return [_refuser("FirstKernel", "first refusal")]
        return [winner]

    monkeypatch.setattr(mxfp4_oracle, "backend_to_kernel_cls", kernels)

    backend, experts_cls = select_deepseek_v4_mxfp4_moe_backend(_config())

    assert backend is CANDIDATES[1]
    assert experts_cls is winner
