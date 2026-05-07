# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.mamba.gdn_linear_attn import ChunkGatedDeltaRule


def _make_gdn_op(
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend: str | None,
    tp_size: int,
    is_cuda: bool = True,
    has_sm90: bool = True,
) -> tuple[ChunkGatedDeltaRule, Mock]:
    logger = Mock()
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.gdn_linear_attn.logger", logger
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.gdn_linear_attn."
        "get_tensor_model_parallel_world_size",
        lambda: tp_size,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.gdn_linear_attn.current_platform.is_cuda",
        lambda: is_cuda,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.gdn_linear_attn."
        "current_platform.is_device_capability",
        lambda capability: has_sm90 and capability == 90,
    )

    additional_config = {}
    if backend is not None:
        additional_config["gdn_prefill_backend"] = backend
    vllm_config = VllmConfig(additional_config=additional_config)

    with set_current_vllm_config(vllm_config):
        return ChunkGatedDeltaRule(), logger


def test_gdn_auto_uses_flashinfer_for_single_tp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    op, logger = _make_gdn_op(monkeypatch, backend=None, tp_size=1)

    assert op._forward_method.__name__ == "forward_cuda"
    logger.warning_once.assert_not_called()


def test_gdn_auto_falls_back_to_triton_for_multi_tp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    op, logger = _make_gdn_op(monkeypatch, backend=None, tp_size=8)

    assert op._forward_method.__name__ == "forward_native"
    logger.warning_once.assert_called_once()
    warning_args = logger.warning_once.call_args[0]
    assert "GDN prefill backend 'auto' is falling back to Triton/FLA" in warning_args[
        0
    ]
    assert warning_args[1] == 8


def test_gdn_flashinfer_override_keeps_flashinfer_for_multi_tp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    op, logger = _make_gdn_op(monkeypatch, backend="flashinfer", tp_size=8)

    assert op._forward_method.__name__ == "forward_cuda"
    logger.warning_once.assert_not_called()


def test_gdn_triton_override_keeps_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    op, logger = _make_gdn_op(monkeypatch, backend="triton", tp_size=1)

    assert op._forward_method.__name__ == "forward_native"
    logger.warning_once.assert_not_called()


def test_gdn_flashinfer_falls_back_when_platform_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    op, logger = _make_gdn_op(
        monkeypatch,
        backend="flashinfer",
        tp_size=1,
        is_cuda=False,
        has_sm90=False,
    )

    assert op._forward_method.__name__ == "forward_native"
    logger.warning_once.assert_called_once()
    assert "cannot use this kernel on the current platform" in (
        logger.warning_once.call_args[0][0]
    )
