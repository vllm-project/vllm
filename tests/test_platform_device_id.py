# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``Platform.device_id_to_physical_device_id``.

Regression coverage for the ``ValueError`` raised when the device-control env
var (e.g. ``CUDA_VISIBLE_DEVICES``) contains a GPU or MIG instance UUID rather
than an integer index.
"""

from vllm.platforms.interface import Platform


class _StubPlatform(Platform):
    device_control_env_var = "CUDA_VISIBLE_DEVICES"
    device_name = "stub"


def test_integer_indices_unchanged(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert _StubPlatform.device_id_to_physical_device_id(0) == 2
    assert _StubPlatform.device_id_to_physical_device_id(1) == 3


def test_mig_uuid_does_not_crash(monkeypatch):
    # Single MIG instance pinned by UUID -- CUDA exposes it as logical device 0.
    monkeypatch.setenv(
        "CUDA_VISIBLE_DEVICES", "MIG-377e0049-554c-540b-93c6-d0976f8426cb"
    )
    assert _StubPlatform.device_id_to_physical_device_id(0) == 0


def test_gpu_uuid_list(monkeypatch):
    monkeypatch.setenv(
        "CUDA_VISIBLE_DEVICES",
        "GPU-16d07083-ab1d-cd47-d8bb-9e7f83104985,"
        "GPU-aa11bb22-cc33-dd44-ee55-ff6677889900",
    )
    assert _StubPlatform.device_id_to_physical_device_id(0) == 0
    assert _StubPlatform.device_id_to_physical_device_id(1) == 1


def test_empty_env_returns_logical_id(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert _StubPlatform.device_id_to_physical_device_id(3) == 3
