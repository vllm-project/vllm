# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "vllm" / "platforms" / "xpu_utils.py"
_SPEC = importlib.util.spec_from_file_location("xpu_utils", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

has_multi_gpu_battlemage = _MODULE.has_multi_gpu_battlemage
is_intel_battlemage = _MODULE.is_intel_battlemage
maybe_apply_battlemage_xccl_workaround = (
    _MODULE.maybe_apply_battlemage_xccl_workaround
)


def test_is_intel_battlemage() -> None:
    assert is_intel_battlemage("Intel(R) Arc(TM) Pro B70 Graphics")
    assert is_intel_battlemage("Intel Arc B580 Graphics")
    assert is_intel_battlemage("Intel Battlemage Test Device")
    assert not is_intel_battlemage("Intel(R) Data Center GPU Max 1550")
    assert not is_intel_battlemage("Intel(R) Arc(TM) A770 Graphics")


def test_has_multi_gpu_battlemage() -> None:
    assert has_multi_gpu_battlemage(
        [
            "Intel(R) Arc(TM) Pro B70 Graphics",
            "Intel Arc B580 Graphics",
        ]
    )
    assert not has_multi_gpu_battlemage(["Intel(R) Arc(TM) Pro B70 Graphics"])
    assert not has_multi_gpu_battlemage(
        [
            "Intel(R) Arc(TM) Pro B70 Graphics",
            "Intel(R) Arc(TM) A770 Graphics",
        ]
    )


def test_maybe_apply_battlemage_xccl_workaround_sets_env() -> None:
    environ: dict[str, str] = {}
    applied = maybe_apply_battlemage_xccl_workaround(
        get_device_count=lambda: 2,
        get_device_name=lambda idx: (
            "Intel(R) Arc(TM) Pro B70 Graphics"
            if idx == 0
            else "Intel Arc B580 Graphics"
        ),
        environ=environ,
    )

    assert applied
    assert environ["CCL_ZE_CACHE_OPEN_IPC_HANDLES"] == "0"


def test_maybe_apply_battlemage_xccl_workaround_preserves_override() -> None:
    environ = {"CCL_ZE_CACHE_OPEN_IPC_HANDLES": "1"}
    applied = maybe_apply_battlemage_xccl_workaround(
        get_device_count=lambda: 2,
        get_device_name=lambda idx: "Intel(R) Arc(TM) Pro B70 Graphics",
        environ=environ,
    )

    assert not applied
    assert environ["CCL_ZE_CACHE_OPEN_IPC_HANDLES"] == "1"


def test_maybe_apply_battlemage_xccl_workaround_skips_other_devices() -> None:
    environ: dict[str, str] = {}
    applied = maybe_apply_battlemage_xccl_workaround(
        get_device_count=lambda: 2,
        get_device_name=lambda idx: "Intel(R) Data Center GPU Max 1550",
        environ=environ,
    )

    assert not applied
    assert "CCL_ZE_CACHE_OPEN_IPC_HANDLES" not in environ
