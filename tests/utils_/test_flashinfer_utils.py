# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

import vllm.utils.flashinfer as fi


def _make_exe(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(mode=0o755)


@pytest.fixture
def default_cuda_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Hide any real toolkit, keep ninja on PATH, redirect /usr/local/cuda."""
    for var in ("CUDA_HOME", "CUDA_PATH", "FLASHINFER_NVCC"):
        monkeypatch.delenv(var, raising=False)
    _make_exe(tmp_path / "venv" / "bin" / "ninja")
    monkeypatch.setenv("PATH", str(tmp_path / "venv" / "bin"))
    home = tmp_path / "usr_local_cuda"
    monkeypatch.setattr(fi, "_DEFAULT_CUDA_HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def flashinfer_without_cubin(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pretend flashinfer-python is installed without flashinfer-cubin."""
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args: (
            object() if name == "flashinfer" else real_find_spec(name, *args)
        ),
    )
    monkeypatch.setattr(fi, "has_flashinfer_cubin", lambda: False)
    fi.has_flashinfer.cache_clear()
    yield
    fi.has_flashinfer.cache_clear()


def test_has_flashinfer_finds_toolkit_off_path(default_cuda_home: Path):
    """Bare-metal installs keep nvcc in /usr/local/cuda, off PATH, which is
    where FlashInfer's JIT finds it."""
    _make_exe(default_cuda_home / "bin" / "nvcc")
    assert fi.has_flashinfer()


def test_has_flashinfer_requires_ninja(
    default_cuda_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """FlashInfer runs `ninja` from PATH, which lacks the venv's bin directory
    when vLLM is launched without activating the venv."""
    _make_exe(tmp_path / "cuda" / "bin" / "nvcc")
    monkeypatch.setenv("PATH", str(tmp_path / "cuda" / "bin"))
    assert not fi.has_flashinfer()


@pytest.mark.parametrize(
    (
        "flashinfer",
        "jit_cache",
        "cubin",
        "nvcc",
        "ninja",
        "device",
        "capability_unavailable",
        "provider_arch",
        "provider_modules",
        "expected",
    ),
    [
        # A shim by itself, even alongside FlashInfer cubins, has no kernels.
        (True, True, True, False, False, (12, 0), False, None, None, False),
        # A provider does not substitute for has_flashinfer()'s cubin/JIT check.
        (
            True,
            True,
            False,
            False,
            False,
            (12, 0),
            False,
            "sm120",
            "attention_module",
            False,
        ),
        # Cubins plus a provider for the active architecture enables the AOT path.
        (
            True,
            True,
            True,
            False,
            False,
            (12, 0),
            False,
            "sm120",
            "attention_module",
            True,
        ),
        # A provider with no modules is not usable.
        (True, True, True, False, False, (12, 0), False, "sm120", None, False),
        # An architecture-specific target matches the exact active architecture.
        (
            True,
            True,
            True,
            False,
            False,
            (12, 0),
            False,
            "sm120a",
            "attention_module",
            True,
        ),
        # A family target matches SM120.
        (
            True,
            True,
            True,
            False,
            False,
            (12, 0),
            False,
            "sm120f",
            "attention_module",
            True,
        ),
        # A family target is forward-compatible with SM121.
        (
            True,
            True,
            True,
            False,
            False,
            (12, 1),
            False,
            "sm120f",
            "attention_module",
            True,
        ),
        # An architecture-specific target is not forward-compatible with SM121.
        (
            True,
            True,
            True,
            False,
            False,
            (12, 1),
            False,
            "sm120a",
            "attention_module",
            False,
        ),
        # A provider built for another architecture cannot enable the backend.
        (
            True,
            True,
            True,
            False,
            False,
            (12, 0),
            False,
            "sm90a",
            "attention_module",
            False,
        ),
        # If capability is unavailable, non-empty provider metadata is the fallback.
        (
            True,
            True,
            True,
            False,
            False,
            (12, 0),
            True,
            "sm120a",
            "attention_module",
            True,
        ),
        (True, True, True, False, False, (12, 0), True, "sm120", None, False),
        # Runtime compilation remains a valid path without an AOT provider.
        (True, False, False, True, True, (12, 0), False, None, None, True),
        # The SM75 guard still rejects the jit-cache path without a compiler.
        (
            True,
            True,
            True,
            False,
            False,
            (7, 5),
            False,
            "sm75",
            "attention_module",
            False,
        ),
        (False, True, True, False, False, (12, 0), False, None, None, False),
    ],
)
def test_is_flashinfer_jit_usable(
    monkeypatch: pytest.MonkeyPatch,
    flashinfer: bool,
    jit_cache: bool,
    cubin: bool,
    nvcc: bool,
    ninja: bool,
    device: tuple[int, int],
    capability_unavailable: bool,
    provider_arch: str | None,
    provider_modules: str | None,
    expected: bool,
):
    from vllm.platforms.interface import DeviceCapability

    providers = (
        (
            SimpleNamespace(
                modules={provider_modules} if provider_modules else set(),
                cuda_architectures={provider_arch},
            ),
        )
        if provider_arch
        else ()
    )

    monkeypatch.setattr(
        fi.importlib.util,
        "find_spec",
        lambda name: object()
        if (name == "flashinfer" and flashinfer)
        or (name == "flashinfer_jit_cache" and jit_cache)
        else None,
    )
    monkeypatch.setattr(fi, "has_flashinfer_cubin", lambda: cubin)
    fi.has_flashinfer.cache_clear()
    monkeypatch.setattr(
        fi,
        "_flashinfer_nvcc_path",
        lambda: "/usr/local/cuda/bin/nvcc" if nvcc else None,
    )
    monkeypatch.setattr(
        fi.shutil,
        "which",
        lambda name: "/usr/bin/ninja" if name == "ninja" and ninja else None,
    )
    major, minor = device
    monkeypatch.setattr(
        fi.current_platform,
        "is_device_capability",
        lambda capability: (major, minor) == (7, 5) and capability == 75,
    )
    monkeypatch.setattr(
        fi.current_platform,
        "get_device_capability",
        lambda: None if capability_unavailable else DeviceCapability(major, minor),
    )
    monkeypatch.setattr(
        fi.importlib,
        "import_module",
        lambda name: SimpleNamespace(get_jit_cache_providers=lambda: providers),
    )

    assert fi.is_flashinfer_jit_usable() is expected
    fi.has_flashinfer.cache_clear()


@pytest.mark.parametrize(
    ("provider_architecture", "device_architecture", "expected"),
    [
        ("sm120f", "sm120", True),  # stock FlashInfer 0.7.2 provider on SM120
        ("sm120f", "sm121", True),  # forward-compatible family target
        ("sm120a", "sm120", True),  # exact architecture-specific target
        ("sm120a", "sm121", False),  # architecture-specific target is not forward-compatible
        ("sm120", "sm121", True),  # unsuffixed forward-compatible within the major
        ("sm121", "sm120", False),  # provider target is newer than the device
        ("sm90a", "sm120", False),
        ("not-an-arch", "sm120", False),
    ],
)
def test_flashinfer_cuda_architecture_compatibility(
    provider_architecture: str,
    device_architecture: str,
    expected: bool,
):
    assert (
        fi._is_compatible_flashinfer_cuda_architecture(
            provider_architecture, device_architecture
        )
        is expected
    )
