# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import json
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

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


@pytest.fixture
def tuner_loads(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    loads: list[str] = []

    class FakeTuner:
        def load_configs(self, path: str) -> bool:
            loads.append(Path(path).name)
            if Path(path).name == "broken.json":
                raise IndexError("not a tuning file")
            return Path(path).name != "other-gpu.json"

    class FakeAutoTuner:
        @staticmethod
        def get() -> FakeTuner:
            return FakeTuner()

    module = types.ModuleType("flashinfer.autotuner")
    module.AutoTuner = FakeAutoTuner  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", module)
    monkeypatch.setattr(fi, "_bf16_tuning_files_loaded", False)
    fi.load_flashinfer_bf16_tuning_dir.cache_clear()
    yield loads
    fi.load_flashinfer_bf16_tuning_dir.cache_clear()


@pytest.fixture
def tuning_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "gemmbf16_configs"
    directory.mkdir()
    monkeypatch.setattr(fi, "_BF16_TUNING_DIR", directory)
    return directory


def test_load_tuning_dir_loads_every_json(tuner_loads: list[str], tuning_dir: Path):
    for name in ("b.json", "a.json", "other-gpu.json", "broken.json", "notes.txt"):
        (tuning_dir / name).write_text("{}")
    fi.load_flashinfer_bf16_tuning_dir()
    assert tuner_loads == ["a.json", "b.json", "broken.json", "other-gpu.json"]
    assert fi.is_flashinfer_bf16_tuning_loaded()


def test_load_tuning_dir_missing_loads_nothing(
    tuner_loads: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(fi, "_BF16_TUNING_DIR", tmp_path / "missing")
    fi.load_flashinfer_bf16_tuning_dir()
    assert tuner_loads == []
    assert not fi.is_flashinfer_bf16_tuning_loaded()


def test_load_tuning_dir_empty_dir_loads_nothing(
    tuner_loads: list[str], tuning_dir: Path
):
    fi.load_flashinfer_bf16_tuning_dir()
    assert tuner_loads == []


@pytest.fixture
def tuner_gpus(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, str]]:
    """GPU name recorded in each tuning file at the moment FlashInfer loads it."""
    gpus: dict[str, str] = {}

    class FakeTuner:
        def load_configs(self, path: str) -> bool:
            gpus[Path(path).name] = json.loads(Path(path).read_text())["_metadata"][
                "gpu"
            ]
            return True

    class FakeAutoTuner:
        @staticmethod
        def get() -> FakeTuner:
            return FakeTuner()

    module = types.ModuleType("flashinfer.autotuner")
    module.AutoTuner = FakeAutoTuner  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", module)
    fi.load_flashinfer_bf16_tuning_dir.cache_clear()
    yield gpus
    fi.load_flashinfer_bf16_tuning_dir.cache_clear()


def _set_gpu(
    monkeypatch: pytest.MonkeyPatch, name: str, capability: tuple[int, int]
) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: name)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: capability)


@pytest.mark.parametrize(
    "saved, current, capability, expected",
    [
        ("NVIDIA GB200", "NVIDIA B200", (10, 0), "NVIDIA B200"),
        ("NVIDIA GB200", "NVIDIA GB300", (10, 3), "NVIDIA GB300"),
        ("NVIDIA B300 SXM6 AC", "NVIDIA GB200", (10, 0), "NVIDIA GB200"),
        ("NVIDIA GB200", "NVIDIA GB200", (10, 0), "NVIDIA GB200"),
        ("NVIDIA GB200", "NVIDIA GeForce RTX 5090", (12, 0), "NVIDIA GB200"),
        ("NVIDIA GB200", "NVIDIA VR NVL72", (10, 7), "NVIDIA GB200"),
        ("NVIDIA GB200", "NVIDIA H100", (9, 0), "NVIDIA GB200"),
        ("NVIDIA VR NVL72", "NVIDIA B200", (10, 0), "NVIDIA VR NVL72"),
    ],
)
def test_load_tuning_dir_datacenter_blackwell_family(
    tuner_gpus: dict[str, str],
    tuning_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    saved: str,
    current: str,
    capability: tuple[int, int],
    expected: str,
):
    path = tuning_dir / "model.json"
    path.write_text(json.dumps({"_metadata": {"gpu": saved, "cuda_version": "13.0"}}))
    _set_gpu(monkeypatch, current, capability)
    fi.load_flashinfer_bf16_tuning_dir()
    assert tuner_gpus == {"model.json": expected}
    assert json.loads(path.read_text())["_metadata"]["gpu"] == saved
