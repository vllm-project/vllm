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


def _fake_flashinfer_autotuner(
    monkeypatch: pytest.MonkeyPatch, loaded: dict[str, str], accept: bool = True
) -> None:
    """Record the GPU name each tuning file has when FlashInfer loads it."""

    class FakeTuner:
        def load_configs(self, path: str) -> bool:
            loaded[Path(path).name] = json.loads(Path(path).read_text())[
                "_metadata"
            ].get("gpu", "")
            return accept

    class FakeAutoTuner:
        @staticmethod
        def get() -> FakeTuner:
            return FakeTuner()

    module = types.ModuleType("flashinfer.autotuner")
    module.AutoTuner = FakeAutoTuner  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", module)


@pytest.fixture
def tuner_gpus(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, str]]:
    """GPU name recorded in each tuning file at the moment FlashInfer loads it."""
    gpus: dict[str, str] = {}
    _fake_flashinfer_autotuner(monkeypatch, gpus)
    monkeypatch.setattr(fi, "_bf16_tuning_files_loaded", False)
    monkeypatch.setattr(fi, "_bf16_tuning_points", frozenset())
    fi.load_flashinfer_bf16_tuning.cache_clear()
    yield gpus
    fi.load_flashinfer_bf16_tuning.cache_clear()


@pytest.fixture
def tuning_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "gemmbf16_configs"
    directory.mkdir()
    monkeypatch.setattr(fi, "_BF16_TUNING_DIR", directory)
    return directory


def _set_gpu(
    monkeypatch: pytest.MonkeyPatch, name: str, capability: tuple[int, int]
) -> None:
    monkeypatch.setattr(
        fi, "current_platform", types.SimpleNamespace(is_cuda=lambda: True)
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: name)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: capability)


def _write_tuning_files(directory: Path, names: list[str], gpu: str = "NVIDIA GB200"):
    for name in names:
        (directory / name).write_text(json.dumps({"_metadata": {"gpu": gpu}}))


_FILES = [
    "Qwen3.8-27B-tp1-blackwell.json",
    "Qwen3.8-27B-tp2-blackwell.json",
    "Qwen3.8-Flash-Next-tp1-blackwell.json",
    "Qwen3.8-tp1-blackwell.json",
    "Other-tp1-hopper.json",
    "notes.json",
]


@pytest.mark.parametrize(
    "model_names, tp_size, capability, expected",
    [
        (("Qwen/Qwen3.8-27B",), 1, (10, 0), "Qwen3.8-27B-tp1-blackwell.json"),
        (("Qwen/Qwen3.8-27B",), 1, (10, 3), "Qwen3.8-27B-tp1-blackwell.json"),
        (("qwen3.8-27b",), 2, (10, 0), "Qwen3.8-27B-tp2-blackwell.json"),
        # a local path, and a checkpoint derived from the tuned model
        (("/models/Qwen3.8-27B-FP8/",), 1, (10, 0), "Qwen3.8-27B-tp1-blackwell.json"),
        (("my-name", "Qwen/Qwen3.8-27B"), 1, (10, 0), "Qwen3.8-27B-tp1-blackwell.json"),
        (
            ("Qwen/Qwen3.8-Flash-Next",),
            1,
            (10, 0),
            "Qwen3.8-Flash-Next-tp1-blackwell.json",
        ),
        # a name that continues the model of the file matches (the shorter file
        # "Qwen3.8" here), and the longest matching model wins over it
        (("Qwen/Qwen3.8-Dense",), 1, (10, 0), "Qwen3.8-tp1-blackwell.json"),
        (("Qwen/Qwen3.8-27B-FP8",), 1, (10, 0), "Qwen3.8-27B-tp1-blackwell.json"),
        (("Qwen/Qwen3.9-27B",), 1, (10, 0), None),
        (("Qwen/Qwen3.8-27B",), 4, (10, 0), None),
        (("Qwen/Qwen3.8-27B",), 1, (9, 0), None),
        (("Qwen/Qwen3.8-27B",), 1, (12, 0), None),
        (("Other",), 1, (10, 0), None),
        ((), 1, (10, 0), None),
    ],
)
def test_find_tuning_file_by_model_tp_and_gpu(
    tuning_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_names: tuple[str, ...],
    tp_size: int,
    capability: tuple[int, int],
    expected: str | None,
):
    _write_tuning_files(tuning_dir, _FILES)
    _set_gpu(monkeypatch, "NVIDIA GB200", capability)
    found = fi.find_flashinfer_bf16_tuning_file(model_names, tp_size)
    assert (found.name if found else None) == expected


def test_find_tuning_file_name_must_continue_after_a_separator(
    tuning_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_tuning_files(tuning_dir, ["Qwen3.8-27B-tp1-blackwell.json"])
    _set_gpu(monkeypatch, "NVIDIA GB200", (10, 0))
    assert fi.find_flashinfer_bf16_tuning_file(("Qwen/Qwen3.8-27BX",), 1) is None
    assert fi.find_flashinfer_bf16_tuning_file(("Qwen/Qwen3.8-27B_fp8",), 1) is not None


def test_find_tuning_file_missing_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(fi, "_BF16_TUNING_DIR", tmp_path / "missing")
    _set_gpu(monkeypatch, "NVIDIA GB200", (10, 0))
    assert fi.find_flashinfer_bf16_tuning_file(("Qwen3.8-27B",), 1) is None


def test_load_tuning_loads_only_the_matching_file(
    tuner_gpus: dict[str, str], tuning_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_tuning_files(tuning_dir, _FILES)
    key = (
        "('bf16_gemm_pdl', 'CudnnBf16GemmRunner', ((16, 5120), (5120, 34816), "
        "(0,), (0,), (-1, 34816), (-1,)), (torch.bfloat16, False, True, True))"
    )
    (tuning_dir / "Qwen3.8-27B-tp1-blackwell.json").write_text(
        json.dumps({"_metadata": {"gpu": "NVIDIA GB200"}, key: ["x"]})
    )
    _set_gpu(monkeypatch, "NVIDIA GB200", (10, 0))
    assert fi.load_flashinfer_bf16_tuning(("Qwen/Qwen3.8-27B",), 1)
    assert list(tuner_gpus) == ["Qwen3.8-27B-tp1-blackwell.json"]
    assert fi.is_flashinfer_bf16_tuning_loaded()
    monkeypatch.setattr(fi, "_bf16_gemm_tuning_buckets", (16, 32))
    assert fi.is_flashinfer_bf16_gemm_tuned(16, 34816, 5120)
    assert not fi.is_flashinfer_bf16_gemm_tuned(16, 5120, 34816)
    assert not fi.is_flashinfer_bf16_gemm_tuned(32, 34816, 5120)


def test_load_tuning_without_a_file_loads_nothing(
    tuner_gpus: dict[str, str], tuning_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_tuning_files(tuning_dir, _FILES)
    _set_gpu(monkeypatch, "NVIDIA GB200", (10, 0))
    assert not fi.load_flashinfer_bf16_tuning(("Unknown/Model",), 1)
    assert fi.flashinfer_bf16_unusable_tuning_file() is None
    assert tuner_gpus == {}
    assert not fi.is_flashinfer_bf16_tuning_loaded()
    monkeypatch.setattr(fi, "_bf16_gemm_tuning_buckets", (16, 32))
    assert not fi.is_flashinfer_bf16_gemm_tuned(16, 34816, 5120)


def test_load_tuning_rejected_by_flashinfer(
    tuning_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    loaded: dict[str, str] = {}
    _fake_flashinfer_autotuner(monkeypatch, loaded, accept=False)
    monkeypatch.setattr(fi, "_bf16_tuning_files_loaded", False)
    fi.load_flashinfer_bf16_tuning.cache_clear()
    _write_tuning_files(tuning_dir, _FILES)
    _set_gpu(monkeypatch, "NVIDIA GB200", (10, 0))
    assert not fi.load_flashinfer_bf16_tuning(("Qwen/Qwen3.8-27B",), 1)
    assert list(loaded) == ["Qwen3.8-27B-tp1-blackwell.json"]
    unusable = fi.flashinfer_bf16_unusable_tuning_file()
    assert unusable is not None and unusable.name == "Qwen3.8-27B-tp1-blackwell.json"
    assert not fi.is_flashinfer_bf16_tuning_loaded()
    fi.load_flashinfer_bf16_tuning.cache_clear()


@pytest.mark.parametrize(
    "saved, current, expected",
    [
        ("NVIDIA GB200", "NVIDIA B200", "NVIDIA B200"),
        ("NVIDIA GB200", "NVIDIA GB300", "NVIDIA GB300"),
        ("NVIDIA B300 SXM6 AC", "NVIDIA GB200", "NVIDIA GB200"),
        ("NVIDIA GB200", "NVIDIA GB200", "NVIDIA GB200"),
        # not a datacenter Blackwell name: the recorded name stays
        ("NVIDIA VR NVL72", "NVIDIA B200", "NVIDIA VR NVL72"),
    ],
)
def test_load_tuning_gpu_name_of_datacenter_blackwell_family(
    tuner_gpus: dict[str, str],
    tuning_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    saved: str,
    current: str,
    expected: str,
):
    path = tuning_dir / "model-tp4-blackwell.json"
    path.write_text(json.dumps({"_metadata": {"gpu": saved, "cuda_version": "13.0"}}))
    _set_gpu(monkeypatch, current, (10, 0))
    assert fi.load_flashinfer_bf16_tuning(("model",), 4)
    assert tuner_gpus == {"model-tp4-blackwell.json": expected}
    assert json.loads(path.read_text())["_metadata"]["gpu"] == saved


def test_tuning_file_checks_only_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "model.json"
    metadata = {
        "flashinfer_version": "0.7.4",
        "cuda_version": "13.0",
        "cublas_version": "13.8.0",
        "cudnn_version": "92400",
        "cudnn_frontend_version": "1.32.3",
        "cutlass_dsl_version": "4.8.0",
        "gpu": "NVIDIA H100",
    }
    source.write_text(json.dumps({"_metadata": metadata, "entry": [1]}))
    _set_gpu(monkeypatch, "NVIDIA H100", (9, 0))
    workdir = tmp_path / "work"
    workdir.mkdir()
    loaded = json.loads(fi._bf16_tuning_path_for_this_gpu(source, workdir).read_text())
    assert loaded["_metadata"] == {
        **{key: "*" for key in metadata},
        "gpu": "NVIDIA H100",
    }
    assert loaded["entry"] == [1]
    assert json.loads(source.read_text())["_metadata"] == metadata


@pytest.fixture
def cuda_dispatch(monkeypatch: pytest.MonkeyPatch):
    """dispatch_unquantized_gemm on a CUDA device where FlashInfer mm_bf16 works."""
    import vllm.model_executor.layers.utils as layer_utils

    monkeypatch.setattr(
        layer_utils,
        "current_platform",
        types.SimpleNamespace(
            is_rocm=lambda: False,
            is_cpu=lambda: False,
            is_cuda=lambda: True,
            is_arch_support_pdl=lambda: True,
        ),
    )
    monkeypatch.setitem(
        layer_utils._FLASHINFER_BF16_BACKENDS,
        "flashinfer",
        layer_utils._FlashInferBf16Backend(
            flashinfer_backend="auto",
            is_supported=lambda: True,
            can_implement=lambda *args: True,
        ),
    )
    monkeypatch.setattr(
        layer_utils, "init_flashinfer_bf16_tuning_buckets", lambda: None
    )
    return layer_utils


def test_dispatch_flashinfer_without_tuning_file_uses_torch(
    cuda_dispatch, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        cuda_dispatch, "load_flashinfer_bf16_tuning_for_current_model", lambda: False
    )
    assert (
        cuda_dispatch.dispatch_unquantized_gemm("flashinfer")
        is cuda_dispatch.cuda_unquantized_gemm
    )


def test_dispatch_flashinfer_with_tuning_file_uses_flashinfer(
    cuda_dispatch, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        cuda_dispatch, "load_flashinfer_bf16_tuning_for_current_model", lambda: True
    )
    gemm = cuda_dispatch.dispatch_unquantized_gemm("flashinfer")
    assert gemm.func is cuda_dispatch.cuda_flashinfer_bf16_gemm
    assert gemm.keywords["vllm_backend"] == "flashinfer"


_TUNED = frozenset({(16, 34816, 5120), (32, 34816, 5120), (11, 1024, 512)})


@pytest.mark.parametrize(
    "m, n, k, expected",
    [
        # M is rounded up to the capture size, and the file has that bucket
        (16, 34816, 5120, True),
        (12, 34816, 5120, True),
        (32, 34816, 5120, True),
        (17, 34816, 5120, True),
        # above the largest bucket FlashInfer clamps to it
        (100, 34816, 5120, True),
        # the capture size 11 is not tuned for this GEMM, though N and K are
        (11, 34816, 5120, False),
        (1, 34816, 5120, False),
        (16, 5120, 34816, False),
        (16, 34816, 4096, False),
        # a bucket tuned for another GEMM only
        (11, 1024, 512, True),
        (12, 1024, 512, False),
    ],
)
def test_gemm_tuned_by_m_bucket(
    monkeypatch: pytest.MonkeyPatch, m: int, n: int, k: int, expected: bool
):
    monkeypatch.setattr(fi, "_bf16_tuning_points", _TUNED)
    monkeypatch.setattr(fi, "_bf16_gemm_tuning_buckets", (1, 11, 16, 32))
    assert fi.is_flashinfer_bf16_gemm_tuned(m, n, k) is expected


def test_gemm_tuned_with_flashinfer_default_buckets(monkeypatch: pytest.MonkeyPatch):
    module = types.ModuleType("flashinfer.fused_moe.utils")
    module.map_to_hybrid_bucket_uncapped = lambda x: 16 if x <= 16 else 32  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "flashinfer.fused_moe.utils", module)
    monkeypatch.setattr(fi, "_bf16_tuning_points", _TUNED)
    monkeypatch.setattr(fi, "_bf16_gemm_tuning_buckets", None)
    assert fi.is_flashinfer_bf16_gemm_tuned(11, 34816, 5120)
    assert fi.is_flashinfer_bf16_gemm_tuned(20, 34816, 5120)
    assert not fi.is_flashinfer_bf16_gemm_tuned(11, 1024, 512)


def test_gemm_not_tuned_when_default_buckets_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(sys.modules, "flashinfer.fused_moe.utils", None)
    monkeypatch.setattr(fi, "_bf16_tuning_points", _TUNED)
    monkeypatch.setattr(fi, "_bf16_gemm_tuning_buckets", None)
    assert not fi.is_flashinfer_bf16_gemm_tuned(16, 34816, 5120)


def test_reload_loads_the_tuning_file_again(
    tuner_gpus: dict[str, str], tuning_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    key = (
        "('bf16_gemm_pdl', 'CudnnBf16GemmRunner', ((16, 5120), (5120, 34816), "
        "(0,), (0,), (-1, 34816), (-1,)), (torch.bfloat16, False, True, True))"
    )
    (tuning_dir / "model-tp1-blackwell.json").write_text(
        json.dumps({"_metadata": {"gpu": "NVIDIA GB200"}, key: ["x"]})
    )
    _set_gpu(monkeypatch, "NVIDIA GB200", (10, 0))
    monkeypatch.setattr(fi, "_bf16_gemm_tuning_buckets", (16,))
    assert fi.load_flashinfer_bf16_tuning(("model",), 1)
    tuner_gpus.clear()
    fi.reload_flashinfer_bf16_tuning()
    assert list(tuner_gpus) == ["model-tp1-blackwell.json"]
    assert fi.is_flashinfer_bf16_tuning_loaded()
    assert fi.is_flashinfer_bf16_gemm_tuned(16, 34816, 5120)


def test_reload_without_a_loaded_file_does_nothing(tuner_gpus: dict[str, str]):
    fi.reload_flashinfer_bf16_tuning()
    assert tuner_gpus == {}


def test_reload_failure_leaves_bf16_gemms_on_torch(
    tuning_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    accepted = {"load": True}

    class FakeTuner:
        def load_configs(self, path: str) -> bool:
            return accepted["load"]

    class FakeAutoTuner:
        @staticmethod
        def get() -> FakeTuner:
            return FakeTuner()

    module = types.ModuleType("flashinfer.autotuner")
    module.AutoTuner = FakeAutoTuner  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", module)
    monkeypatch.setattr(fi, "_bf16_tuning_files_loaded", False)
    monkeypatch.setattr(fi, "_bf16_tuning_points", frozenset())
    fi.load_flashinfer_bf16_tuning.cache_clear()
    key = (
        "('bf16_gemm_pdl', 'CudnnBf16GemmRunner', ((16, 5120), (5120, 34816), "
        "(0,), (0,), (-1, 34816), (-1,)), (torch.bfloat16, False, True, True))"
    )
    (tuning_dir / "model-tp1-blackwell.json").write_text(
        json.dumps({"_metadata": {"gpu": "NVIDIA GB200"}, key: ["x"]})
    )
    _set_gpu(monkeypatch, "NVIDIA GB200", (10, 0))
    monkeypatch.setattr(fi, "_bf16_gemm_tuning_buckets", (16,))
    assert fi.load_flashinfer_bf16_tuning(("model",), 1)
    accepted["load"] = False
    fi.reload_flashinfer_bf16_tuning()
    # Not loaded any more: the autotune warmup no longer skips bf16 GEMMs, and no
    # GEMM counts as tuned.
    assert not fi.is_flashinfer_bf16_tuning_loaded()
    assert not fi.is_flashinfer_bf16_gemm_tuned(16, 34816, 5120)
    fi.load_flashinfer_bf16_tuning.cache_clear()


def test_warn_flashinfer_bf16_untuned_warns_once_per_point(monkeypatch):
    from unittest.mock import MagicMock

    import vllm.utils.flashinfer as fi

    monkeypatch.setattr(fi, "_warned_bf16_points", set())
    monkeypatch.setattr(fi, "_bf16_lookup_bucket", lambda m: 8 if m <= 8 else None)
    monkeypatch.setattr(
        fi, "_bf16_tuning_points", frozenset({(8, 64, 128), (16, 64, 128)})
    )
    warn = MagicMock()
    monkeypatch.setattr(fi.logger, "warning", warn)

    # The file has (N=64, K=128), but not for these M buckets: once per bucket.
    fi.warn_flashinfer_bf16_untuned(5, 64, 128)
    fi.warn_flashinfer_bf16_untuned(7, 64, 128)  # same M bucket
    assert warn.call_count == 1
    fi.warn_flashinfer_bf16_untuned(100, 64, 128)  # another bucket
    assert warn.call_count == 2

    # The file has no (N=65, K=128) at all: once, whatever M.
    fi.warn_flashinfer_bf16_untuned(5, 65, 128)
    assert warn.call_count == 3
    fi.warn_flashinfer_bf16_untuned(100, 65, 128)
    fi.warn_flashinfer_bf16_untuned(3, 65, 128)
    assert warn.call_count == 3


def test_flashinfer_bf16_mm_warns_when_flashinfer_misses_the_cache(monkeypatch):
    import types
    from unittest.mock import MagicMock

    import torch

    import vllm.utils.flashinfer as fi

    stats = types.SimpleNamespace(cache_misses=0)
    calls: list[int] = []

    def fake_mm(a, b, bias, pdl, backend):
        if len(calls) % 2 == 0:  # the first, third, ... call misses
            stats.cache_misses += 1
        calls.append(1)
        return a

    monkeypatch.setattr(fi, "_bf16_tuning_files_loaded", True)
    monkeypatch.setattr(fi, "_warned_bf16_misses", set())
    monkeypatch.setattr(fi, "_bf16_lookup_bucket", lambda m: m)
    monkeypatch.setattr(fi, "_flashinfer_autotuner_stats", lambda: stats)
    monkeypatch.setattr(fi, "_flashinfer_bf16_mm", fake_mm)
    warn = MagicMock()
    monkeypatch.setattr(fi.logger, "warning", warn)

    a, b = torch.empty(4, 8), torch.empty(8, 16)
    fi.flashinfer_bf16_mm(a, b, None, True, "auto")  # miss
    assert warn.call_count == 1
    assert warn.call_args.args[1:] == (16, 8, 4)  # N, K, M bucket
    fi.flashinfer_bf16_mm(a, b, None, True, "auto")  # hit
    fi.flashinfer_bf16_mm(a, b, None, True, "auto")  # miss again, same shape
    assert warn.call_count == 1  # reported once
    fi.flashinfer_bf16_mm(a, b, None, True, "auto")  # hit
    fi.flashinfer_bf16_mm(torch.empty(6, 8), b, None, True, "auto")  # miss, other M
    assert warn.call_count == 1  # the shape (N, K) is reported once


def test_flashinfer_bf16_mm_is_silent_without_a_tuning_file(monkeypatch):
    import types
    from unittest.mock import MagicMock

    import torch

    import vllm.utils.flashinfer as fi

    stats = types.SimpleNamespace(cache_misses=0)

    def fake_mm(a, b, bias, pdl, backend):
        stats.cache_misses += 1
        return a

    monkeypatch.setattr(fi, "_bf16_tuning_files_loaded", False)
    monkeypatch.setattr(fi, "_flashinfer_autotuner_stats", lambda: stats)
    monkeypatch.setattr(fi, "_flashinfer_bf16_mm", fake_mm)
    warn = MagicMock()
    monkeypatch.setattr(fi.logger, "warning", warn)
    fi.flashinfer_bf16_mm(torch.empty(4, 8), torch.empty(8, 16), None, True, "auto")
    assert warn.call_count == 0


def test_dispatch_without_a_tuning_file_only_logs_info(
    cuda_dispatch, monkeypatch: pytest.MonkeyPatch
):
    from unittest.mock import MagicMock

    monkeypatch.setattr(
        cuda_dispatch, "load_flashinfer_bf16_tuning_for_current_model", lambda: False
    )
    monkeypatch.setattr(
        cuda_dispatch, "flashinfer_bf16_unusable_tuning_file", lambda: None
    )
    warning, info = MagicMock(), MagicMock()
    monkeypatch.setattr(cuda_dispatch.logger, "warning_once", warning)
    monkeypatch.setattr(cuda_dispatch.logger, "info_once", info)
    assert (
        cuda_dispatch.dispatch_unquantized_gemm("flashinfer")
        is cuda_dispatch.cuda_unquantized_gemm
    )
    assert info.call_count == 1
    assert warning.call_count == 0


def test_dispatch_with_an_unusable_tuning_file_warns(
    cuda_dispatch, monkeypatch: pytest.MonkeyPatch
):
    from unittest.mock import MagicMock

    monkeypatch.setattr(
        cuda_dispatch, "load_flashinfer_bf16_tuning_for_current_model", lambda: False
    )
    monkeypatch.setattr(
        cuda_dispatch,
        "flashinfer_bf16_unusable_tuning_file",
        lambda: Path("Qwen3.8-27B-tp1-blackwell.json"),
    )
    warning, info = MagicMock(), MagicMock()
    monkeypatch.setattr(cuda_dispatch.logger, "warning_once", warning)
    monkeypatch.setattr(cuda_dispatch.logger, "info_once", info)
    assert (
        cuda_dispatch.dispatch_unquantized_gemm("flashinfer")
        is cuda_dispatch.cuda_unquantized_gemm
    )
    assert warning.call_count == 1
    assert warning.call_args.args[1] == "Qwen3.8-27B-tp1-blackwell.json"
    assert info.call_count == 0
