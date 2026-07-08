# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/scripts/resolve_perfgate_spec_file.py"
)


def load_resolver():
    spec = importlib.util.spec_from_file_location(
        "resolve_perfgate_spec_file",
        SCRIPT_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_fake_benchmark_repo(tmp_path: Path) -> Path:
    package_dir = tmp_path / "src" / "vllm_hust_benchmark"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "perfgate_specs.py").write_text(
        """
from pathlib import Path


def resolve_perfgate_spec_file(*, scenario, hardware_chip_model, repo_root):
    entries = {
        ("random-online", "910B2"): (
            "docs/official-baselines/perfgate-random-910b2.json"
        ),
        ("sharegpt-online", "910B2"): (
            "docs/official-baselines/perfgate-sharegpt-910b2.json"
        ),
    }
    key = (scenario, hardware_chip_model.upper())
    if key not in entries:
        raise ValueError(
            "No perfgate spec registered for "
            f"scenario={scenario}, hardware_chip_model={hardware_chip_model}"
        )
    return (Path(repo_root) / entries[key]).resolve()
""".lstrip(),
        encoding="utf-8",
    )
    for spec_file in (
        "docs/official-baselines/perfgate-random-910b2.json",
        "docs/official-baselines/perfgate-sharegpt-910b2.json",
    ):
        spec_path = tmp_path / spec_file
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text("{}", encoding="utf-8")
    return tmp_path


@pytest.fixture(autouse=True)
def clear_fake_benchmark_package():
    yield
    sys.modules.pop("vllm_hust_benchmark.perfgate_specs", None)
    sys.modules.pop("vllm_hust_benchmark", None)


def test_detect_chip_model_from_npu_smi_text() -> None:
    resolver = load_resolver()

    assert resolver.detect_chip_model_from_text("Name: Ascend 910B2") == "910B2"
    assert resolver.detect_chip_model_from_text("chip model: 910B3") == "910B3"
    assert resolver.detect_chip_model_from_text("chip model: 910B4") == "910B4"


def test_resolve_spec_file_prefers_explicit_override() -> None:
    resolver = load_resolver()

    spec_file, chip_model, spec_path = resolver.resolve_spec_file(
        explicit_spec_file="docs/official-baselines/custom.json",
        scenario="random-online",
        explicit_chip_model="",
        npu_smi_bin="",
        benchmark_repo="",
    )

    assert spec_file == "docs/official-baselines/custom.json"
    assert chip_model == ""
    assert spec_path is None


def test_resolve_spec_file_uses_shared_registry_for_scenario_and_chip(
    tmp_path: Path,
) -> None:
    resolver = load_resolver()
    benchmark_repo = write_fake_benchmark_repo(tmp_path)

    spec_file, chip_model, spec_path = resolver.resolve_spec_file(
        explicit_spec_file="",
        scenario="sharegpt-online",
        explicit_chip_model="910B2",
        npu_smi_bin="",
        benchmark_repo=str(benchmark_repo),
    )

    assert spec_file == "docs/official-baselines/perfgate-sharegpt-910b2.json"
    assert chip_model == "910B2"
    assert spec_path == (
        benchmark_repo / "docs/official-baselines/perfgate-sharegpt-910b2.json"
    )


def test_resolve_spec_file_fails_closed_for_unknown_chip_model(
    tmp_path: Path,
) -> None:
    resolver = load_resolver()

    with pytest.raises(ValueError, match="No perfgate spec registered"):
        resolver.resolve_spec_file(
            explicit_spec_file="",
            scenario="random-online",
            explicit_chip_model="910A",
            npu_smi_bin="",
            benchmark_repo=str(write_fake_benchmark_repo(tmp_path)),
        )


def test_main_writes_absolute_same_spec_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver = load_resolver()
    benchmark_repo = write_fake_benchmark_repo(tmp_path)
    spec_file = (
        benchmark_repo / "docs" / "official-baselines" / "perfgate-random-910b2.json"
    )
    github_env = tmp_path / "github-env"

    monkeypatch.setattr(
        "sys.argv",
        [
            "resolve_perfgate_spec_file.py",
            "--explicit-chip-model",
            "910B2",
            "--scenario",
            "random-online",
            "--benchmark-repo",
            str(benchmark_repo),
            "--github-env",
            str(github_env),
        ],
    )

    assert resolver.main() == 0
    env_text = github_env.read_text(encoding="utf-8")
    assert f"SAME_SPEC_SPEC_FILE={spec_file}" in env_text


def test_main_fails_when_shared_registry_has_no_matching_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    resolver = load_resolver()
    benchmark_repo = write_fake_benchmark_repo(tmp_path)
    github_env = tmp_path / "github-env"

    monkeypatch.setattr(
        "sys.argv",
        [
            "resolve_perfgate_spec_file.py",
            "--explicit-chip-model",
            "910B3",
            "--benchmark-repo",
            str(benchmark_repo),
            "--github-env",
            str(github_env),
        ],
    )

    assert resolver.main() == 2
    captured = capsys.readouterr()
    assert "No perfgate spec registered" in captured.err
    assert not github_env.exists()
