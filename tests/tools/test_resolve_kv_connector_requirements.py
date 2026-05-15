# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys
from pathlib import Path

from tools.resolve_kv_connector_requirements import resolve_requirements

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
KV_CONNECTOR_REQUIREMENTS = REPO_ROOT / "requirements" / "kv_connectors.txt"


def _section_between(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[start:end]


def test_rewrites_kv_connector_requirements_for_cuda12() -> None:
    content = (
        "lmcache >= 0.3.9\n"
        "nixl >= 1.1.0 # Required for disaggregated prefill\n"
        "mooncake-transfer-engine >= 0.3.8\n"
    )

    resolved = resolve_requirements(content, cuda_major=12)

    assert resolved == (
        "lmcache >= 0.3.9, < 0.4.5\n"
        "nixl-cu12 >= 1.1.0 # Required for disaggregated prefill\n"
        "mooncake-transfer-engine >= 0.3.8\n"
    )


def test_rewrites_kv_connector_requirements_for_cuda13() -> None:
    content = (
        "lmcache >= 0.3.9\n"
        "nixl >= 1.1.0 # Required for disaggregated prefill\n"
        "mooncake-transfer-engine >= 0.3.8\n"
    )

    resolved = resolve_requirements(content, cuda_major=13)

    assert resolved == (
        "lmcache >= 0.4.5\n"
        "nixl-cu13 >= 1.1.0 # Required for disaggregated prefill\n"
        "mooncake-transfer-engine >= 0.3.8\n"
    )


def test_preserves_blank_lines_comments_and_indentation() -> None:
    content = "\n# comment\n  lmcache >= 0.3.9 # cache connector\n\nnixl >= 1.1.0\n"

    resolved = resolve_requirements(content, cuda_major=13)

    assert resolved == (
        "\n# comment\n  lmcache >= 0.4.5 # cache connector\n\nnixl-cu13 >= 1.1.0\n"
    )


def test_repo_kv_connector_requirements_resolve_cleanly_for_cuda12() -> None:
    resolved = resolve_requirements(
        KV_CONNECTOR_REQUIREMENTS.read_text(),
        cuda_major=12,
    )

    assert "lmcache >= 0.3.9, < 0.4.5" in resolved
    assert "nixl-cu12 >= 1.1.0" in resolved
    assert "cupy-cuda13x" not in resolved
    assert "\nnixl >=" not in f"\n{resolved}"


def test_repo_kv_connector_requirements_resolve_cleanly_for_cuda13() -> None:
    resolved = resolve_requirements(
        KV_CONNECTOR_REQUIREMENTS.read_text(),
        cuda_major=13,
    )

    assert "lmcache >= 0.4.5" in resolved
    assert "nixl-cu13 >= 1.1.0" in resolved
    assert "cupy-cuda12x" not in resolved
    assert "\nnixl >=" not in f"\n{resolved}"


def test_cli_resolves_cuda13_requirements(tmp_path: Path) -> None:
    input_path = tmp_path / "kv_connectors.txt"
    output_path = tmp_path / "kv_connectors.resolved.txt"
    input_path.write_text(
        "lmcache >= 0.3.9\n"
        "nixl >= 1.1.0 # Required for disaggregated prefill\n"
        "mooncake-transfer-engine >= 0.3.8\n"
    )

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "resolve_kv_connector_requirements.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--cuda-major",
            "13",
        ],
        check=True,
    )

    resolved = output_path.read_text()
    assert "lmcache >= 0.4.5" in resolved
    assert "nixl-cu13 >= 1.1.0" in resolved
    assert "nixl-cu12" not in resolved


def test_dockerfile_uses_resolved_kv_connector_requirements() -> None:
    dockerfile = DOCKERFILE.read_text()
    kv_section = _section_between(
        dockerfile,
        "# install kv_connectors if requested",
        "ENV VLLM_USAGE_SOURCE production-docker-image",
    )

    assert "tools/resolve_kv_connector_requirements.py" in kv_section
    assert "/tmp/kv_connectors.resolved.txt" in kv_section
    assert "python3 /tmp/resolve_kv_connector_requirements.py" in kv_section
    assert '--cuda-major "${CUDA_MAJOR}"' in kv_section
    assert "uv pip install --system -r /tmp/kv_connectors.resolved.txt --no-build" in (
        kv_section
    )
    assert (
        "uv pip install --system -r /tmp/kv_connectors.resolved.txt "
        "--no-build-isolation"
    ) in kv_section
