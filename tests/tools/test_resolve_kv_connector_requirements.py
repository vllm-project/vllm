# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
import re
import subprocess
import sys

from tools.resolve_kv_connector_requirements import resolve_requirements


DOCKERFILE = Path(__file__).resolve().parents[2] / "docker" / "Dockerfile"


def _section_between(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[start:end]


def test_rewrites_nixl_for_cuda12():
    content = "lmcache >= 0.3.9\nnixl >= 0.7.1, < 0.10.0 # Required\n"

    resolved = resolve_requirements(content, cuda_major=12)

    assert resolved == (
        "lmcache >= 0.3.9\n"
        "nixl-cu12 >= 0.7.1, < 0.10.0 # Required\n"
    )


def test_rewrites_nixl_for_cuda13():
    content = "nixl >= 0.7.1, < 0.10.0 # Required\n"

    resolved = resolve_requirements(content, cuda_major=13)

    assert resolved == "nixl-cu13 >= 0.7.1, < 0.10.0 # Required\n"


def test_skips_requested_packages():
    content = (
        "# comment\n"
        "lmcache >= 0.3.9\n"
        "nixl >= 0.7.1, < 0.10.0 # Required\n"
        "mooncake-transfer-engine >= 0.3.8\n"
    )

    resolved = resolve_requirements(
        content,
        cuda_major=13,
        skip_packages={"lmcache"},
    )

    assert resolved == (
        "# comment\n"
        "nixl-cu13 >= 0.7.1, < 0.10.0 # Required\n"
        "mooncake-transfer-engine >= 0.3.8\n"
    )


def test_cuda13_resolution_never_emits_any_cu12_packages():
    content = (
        "lmcache >= 0.3.9\n"
        "nixl >= 0.7.1, < 0.10.0 # Required for disaggregated prefill\n"
        "mooncake-transfer-engine >= 0.3.8\n"
    )

    resolved = resolve_requirements(
        content,
        cuda_major=13,
        skip_packages={"lmcache"},
    )

    assert "lmcache" not in resolved
    assert "nixl-cu13" in resolved
    assert "\nnixl >=" not in f"\n{resolved}"
    assert re.search(r"\b[a-z0-9._-]*cu12[a-z0-9._-]*\b", resolved, re.IGNORECASE) is None


def test_repo_kv_connector_requirements_resolve_cleanly_for_cuda13():
    requirements_path = (
        Path(__file__).resolve().parents[2] / "requirements" / "kv_connectors.txt"
    )

    resolved = resolve_requirements(
        requirements_path.read_text(),
        cuda_major=13,
        skip_packages={"lmcache"},
    )

    assert "lmcache" not in resolved
    assert "nixl-cu13" in resolved
    assert re.search(r"\b[a-z0-9._-]*cu12[a-z0-9._-]*\b", resolved, re.IGNORECASE) is None


def test_cli_resolves_cuda13_requirements(tmp_path):
    input_path = tmp_path / "kv_connectors.txt"
    output_path = tmp_path / "kv_connectors.resolved.txt"
    input_path.write_text(
        "lmcache >= 0.3.9\n"
        "nixl >= 0.7.1, < 0.10.0 # Required for disaggregated prefill\n"
        "mooncake-transfer-engine >= 0.3.8\n"
    )

    subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[2]
                / "tools"
                / "resolve_kv_connector_requirements.py"
            ),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--cuda-major",
            "13",
            "--skip-package",
            "lmcache",
        ],
        check=True,
    )

    resolved = output_path.read_text()
    assert "lmcache" not in resolved
    assert "nixl-cu13" in resolved
    assert "nixl-cu12" not in resolved
    assert "\nnixl >=" not in f"\n{resolved}"


def test_dockerfile_uses_resolved_cuda13_requirements():
    dockerfile = DOCKERFILE.read_text()
    kv_section = _section_between(
        dockerfile,
        "# install kv_connectors if requested",
        "ENV VLLM_USAGE_SOURCE production-docker-image",
    )
    cuda13_branch = kv_section.split(
        'if [ "$CUDA_MAJOR" -ge 13 ]; then \\',
        maxsplit=1,
    )[1].split("else \\", maxsplit=1)[0]
    cuda12_branch = kv_section.split("else \\", maxsplit=1)[1]

    assert "tools/resolve_kv_connector_requirements.py" in dockerfile
    assert "/tmp/resolve_kv_connector_requirements.py" in dockerfile
    assert "/tmp/kv_connectors_cuda13_runtime.txt" in dockerfile
    assert '--cuda-major "${CUDA_MAJOR}"' in cuda13_branch
    assert "--skip-package lmcache" in cuda13_branch
    assert "uv pip install --system cupy-cuda13x" in cuda13_branch
    assert "--no-binary lmcache \\" in cuda13_branch
    assert "--no-deps \\" in cuda13_branch
    assert "--no-build ||" in cuda13_branch
    assert "--no-build-isolation &&" in cuda13_branch
    assert "cu12" not in cuda13_branch.lower()
    assert "uv pip install --system -r /tmp/kv_connectors.txt --no-build" in cuda12_branch
    assert "uv pip install --system -r /tmp/kv_connectors.txt --no-build-isolation" in cuda12_branch
    assert "cupy-cuda13x" not in cuda12_branch
    assert "nixl-cu13" not in cuda12_branch


def test_preserves_blank_lines():
    content = "\n# comment\n\nnixl >= 0.7.1\n"

    resolved = resolve_requirements(content, cuda_major=13)

    assert resolved == "\n# comment\n\nnixl-cu13 >= 0.7.1\n"
