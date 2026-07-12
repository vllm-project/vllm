# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / ".buildkite" / "scripts" / "hardware_ci" / "run-amd-test.sh"
CI_BAKE = REPO_ROOT / ".buildkite" / "scripts" / "ci-bake-rocm.sh"

EXPECTED_COMMIT = "0123456789abcdef"
EXPECTED_BASE_IMAGE = "rocm/vllm-dev:ci_base-test"
WHEEL_NAME = "vllm-0.0.0-cp38-abi3-manylinux_2_28_x86_64.whl"


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


def _create_artifact(
    tmp_path: Path,
    *,
    recorded_commit: str = EXPECTED_COMMIT,
    recorded_base: str = EXPECTED_BASE_IMAGE,
    wheel_names: tuple[str, ...] = (WHEEL_NAME,),
) -> tuple[Path, Path]:
    payload = tmp_path / "payload"
    metadata = payload / ".vllm-ci-artifact"
    metadata.mkdir(parents=True)

    (metadata / "commit.txt").write_text(f"{recorded_commit}\n")
    (metadata / "native-base-image.txt").write_text(f"{recorded_base}\n")
    (metadata / "wheel-filename.txt").write_text(f"{wheel_names[0]}\n")
    (metadata / "ci-base-image.txt").write_text("ci-base\n")
    (metadata / "fallback-image.txt").write_text("fallback\n")

    for required_dir in ("tests", ".buildkite", "requirements"):
        directory = payload / required_dir
        directory.mkdir()
        (directory / "artifact-marker.txt").write_text(required_dir)

    for wheel_name in wheel_names:
        (payload / wheel_name).write_bytes(b"not-a-real-wheel")

    source = tmp_path / "artifact-source"
    source.mkdir()
    archive = source / "vllm-rocm-install.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for child in payload.iterdir():
            tar.add(child, arcname=child.name)

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = source / "vllm-rocm-install.tar.gz.sha256"
    checksum.write_text(f"{digest}  {archive.name}\n")
    return archive, checksum


def _create_fake_bin(tmp_path: Path) -> Path:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()

    _write_executable(
        fake_bin / "buildkite-agent",
        """#!/bin/bash
set -u
printf '%s\n' "$*" >> "${FAKE_BUILDKITE_LOG}"
if [[ "${1:-}" != "artifact" || "${2:-}" != "download" ]]; then
  exit 2
fi
query="${3}"
destination="${4}"
output_dir="${destination}/artifacts/vllm-rocm-install"
mkdir -p "${output_dir}"

case "${query}" in
  *.sha256)
    attempts=0
    if [[ -f "${FAKE_CHECKSUM_STATE}" ]]; then
      read -r attempts < "${FAKE_CHECKSUM_STATE}"
    fi
    attempts=$((attempts + 1))
    printf '%s\n' "${attempts}" > "${FAKE_CHECKSUM_STATE}"
    if (( attempts <= FAKE_CHECKSUM_FAILURES )); then
      exit 1
    fi
    cp "${FAKE_CHECKSUM}" \
      "${output_dir}/vllm-rocm-install.tar.gz.sha256"
    ;;
  *)
    cp "${FAKE_ARCHIVE}" "${output_dir}/vllm-rocm-install.tar.gz"
    ;;
esac
""",
    )
    _write_executable(
        fake_bin / "python3",
        """#!/bin/bash
printf '%s\n' "$*" >> "${FAKE_PYTHON_LOG}"
exit 0
""",
    )
    _write_executable(
        fake_bin / "id",
        """#!/bin/bash
if [[ "${1:-}" == "-u" ]]; then
  printf '0\n'
  exit 0
fi
exec /usr/bin/id "$@"
""",
    )
    _write_executable(fake_bin / "sleep", "#!/bin/bash\nexit 0\n")
    return fake_bin


def _native_env(
    tmp_path: Path,
    archive: Path,
    checksum: Path,
    *,
    checksum_failures: int = 0,
    workspace: Path | None = None,
    checkout: Path | None = None,
) -> dict[str, str]:
    fake_bin = _create_fake_bin(tmp_path)
    workspace = workspace or tmp_path / "workspace"
    checkout = checkout or tmp_path / "checkout"
    temp_dir = tmp_path / "tmp"
    hf_home = tmp_path / "huggingface"
    temp_dir.mkdir()

    env = os.environ.copy()
    for name in ("AMD_CI_RUNTIME", "NATIVE_CI", "NUM_NODES", "VLLM_TEST_COMMANDS"):
        env.pop(name, None)
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "VLLM_CI_RUN_AMD_TEST_LIB_ONLY": "1",
            "VLLM_CI_USE_ARTIFACTS": "1",
            "VLLM_CI_ARTIFACT_GLOB": (
                "artifacts/vllm-rocm-install/vllm-rocm-install.tar.gz"
            ),
            "VLLM_CI_ARTIFACT_STEP": "image-build-amd",
            "VLLM_CI_REQUIRE_WORKSPACE_MOUNT": "0",
            "VLLM_CI_REQUIRE_PERSISTENT_HF_CACHE": "0",
            "VLLM_CI_WORKSPACE": str(workspace),
            "VLLM_CI_BASE_IMAGE": EXPECTED_BASE_IMAGE,
            "BUILDKITE_BUILD_CHECKOUT_PATH": str(checkout),
            "BUILDKITE_COMMIT": EXPECTED_COMMIT,
            "BUILDKITE_JOB_ID": "native-runner-test",
            "TMPDIR": str(temp_dir),
            "HF_HOME": str(hf_home),
            "FAKE_ARCHIVE": str(archive),
            "FAKE_CHECKSUM": str(checksum),
            "FAKE_CHECKSUM_FAILURES": str(checksum_failures),
            "FAKE_CHECKSUM_STATE": str(tmp_path / "checksum-attempts"),
            "FAKE_BUILDKITE_LOG": str(tmp_path / "buildkite-agent.log"),
            "FAKE_PYTHON_LOG": str(tmp_path / "python3.log"),
        }
    )
    return env


def _run_prepare(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"\nartifact_work_dir=""\nprepare_native_workspace',
            "run-amd-test",
            str(RUNNER),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _producer_env(tmp_path: Path, *, upload_failure: bool = False) -> dict[str, str]:
    fake_bin = tmp_path / "producer-bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "buildkite-agent",
        """#!/bin/bash
printf '%s\n' "$*" >> "${FAKE_BUILDKITE_LOG}"
if [[ "${FAKE_UPLOAD_FAILURE}" == "1" ]]; then
  exit 1
fi
exit 0
""",
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "VLLM_CI_BAKE_ROCM_LIB_ONLY": "1",
            "UPLOAD_ROCM_WHEEL_ARTIFACTS": "1",
            "BUILDKITE": "true",
            "BUILDKITE_COMMIT": EXPECTED_COMMIT,
            "CI_BASE_IMAGE_TAG_COMMIT_REF": EXPECTED_BASE_IMAGE,
            "CI_BASE_IMAGE": "rocm/vllm-dev:ci_base-content",
            "IMAGE_TAG": "rocm/vllm-ci:test",
            "FAKE_UPLOAD_FAILURE": "1" if upload_failure else "0",
            "FAKE_BUILDKITE_LOG": str(tmp_path / "producer-agent.log"),
        }
    )
    return env


def _run_producer(
    tmp_path: Path,
    *,
    wheel_names: tuple[str, ...],
    upload_failure: bool = False,
) -> subprocess.CompletedProcess[str]:
    wheel_export = tmp_path / "wheel-export"
    wheel_export.mkdir()
    for directory in ("tests", ".buildkite", "requirements"):
        (wheel_export / directory).mkdir()
    for wheel_name in wheel_names:
        (wheel_export / wheel_name).write_bytes(b"wheel")
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"\nupload_wheel_artifacts_if_present',
            "ci-bake-rocm",
            str(CI_BAKE),
        ],
        cwd=tmp_path,
        env=_producer_env(tmp_path, upload_failure=upload_failure),
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_rocm_artifact_producer_packages_manifest_and_checksum(tmp_path: Path) -> None:
    result = _run_producer(tmp_path, wheel_names=(WHEEL_NAME,))

    assert result.returncode == 0, result.stdout + result.stderr
    artifact_dir = tmp_path / "artifacts" / "vllm-rocm-install"
    archive = artifact_dir / "vllm-rocm-install.tar.gz"
    checksum = artifact_dir / "vllm-rocm-install.tar.gz.sha256"
    expected_digest = checksum.read_text().split()[0]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected_digest
    with tarfile.open(archive, "r:gz") as tar:
        members = {name.removeprefix("./") for name in tar.getnames()}
        assert WHEEL_NAME in members
        assert ".vllm-ci-artifact/commit.txt" in members
        assert ".vllm-ci-artifact/native-base-image.txt" in members
        assert ".vllm-ci-artifact/wheel-filename.txt" in members
    assert (artifact_dir / "commit.txt").read_text().strip() == EXPECTED_COMMIT
    assert (artifact_dir / "native-base-image.txt").read_text().strip() == (
        EXPECTED_BASE_IMAGE
    )
    assert (artifact_dir / "wheel-filename.txt").read_text().strip() == WHEEL_NAME
    assert (
        "artifact upload artifacts/vllm-rocm-install/*"
        in (tmp_path / "producer-agent.log").read_text()
    )
    assert not (tmp_path / "wheel-export").exists()


@pytest.mark.parametrize("wheel_names", [(), (WHEEL_NAME, "second.whl")])
def test_rocm_artifact_producer_rejects_wrong_wheel_count(
    tmp_path: Path,
    wheel_names: tuple[str, ...],
) -> None:
    result = _run_producer(tmp_path, wheel_names=wheel_names)

    assert result.returncode != 0
    assert f"found {len(wheel_names)}" in result.stderr
    assert not (tmp_path / "producer-agent.log").exists()


def test_rocm_artifact_producer_propagates_upload_failure(tmp_path: Path) -> None:
    result = _run_producer(
        tmp_path,
        wheel_names=(WHEEL_NAME,),
        upload_failure=True,
    )

    assert result.returncode != 0
    assert (
        "artifact upload artifacts/vllm-rocm-install/*"
        in (tmp_path / "producer-agent.log").read_text()
    )
    assert (tmp_path / "wheel-export").exists()


def test_native_artifact_retries_when_checksum_is_initially_missing(
    tmp_path: Path,
) -> None:
    archive, checksum = _create_artifact(tmp_path)
    env = _native_env(tmp_path, archive, checksum, checksum_failures=1)

    result = _run_prepare(env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "checksum-attempts").read_text().strip() == "2"
    assert (tmp_path / "workspace" / "tests" / "artifact-marker.txt").is_file()
    agent_log = (tmp_path / "buildkite-agent.log").read_text()
    assert agent_log.count("vllm-rocm-install.tar.gz.sha256") == 2
    assert "--step image-build-amd" in agent_log


@pytest.mark.parametrize(
    ("recorded_commit", "recorded_base", "expected_message"),
    [
        (
            "wrong-commit",
            EXPECTED_BASE_IMAGE,
            "ROCm artifact commit wrong-commit does not match " + EXPECTED_COMMIT,
        ),
        (
            EXPECTED_COMMIT,
            "rocm/wrong-base:latest",
            "ROCm artifact base rocm/wrong-base:latest does not match "
            + EXPECTED_BASE_IMAGE,
        ),
    ],
)
def test_native_artifact_rejects_wrong_provenance(
    tmp_path: Path,
    recorded_commit: str,
    recorded_base: str,
    expected_message: str,
) -> None:
    archive, checksum = _create_artifact(
        tmp_path,
        recorded_commit=recorded_commit,
        recorded_base=recorded_base,
    )
    env = _native_env(tmp_path, archive, checksum)

    result = _run_prepare(env)

    assert result.returncode != 0
    assert expected_message in result.stderr


def test_native_artifact_rejects_duplicate_top_level_wheels(tmp_path: Path) -> None:
    archive, checksum = _create_artifact(
        tmp_path,
        wheel_names=(WHEEL_NAME, "second-wheel-0.0.0-py3-none-any.whl"),
    )
    env = _native_env(tmp_path, archive, checksum)

    result = _run_prepare(env)

    assert result.returncode != 0
    assert "must contain exactly one top-level wheel; found 2" in result.stderr


@pytest.mark.parametrize(
    "relationship",
    ["same", "checkout-parent", "workspace-parent"],
)
def test_native_workspace_refuses_to_overlap_checkout(
    tmp_path: Path,
    relationship: str,
) -> None:
    paths = tmp_path / "paths"
    if relationship == "same":
        workspace = checkout = paths / "workspace"
    elif relationship == "checkout-parent":
        checkout = paths / "checkout"
        workspace = checkout / "workspace"
    else:
        workspace = paths / "workspace"
        checkout = workspace / "checkout"

    workspace.mkdir(parents=True)
    checkout.mkdir(parents=True, exist_ok=True)
    sentinel = workspace / "keep-me"
    sentinel.write_text("legitimate checkout content")
    archive, checksum = _create_artifact(tmp_path)
    env = _native_env(
        tmp_path,
        archive,
        checksum,
        workspace=workspace,
        checkout=checkout,
    )

    result = _run_prepare(env)

    assert result.returncode != 0
    assert "Refusing to replace" in result.stderr
    assert sentinel.read_text() == "legitimate checkout content"
    assert not (tmp_path / "python3.log").exists()
    assert not (tmp_path / "buildkite-agent.log").exists()


def test_native_child_shell_propagates_pipeline_failure(tmp_path: Path) -> None:
    archive, checksum = _create_artifact(tmp_path)
    env = _native_env(tmp_path, archive, checksum)
    env.update(
        {
            "AMD_CI_RUNTIME": "native",
            "VLLM_CI_RUN_AMD_TEST_LIB_ONLY": "0",
            "VLLM_CI_EXPECTED_GPU_COUNT": "0",
            "VLLM_TEST_COMMANDS": "false | true",
        }
    )

    result = subprocess.run(
        ["bash", str(RUNNER)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "Native test commands: false | true" in result.stdout
    assert "Native CPU-only AMD job" in result.stdout
