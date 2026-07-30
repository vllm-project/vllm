# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_BAKE = REPO_ROOT / ".buildkite/scripts/ci-bake-rocm.sh"
BUILD_TEST_IMAGE = REPO_ROOT / ".buildkite/scripts/rocm/build-test-image.sh"
AMD_PIPELINE = REPO_ROOT / ".buildkite/hardware_tests/amd.yaml"


def run_bash(
    command: str,
    *args: Path,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    clean_env = os.environ.copy()
    for key in tuple(clean_env):
        if key.startswith(("BUILDKITE", "CI_BASE_", "ROCM_")) or key in {
            "BASE_IMAGE",
            "FORCE_BUILD",
            "IMAGE_TAG",
            "NIGHTLY",
            "REMOTE_VLLM",
            "TARGET",
            "VLLM_BAKE_FILE",
            "VLLM_BRANCH",
            "VLLM_REPO",
        }:
            clean_env.pop(key)
    clean_env.update(env or {})

    result = subprocess.run(
        [
            "bash",
            "-c",
            "buildkite-agent() { return 127; }\n" + command,
            "rocm-ci-cache-test",
            *(str(arg) for arg in args),
        ],
        check=False,
        cwd=REPO_ROOT,
        env=clean_env,
        capture_output=True,
        text=True,
    )
    if check:
        result.check_returncode()
    return result


def run_sourced(
    script: Path,
    command: str,
    *args: Path,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return run_bash(
        'source "$1"\nshift\n' + command,
        script,
        *args,
        env=env,
        check=check,
    )


def write_fake_docker(tmp_path: Path, body: str) -> tuple[Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "docker-calls"
    docker = fake_bin / "docker"
    docker.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    docker.chmod(0o755)
    return fake_bin, calls


def test_ci_base_hcl_and_hash_input_contract() -> None:
    contract = run_sourced(
        CI_BAKE,
        'printf "%s\\n%s\\n%s\\n" "$DEFAULT_CI_BASE_CONTENT_FILES" '
        '"$DEFAULT_CI_BASE_DOCKERFILE" "$DEFAULT_CI_BASE_DOCKERFILE_STAGES"',
    ).stdout.splitlines()
    files, dockerfile, stages = contract

    expected_files = """
        .dockerignore .buildkite/scripts/ci-bake-rocm.sh
        .buildkite/scripts/rocm/build-ci-base.sh
        docker/ci-rocm.hcl docker/docker-bake-rocm.hcl requirements/common.txt
        requirements/rocm.txt requirements/test/rocm.txt rust-toolchain.toml
        tests/vllm_test_utils tools/install_protoc.sh
        tools/install_torchcodec_rocm.sh
    """
    assert set(expected_files.split()) == set(files.split())
    assert dockerfile == "docker/Dockerfile.rocm"
    required_stages = (
        "base rust-toolchain build_nixl build_rocshmem build_deepep mori_base ci_base"
    )
    assert set(required_stages.split()) <= set(stages.split())

    ci = (REPO_ROOT / "docker/ci-rocm.hcl").read_text()
    assert "CI_BASE_STABLE_CACHE_REF" in ci


def test_content_hash_tracks_changes_and_fails_closed(tmp_path: Path) -> None:
    content = tmp_path / "content"
    content.mkdir()
    source = content / "input.py"
    source.write_text("VALUE = 1\n")
    alternate = content / "alternate.py"
    alternate.write_text("VALUE = 3\n")
    command = (
        'list_content_files() { find "$1" \\( -type f -o -type l \\) -print0; }\n'
        'compute_content_hash "$1"'
    )

    initial = run_sourced(CI_BAKE, command, content).stdout.strip()
    source.write_text("VALUE = 2\n")
    changed = run_sourced(CI_BAKE, command, content).stdout.strip()
    source.chmod(0o755)
    executable = run_sourced(CI_BAKE, command, content).stdout.strip()
    (content / "link.py").symlink_to("input.py")
    linked = run_sourced(CI_BAKE, command, content).stdout.strip()
    (content / "link.py").unlink()
    (content / "link.py").symlink_to("alternate.py")
    relinked = run_sourced(CI_BAKE, command, content).stdout.strip()

    assert len({initial, changed, executable, linked, relinked}) == 5

    failed = run_sourced(
        CI_BAKE,
        'list_content_files() { return 42; }\ncompute_content_hash "$1"',
        content,
        check=False,
    )
    assert failed.returncode != 0
    assert "failed to hash content" in failed.stderr


def test_one_resolved_base_digest_drives_hash_and_build(tmp_path: Path) -> None:
    docker_dir = tmp_path / "docker"
    docker_dir.mkdir()
    dockerfile = docker_dir / "Dockerfile.rocm"
    dockerfile.write_text(
        "ARG BASE_IMAGE=rocm/example:base\nFROM ${BASE_IMAGE} AS base\n"
    )
    bake_file = docker_dir / "docker-bake-rocm.hcl"
    bake_file.write_text("")
    content = tmp_path / "content"
    content.write_text("cache input\n")
    override = tmp_path / "override.hcl"
    digest = "sha256:" + "a" * 64
    fake_bin, calls = write_fake_docker(
        tmp_path,
        f"""
printf 'call\\n' >> "$FAKE_DOCKER_CALLS"
printf 'Digest: {digest}\\n'
""",
    )
    output = run_sourced(
        CI_BAKE,
        'CI_BASE_DOCKERFILE="$1"\n'
        'CI_BASE_CONTENT_FILES="$2"\n'
        'CI_BASE_DOCKERFILE_STAGES="base"\n'
        'VLLM_BAKE_FILE="$3"\n'
        'ROCM_ARG_OVERRIDE_PATH="$4"\n'
        "pin_base_image >/dev/null\n"
        'hash_dockerfile_arg_values "$1" BASE_IMAGE\n'
        "ci_base_metadata_pairs\n"
        "write_rocm_build_arg_override\n"
        'cat "$4"\n'
        'BASE_IMAGE="rocm/example:equivalent-alias"\n'
        'hash_dockerfile_arg_values "$1" BASE_IMAGE',
        dockerfile,
        content,
        bake_file,
        override,
        env={
            "FAKE_DOCKER_CALLS": str(calls),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        },
    ).stdout

    assert calls.read_text().splitlines() == ["call", "call"]
    assert output.count(f"arg:BASE_IMAGE.digest={digest}") == 2
    assert "arg:BASE_IMAGE=rocm/example:" not in output
    assert f"vllm.rocm.base_image_digest\t{digest}" in output
    assert f'BASE_IMAGE = "rocm/example:base@{digest}"' in output


def test_digest_lookup_retries_and_rejects_nonzero_output(tmp_path: Path) -> None:
    digest = "sha256:" + "c" * 64
    fake_bin, calls = write_fake_docker(
        tmp_path,
        f"""
printf 'call\\n' >> "$FAKE_DOCKER_CALLS"
if [[ "$FAKE_DOCKER_MODE" == retry ]] \
    && [[ "$(wc -l < "$FAKE_DOCKER_CALLS")" == 1 ]]; then
    exit 42
fi
printf 'Digest: {digest}\\n'
[[ "$FAKE_DOCKER_MODE" != nonzero ]]
""",
    )
    env = {
        "FAKE_DOCKER_CALLS": str(calls),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "ROCM_IMAGE_DIGEST_ATTEMPTS": "2",
        "ROCM_IMAGE_DIGEST_RETRY_DELAY": "0",
    }
    retried = run_sourced(
        CI_BAKE,
        'resolve_image_digest "rocm/example:base"',
        env=env | {"FAKE_DOCKER_MODE": "retry"},
    )
    assert retried.stdout.strip() == digest
    assert len(calls.read_text().splitlines()) == 2

    calls.write_text("")
    rejected = run_sourced(
        CI_BAKE,
        'resolve_image_digest "rocm/example:base"',
        env=env
        | {
            "FAKE_DOCKER_MODE": "nonzero",
            "ROCM_IMAGE_DIGEST_ATTEMPTS": "1",
        },
        check=False,
    )
    assert rejected.returncode == 1
    assert "status 1" in rejected.stderr


def test_stable_tag_requires_authorized_build_and_content_hash() -> None:
    command = (
        'TARGET="ci-base-rocm-ci"\n'
        'CI_BASE_IMAGE_TAG="rocm/example:ci_base"\n'
        "configure_ci_base_image_refs >/dev/null\n"
        'printf "primary=%s\\nstable=%s\\n" '
        '"$CI_BASE_IMAGE_TAG" "$CI_BASE_IMAGE_TAG_STABLE"\n'
        'ci_base_candidate_refs | sed "s/^/candidate=/"\n'
        'ci_base_output_refs | sed "s/^/output=/"'
    )
    pr = run_sourced(
        CI_BAKE,
        'CI_BASE_CONTENT_HASH="content"\n' + command,
        env={
            "BUILDKITE_PULL_REQUEST": "48646",
            "CI_BASE_PUSH_STABLE_TAG": "1",
        },
    ).stdout.splitlines()
    main = run_sourced(
        CI_BAKE,
        'CI_BASE_CONTENT_HASH="content"\n' + command,
        env={
            "BUILDKITE_PULL_REQUEST": "false",
            "BUILDKITE_BRANCH": "main",
            "NIGHTLY": "1",
        },
    ).stdout.splitlines()
    missing = run_sourced(CI_BAKE, command, check=False)

    assert "primary=rocm/example:ci_base-content" in pr
    assert "stable=" in pr
    assert "candidate=rocm/example:ci_base" in pr
    assert "output=rocm/example:ci_base" not in pr
    assert "stable=rocm/example:ci_base" in main
    assert "output=rocm/example:ci_base" in main
    assert missing.returncode == 1
    assert "ci_base builds require a content hash" in missing.stderr


def test_amd_pipeline_preserves_tty_and_checkout_mode() -> None:
    steps = {
        step["key"]: step for step in yaml.safe_load(AMD_PIPELINE.read_text())["steps"]
    }
    ensure_env = steps["ensure-ci-base-amd"]["env"]

    assert ensure_env["REMOTE_VLLM"] == "0"
    assert "VLLM_BRANCH" not in ensure_env
    for key in ("refresh-rocm-base-amd", "ensure-ci-base-amd", "image-build-amd"):
        assert steps[key]["env"]["BUILDKIT_PROGRESS"] == "tty"

    remote = run_sourced(
        CI_BAKE,
        'TARGET="ci-base-rocm-ci"\ncompute_ci_base_hash_if_needed',
        env={"CI_BASE_CONTENT_FILES": "unused", "REMOTE_VLLM": "1"},
        check=False,
    )
    assert remote.returncode == 1
    assert "require REMOTE_VLLM=0" in remote.stderr


def test_handoff_is_digest_pinned_for_producer_and_consumer() -> None:
    digest = "sha256:" + "d" * 64
    content_ref = "rocm/example:ci_base-content"
    immutable_ref = f"{content_ref}@{digest}"
    producer = run_sourced(
        CI_BAKE,
        'TARGET="ci-base-rocm-ci"\n'
        'CI_BASE_CONTENT_HASH="content"\n'
        f'CI_BASE_IMAGE_TAG_CONTENT_REF="{content_ref}"\n'
        f'resolve_image_digest() {{ printf "%s\\n" "{digest}"; }}\n'
        'confirm_remote_image_push() { printf "validated=%s\\n" "$1"; }\n'
        'buildkite-agent() { printf "%s=%s\\n" "$3" "$4"; }\n'
        "publish_ci_base_handoff_ref",
    )
    assert f"validated={immutable_ref}" in producer.stdout
    assert f"rocm-ci-base-image={immutable_ref}" in producer.stdout

    consumer = run_bash(
        "buildkite-agent() {\n"
        f'  printf "%s\\n" "{immutable_ref}"\n'
        "}\n"
        'bash() { printf "selected=%s\\n" "$CI_BASE_IMAGE"; }\n'
        'source "$1"',
        BUILD_TEST_IMAGE,
        env={"BUILDKITE": "true"},
    )
    assert f"selected={immutable_ref}" in consumer.stdout


@pytest.mark.parametrize("handoff", ("", "rocm/example:ci_base-content"))
def test_consumer_rejects_missing_or_mutable_handoff(handoff: str) -> None:
    result = run_bash(
        "buildkite-agent() {\n"
        f'  printf "%s\\n" "{handoff}"\n'
        "}\n"
        'bash() { echo "unexpected build"; }\n'
        'source "$1"',
        BUILD_TEST_IMAGE,
        env={"BUILDKITE": "true"},
        check=False,
    )

    assert result.returncode == 1
    assert "unexpected build" not in result.stdout
    assert (
        "Required ROCm ci_base handoff metadata is missing or invalid" in result.stderr
    )
