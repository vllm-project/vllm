# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
import regex as re
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_BAKE = REPO_ROOT / ".buildkite/scripts/ci-bake-rocm.sh"
ROCM_BASE_REFRESH = REPO_ROOT / ".buildkite/scripts/rocm/refresh-base-image.sh"
PROMOTE_STABLE = REPO_ROOT / ".buildkite/scripts/rocm/promote-stable-images.sh"
BUILD_CI_BASE = REPO_ROOT / ".buildkite/scripts/rocm/build-ci-base.sh"
BUILD_TEST_IMAGE = REPO_ROOT / ".buildkite/scripts/rocm/build-test-image.sh"
SMOKE_TEST_IMAGE = REPO_ROOT / ".buildkite/scripts/rocm/smoke-test-image.sh"
AMD_PIPELINE = REPO_ROOT / ".buildkite/hardware_tests/amd.yaml"
DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64
DIGEST_C = "sha256:" + "c" * 64
DIGEST_D = "sha256:" + "d" * 64
DIGEST_E = "sha256:" + "e" * 64
ISOLATED_ENV_VARS = frozenset(
    "BASE_IMAGE FORCE_BUILD IMAGE_TAG NIGHTLY REMOTE_VLLM TARGET "  # noqa: SIM905
    "USE_SCCACHE VLLM_BAKE_FILE VLLM_BRANCH VLLM_REPO".split()
)


def dockerfile_stage_bodies(dockerfile: str) -> dict[str, str]:
    matches = list(
        re.finditer(
            r"^FROM(?:\s+--\S+)*\s+\S+\s+AS\s+(\S+)\s*$",
            dockerfile,
            re.IGNORECASE | re.MULTILINE,
        )
    )
    return {
        match.group(1): dockerfile[
            match.end() : matches[index + 1].start()
            if index + 1 < len(matches)
            else len(dockerfile)
        ]
        for index, match in enumerate(matches)
    }


def run_bash(
    command: str,
    *args: Path,
    env: Mapping[str, str] | None = None,
    cwd: Path = REPO_ROOT,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    clean_env = os.environ.copy()
    for key in tuple(clean_env):
        if key.startswith(("BUILDKITE", "CI_BASE_", "ROCM_")) or (
            key in ISOLATED_ENV_VARS
        ):
            clean_env.pop(key)
    clean_env.update(env or {})

    argv = [
        "bash",
        "-c",
        "buildkite-agent() { return 127; }\n" + command,
        "rocm-ci-cache-test",
        *(str(arg) for arg in args),
    ]
    result = subprocess.run(
        argv,
        check=False,
        cwd=cwd,
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
    cwd: Path = REPO_ROOT,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return run_bash(
        'source "$1"\nshift\n' + command,
        script,
        *args,
        env=env,
        cwd=cwd,
        check=check,
    )


def test_ci_base_contract() -> None:
    files, dockerfile, stages = run_sourced(
        CI_BAKE,
        'printf "%s\\n%s\\n%s\\n" "$DEFAULT_CI_BASE_CONTENT_FILES" '
        '"$DEFAULT_CI_BASE_DOCKERFILE" "$DEFAULT_CI_BASE_DOCKERFILE_STAGES"',
    ).stdout.splitlines()

    configured_files = set(files.split())
    required_files = set(
        """
        .dockerignore .buildkite/scripts/ci-bake-rocm.sh
        .buildkite/scripts/rocm/build-ci-base.sh docker/ci-rocm.hcl
        docker/docker-bake-rocm.hcl requirements/common.txt requirements/rocm.txt
        requirements/test/rocm.txt rust-toolchain.toml tests/vllm_test_utils
        tools/install_protoc.sh tools/install_torchcodec_rocm.sh
        """.split()  # noqa: SIM905
    )
    assert required_files <= configured_files
    assert all((REPO_ROOT / path).exists() for path in configured_files)
    assert dockerfile == "docker/Dockerfile.rocm"

    dockerfile_text = (REPO_ROOT / dockerfile).read_text()
    declared_stages = set(
        re.findall(
            r"^FROM(?:\s+--\S+)*\s+\S+\s+AS\s+(\S+)",
            dockerfile_text,
            re.IGNORECASE | re.MULTILINE,
        )
    )
    configured_stages = set(stages.split())
    required_stages = set(
        "base rust_toolchain_input_0 rust-toolchain-input rust-toolchain "  # noqa: SIM905
        "build_nixl build_rocshmem build_deepep mori_base ci_base".split()
    )
    assert required_stages <= configured_stages <= declared_stages

    ci = (REPO_ROOT / "docker/ci-rocm.hcl").read_text()
    target = ci.split('target "ci-base-rocm-ci" {', 1)[1].split("\n}", 1)[0]
    cache_from = target.split("cache-from =", 1)[1].split("cache-to =", 1)[0]
    tags = target.split("tags     = compact([", 1)[1].split("])", 1)[0]
    assert "CI_BASE_STABLE_CACHE_REF" in cache_from
    assert "CI_BASE_IMAGE_TAG_STABLE" not in cache_from
    assert "CI_BASE_IMAGE_TAG_STABLE" in tags
    assert "CI_BASE_IMAGE_TAG_BUILD_EXTRA" in tags
    assert "CI_BASE_STABLE_CACHE_REF" not in tags

    missing_hash = run_sourced(
        CI_BAKE,
        'TARGET="ci-base-rocm-ci"\nconfigure_ci_base_image_refs',
        check=False,
    )
    assert missing_hash.returncode != 0
    assert "ci_base builds require a content hash" in missing_hash.stderr


def test_rocm_base_identity_tracks_recipe_args_and_parent(tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile.rocm_base"
    dockerfile.write_text(
        "ARG BASE_IMAGE=rocm/example:latest\n"
        "ARG PYTHON_VERSION=3.12\n"
        "ARG SCCACHE_DOWNLOAD_URL=https://example.com/sccache-a.tar.gz\n"
        "FROM ${BASE_IMAGE} AS base\n"
    )
    common_env = {
        "ROCM_BASE_CONTENT_ARGS": ("BASE_IMAGE PYTHON_VERSION SCCACHE_DOWNLOAD_URL"),
        "ROCM_BASE_DOCKERFILE": str(dockerfile),
    }

    def input_hash(
        digest: str,
        *,
        alias: str = "rocm/example:alias-a",
        python_version: str | None = None,
        sccache_download_url: str = "https://example.com/sccache-a.tar.gz",
    ) -> str:
        env = common_env | {
            "BASE_IMAGE": alias,
            "PARENT_DIGEST": digest,
            "SCCACHE_DOWNLOAD_URL": sccache_download_url,
        }
        if python_version is not None:
            env["PYTHON_VERSION"] = python_version
        return run_sourced(
            ROCM_BASE_REFRESH,
            "ROCM_BASE_USE_SCCACHE_EFFECTIVE=0\n"
            'ROCM_BASE_ARG_VALUES[BASE_IMAGE]="$BASE_IMAGE"\n'
            'ROCM_BASE_ARG_VALUES[PYTHON_VERSION]="${PYTHON_VERSION:-3.12}"\n'
            'ROCM_BASE_ARG_VALUES[SCCACHE_DOWNLOAD_URL]="$SCCACHE_DOWNLOAD_URL"\n'
            "ROCM_BASE_PARENT_PINNED=$(canonical_pinned_image_ref "
            '"$BASE_IMAGE" "$PARENT_DIGEST")\n'
            "compute_base_input_hash",
            env=env,
        ).stdout.strip()

    initial = input_hash(DIGEST_A)
    assert input_hash(DIGEST_A, alias="rocm/example:alias-b") == initial
    assert input_hash(DIGEST_A, alias="rocm/other:alias-a") != initial
    assert input_hash(DIGEST_B) != initial
    assert input_hash(DIGEST_A, python_version="3.13") != initial
    assert (
        input_hash(
            DIGEST_A,
            sccache_download_url="https://example.com/sccache-b.tar.gz",
        )
        != initial
    )

    dockerfile.write_text(dockerfile.read_text() + "RUN echo changed\n")
    assert input_hash(DIGEST_A) != initial


def test_rocm_base_scope_and_lookup_trust_boundaries() -> None:
    pr_scope = run_sourced(
        ROCM_BASE_REFRESH,
        "rocm_base_scope",
        env={
            "BUILDKITE_BRANCH": "feature",
            "BUILDKITE_PULL_REQUEST": "48646",
            "BUILDKITE_REPO": "https://github.com/vllm-project/vllm.git",
        },
    ).stdout.strip()
    trusted_scope = run_sourced(
        ROCM_BASE_REFRESH,
        "rocm_base_scope",
        env={
            "BUILDKITE": "true",
            "BUILDKITE_BRANCH": "main",
            "BUILDKITE_PULL_REQUEST": "false",
            "BUILDKITE_REPO": "https://github.com/vllm-project/vllm.git",
        },
    ).stdout.strip()
    assert re.fullmatch(r"pr-48646-[0-9a-f]{12}", pr_scope)
    assert trusted_scope == ""

    result = run_sourced(
        ROCM_BASE_REFRESH,
        """
ROCM_BASE_CANONICAL_TAG=rocm/example:base-pr-48646-input-hash
ROCM_BASE_TRUSTED_TAG=rocm/example:base-input-hash
ROCM_BASE_INPUT_HASH=hash
remote_image_exists() { [[ "$1" == "$ROCM_BASE_TRUSTED_TAG" ]]; }
resolve_image_digest() { printf '%s\n' "$DIGEST_A"; }
remote_rocm_base_matches() { [[ "$1" == "rocm/example@$DIGEST_A" ]]; }
select_cached_base_image
printf '%s\n%s\n' "$SELECTED_BASE_REF" "$ROCM_BASE_CACHE_SOURCE"
""",
        env={"DIGEST_A": DIGEST_A},
    )
    assert result.stdout.splitlines() == [
        f"rocm/example@{DIGEST_A}",
        "trusted-main",
    ]

    trusted_result = run_sourced(
        ROCM_BASE_REFRESH,
        """
ROCM_BASE_CANONICAL_TAG=rocm/example:base-input-hash
ROCM_BASE_TRUSTED_TAG="$ROCM_BASE_CANONICAL_TAG"
ROCM_BASE_INPUT_HASH=hash
remote_image_exists() { return 0; }
resolve_image_digest() { printf '%s\n' "$DIGEST_A"; }
remote_rocm_base_matches() { return 0; }
select_cached_base_image
printf '%s\n%s\n' "$SELECTED_BASE_REF" "$ROCM_BASE_CACHE_SOURCE"
""",
        env={"DIGEST_A": DIGEST_A},
    )
    assert trusted_result.stdout.splitlines() == [
        f"rocm/example@{DIGEST_A}",
        "scope",
    ]

    lookup_failure = run_sourced(
        ROCM_BASE_REFRESH,
        """
ROCM_BASE_CANONICAL_TAG=rocm/example:base-input-hash
ROCM_BASE_TRUSTED_TAG="$ROCM_BASE_CANONICAL_TAG"
ROCM_BASE_INPUT_HASH=hash
remote_image_exists() { return 2; }
select_cached_base_image
""",
        check=False,
    )
    assert lookup_failure.returncode == 3


@pytest.mark.parametrize(
    ("mode", "expected_status", "expected_calls"),
    [
        ("missing", 1, 1),
        ("transient", 2, 2),
        ("credential", 2, 2),
        ("retry", 0, 2),
    ],
)
def test_rocm_base_registry_lookup_distinguishes_miss_from_failure(
    tmp_path: Path, mode: str, expected_status: int, expected_calls: int
) -> None:
    trace = tmp_path / "lookups"
    result = run_sourced(
        ROCM_BASE_REFRESH,
        """
docker() {
  printf 'lookup\n' >> "$TRACE"
  case "$MODE:$(wc -l < "$TRACE")" in
    missing:*) printf 'manifest unknown: not found\n'; return 1 ;;
    transient:*) printf 'connection reset\n'; return 42 ;;
    credential:*) printf 'credential helper executable not found in PATH\n'; return 1 ;;
    retry:1) printf 'connection reset\n'; return 42 ;;
    *) printf 'found\n' ;;
  esac
}
set +e
remote_image_exists rocm/example:base
status=$?
set -e
printf 'status=%s\n' "$status"
""",
        env={
            "MODE": mode,
            "ROCM_BASE_IMAGE_LOOKUP_ATTEMPTS": "2",
            "ROCM_BASE_IMAGE_LOOKUP_RETRY_DELAY": "0",
            "TRACE": str(trace),
        },
    )
    assert result.stdout.strip() == f"status={expected_status}"
    assert len(trace.read_text().splitlines()) == expected_calls


@pytest.mark.parametrize(
    ("cache_hit", "expected_events"),
    [
        ("1", ["publish:1:0"]),
        ("0", ["setup", "build", "publish:0:1"]),
    ],
)
def test_rocm_base_exact_hit_skips_build_and_miss_self_heals(
    cache_hit: str, expected_events: list[str]
) -> None:
    result = run_sourced(
        ROCM_BASE_REFRESH,
        """
prepare_base_inputs() { :; }
select_cached_base_image() {
  if [[ "$CACHE_HIT" == "1" ]]; then
    SELECTED_BASE_REF=rocm/example:base-input-hash
    ROCM_BASE_CACHE_SOURCE=scope
    return 0
  fi
  return 1
}
setup_builder() { printf 'setup\n'; }
build_base_image() {
  printf 'build\n'
  SELECTED_BASE_REF=rocm/example:base-input-hash
  ROCM_BASE_CACHE_SOURCE=built
}
publish_base_handoff() { printf 'publish:%s:%s\n' "$1" "$2"; }
main
""",
        env={"CACHE_HIT": cache_hit},
    )
    assert result.stdout.splitlines() == expected_events


def test_rocm_base_main_maps_lookup_error_to_retryable_failure() -> None:
    result = run_sourced(
        ROCM_BASE_REFRESH,
        """
prepare_base_inputs() { :; }
select_cached_base_image() { return 3; }
setup_builder() { printf 'unexpected build\n'; }
main
""",
        check=False,
    )
    assert result.returncode == 1
    assert "unexpected build" not in result.stdout
    assert "selection failed" in result.stderr


def test_rocm_base_flow_does_not_keep_the_git_diff_gate() -> None:
    refresh = ROCM_BASE_REFRESH.read_text()
    test_image = BUILD_TEST_IMAGE.read_text()

    assert "base_recipe_changed" not in refresh
    assert "rocm-base-refresh" not in refresh
    assert "ROCM_BASE_REFRESH_FORCE" not in refresh
    assert 'metadata_set "rocm-base-image-descriptive"' not in refresh
    assert "rocm-base-refresh" not in test_image


def test_content_addressed_image_labels_exclude_per_build_identity() -> None:
    refresh = ROCM_BASE_REFRESH.read_text()
    ci_bake = CI_BAKE.read_text()
    ci_metadata = ci_bake.split("ci_base_metadata_pairs() {", 1)[1].split("\n}", 1)[0]

    assert "org.opencontainers.image.revision" not in refresh
    assert "vllm.rocm_base.git_commit" not in refresh
    for volatile_label in (
        "vllm.ci_base.image.commit",
        "vllm.ci_base.git_commit",
        "vllm.ci_base.git_branch",
        "vllm.ci_base.vllm_branch",
        "vllm.buildkite.build_number",
        "vllm.buildkite.build_id",
    ):
        assert volatile_label not in ci_metadata


def test_ci_base_bake_targets_exclude_volatile_annotations() -> None:
    base_bake = (REPO_ROOT / "docker/docker-bake-rocm.hcl").read_text()
    ci_bake = (REPO_ROOT / "docker/ci-rocm.hcl").read_text()

    base_ci_target = base_bake.split('target "ci-base-rocm" {', 1)[1].split("\n}", 1)[0]
    ci_target = ci_bake.split('target "ci-base-rocm-ci" {', 1)[1].split("\n}", 1)[0]
    common_ci_target = ci_bake.split('target "_ci-rocm" {', 1)[1].split("\n}", 1)[0]
    test_target = ci_bake.split('target "test-rocm-ci" {', 1)[1].split("\n}", 1)[0]

    assert '"_labels-common"' in base_ci_target
    assert '"_labels"' not in base_ci_target
    assert '"_labels-common"' in ci_target
    assert '"_labels"' not in ci_target
    assert "annotations" not in common_ci_target
    assert "vllm.buildkite.build_number" in test_target
    assert "vllm.buildkite.build_id" in test_target


@pytest.mark.parametrize("no_cache", ["0", "1"])
def test_rocm_base_build_uses_scoped_registry_cache(
    tmp_path: Path, no_cache: str
) -> None:
    trace = tmp_path / "docker-calls"
    run_sourced(
        ROCM_BASE_REFRESH,
        """
ROCM_BASE_CONTENT_ARGS="BASE_IMAGE PYTHON_VERSION"
ROCM_BASE_ARG_VALUES[PYTHON_VERSION]=3.12
ROCM_BASE_PARENT_PINNED="rocm/parent@$DIGEST_A"
ROCM_BASE_PARENT_DIGEST="$DIGEST_A"
ROCM_BASE_INPUT_HASH="input-hash"
ROCM_BASE_CANONICAL_TAG="rocm/example:base-pr-48646-input-input-hash"
ROCM_BASE_STABLE_TAG="rocm/example:base"
ROCM_BASE_LAYER_CACHE_REF="rocm/cache:rocm-base-pr-48646"
ROCM_BASE_TRUSTED_LAYER_CACHE_REF="rocm/cache:rocm-base-main"
docker() { printf '%s\n' "$*" >> "$TRACE"; }
remote_image_exists() { return 0; }
remote_rocm_base_matches() { return 0; }
build_base_image
""",
        env={
            "DIGEST_A": DIGEST_A,
            "ROCM_BASE_NO_CACHE": no_cache,
            "TRACE": str(trace),
        },
    )
    build = next(
        line for line in trace.read_text().splitlines() if "buildx build" in line
    )
    if no_cache == "1":
        assert "--no-cache" in build
        assert "type=registry" not in build
    else:
        assert "--no-cache" not in build
        assert "type=registry,ref=rocm/cache:rocm-base-pr-48646" in build
        assert "type=registry,ref=rocm/cache:rocm-base-main" in build
        assert (
            "type=registry,ref=rocm/cache:rocm-base-pr-48646,"
            "mode=max,ignore-error=true" in build
        )
    assert f"BASE_IMAGE=rocm/parent@{DIGEST_A}" in build
    assert "--tag rocm/example:base --" not in build


def test_rocm_base_stage_boundaries_are_independent() -> None:
    dockerfile = (REPO_ROOT / "docker/Dockerfile.rocm_base").read_text()
    stages = dockerfile_stage_bodies(dockerfile)

    torch = stages["build_pytorch"]
    torch_runtime = stages["build_pytorch_runtime"]
    assert "PYTORCH_VISION_" not in torch
    assert "PYTORCH_AUDIO_" not in torch
    assert "from=build_pytorch" in torch_runtime
    assert "pip install /install/*.whl" in torch_runtime

    torchvision = stages["build_torchvision"]
    torchaudio = stages["build_torchaudio"]
    assert "FROM build_pytorch_runtime AS build_torchvision" in dockerfile
    assert "FROM build_pytorch_runtime AS build_torchaudio" in dockerfile
    assert "pip install /install/*.whl" not in torchvision
    assert "pip install /install/*.whl" not in torchaudio

    for aggregate in ("debs_wheel_release", "debs"):
        body = stages[aggregate]
        for component in ("build_pytorch", "build_torchvision", "build_torchaudio"):
            assert f"from={component}" in body


def test_content_hash_tracks_git_content(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    source = repo / "input.py"
    source.write_text("VALUE = 1\n")
    alternate = repo / "alternate.py"
    alternate.write_text("VALUE = 2\n")
    link = repo / "link.py"
    link.symlink_to("input.py")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)

    def content_hash() -> str:
        return run_sourced(CI_BAKE, 'compute_content_hash "."', cwd=repo).stdout.strip()

    initial = content_hash()
    (repo / "untracked.py").write_text("ignored = True\n")
    assert content_hash() == initial
    source.write_text("VALUE = 2\n")
    assert content_hash() != initial
    linked = content_hash()
    link.unlink()
    link.symlink_to("alternate.py")
    assert content_hash() != linked

    non_git = tmp_path / "non-git"
    non_git.mkdir()
    failed = run_sourced(CI_BAKE, 'compute_content_hash "."', cwd=non_git, check=False)
    assert failed.returncode != 0
    assert "failed to hash content" in failed.stderr


def test_effective_identity_and_base_pinning(tmp_path: Path) -> None:
    calls = tmp_path / "docker-calls"
    override = tmp_path / "override.hcl"
    docker_stub = """
docker() {
  printf '%s\n' "$*" >> "$DOCKER_CALLS"
  if [[ "$*" == *different* ]]; then
    printf 'Digest: %s\n' "$DIGEST_B"
  else
    printf 'Digest: %s\n' "$DIGEST_A"
  fi
}
"""
    result = run_sourced(
        CI_BAKE,
        docker_stub
        + """
CI_BASE_DOCKERFILE="$DEFAULT_CI_BASE_DOCKERFILE"
CI_BASE_CONTENT_FILES="$DEFAULT_CI_BASE_CONTENT_FILES"
CI_BASE_DOCKERFILE_STAGES="$DEFAULT_CI_BASE_DOCKERFILE_STAGES"
REMOTE_VLLM=0
BASE_IMAGE="rocm/example:alias-a"
VLLM_BRANCH="commit-a"
VLLM_REPO="https://example.com/fork-a.git"
compute_ci_base_content_hash_once
BASE_IMAGE="rocm/example:alias-b"
VLLM_BRANCH="commit-b"
VLLM_REPO="https://example.com/fork-b.git"
compute_ci_base_content_hash_once
BASE_IMAGE="rocm/example:different"
compute_ci_base_content_hash_once
: > "$DOCKER_CALLS"
TARGET="ci-base-rocm-ci"
VLLM_BAKE_FILE="docker/docker-bake-rocm.hcl"
ROCM_ARG_OVERRIDE_PATH="$1"
BASE_IMAGE="rocm/example:alias-a"
REMOTE_VLLM=0
compute_ci_base_hash_if_needed
write_rocm_build_arg_override
printf 'pinned=%s\n' "$BASE_IMAGE"
""",
        override,
        env={
            "CI_BASE_HASH_ATTEMPTS": "1",
            "CI_BASE_HASH_RETRY_DELAY": "0",
            "DIGEST_A": DIGEST_A,
            "DIGEST_B": DIGEST_B,
            "DOCKER_CALLS": str(calls),
        },
    )
    hashes = [
        line
        for line in result.stdout.splitlines()
        if re.fullmatch(r"[0-9a-f]{64}", line)
    ]
    assert len(hashes) == 3
    assert hashes[0] == hashes[1]
    assert hashes[2] != hashes[0]
    assert f"pinned=rocm/example:alias-a@{DIGEST_A}" in result.stdout
    assert f'BASE_IMAGE = "rocm/example:alias-a@{DIGEST_A}"' in override.read_text()
    assert calls.read_text().splitlines() == [
        "buildx imagetools inspect rocm/example:alias-a"
    ]


@pytest.mark.parametrize(
    ("mode", "attempts", "succeeds"),
    [("retry", "2", True), ("malformed", "1", False), ("nonzero", "1", False)],
)
def test_digest_lookup_retries_and_rejects_invalid_output(
    tmp_path: Path, mode: str, attempts: str, succeeds: bool
) -> None:
    calls = tmp_path / "docker-calls"
    docker_stub = """
docker() {
  printf 'call\n' >> "$DOCKER_CALLS"
  case "$MODE:$(wc -l < "$DOCKER_CALLS")" in
    retry:1) return 42 ;;
    malformed:*) printf 'Digest: sha256:bad\n'; return 0 ;;
    nonzero:*) printf 'Digest: %s\n' "$DIGEST_A"; return 42 ;;
    *) printf 'Digest: %s\n' "$DIGEST_A" ;;
  esac
}
resolve_image_digest "rocm/example:base"
"""
    result = run_sourced(
        CI_BAKE,
        docker_stub,
        env={
            "DIGEST_A": DIGEST_A,
            "DOCKER_CALLS": str(calls),
            "MODE": mode,
            "ROCM_IMAGE_DIGEST_ATTEMPTS": attempts,
            "ROCM_IMAGE_DIGEST_RETRY_DELAY": "0",
        },
        check=False,
    )
    assert (result.returncode == 0) is succeeds
    assert len(calls.read_text().splitlines()) == (2 if succeeds else 1)
    if succeeds:
        assert result.stdout.strip() == DIGEST_A
    else:
        assert "Failed to resolve digest" in result.stderr


@pytest.mark.parametrize(
    ("mode", "expected_status", "expected_calls"),
    [
        ("missing", 1, 1),
        ("transient", 2, 2),
        ("credential", 2, 2),
        ("retry", 0, 2),
    ],
)
def test_ci_base_registry_lookup_distinguishes_miss_from_failure(
    tmp_path: Path, mode: str, expected_status: int, expected_calls: int
) -> None:
    trace = tmp_path / "lookups"
    result = run_sourced(
        CI_BAKE,
        """
docker() {
  printf 'lookup\n' >> "$TRACE"
  case "$MODE:$(wc -l < "$TRACE")" in
    missing:*) printf 'manifest unknown: not found\n'; return 1 ;;
    transient:*) printf 'connection reset\n'; return 42 ;;
    credential:*) printf 'credential helper executable not found in PATH\n'; return 1 ;;
    retry:1) printf 'connection reset\n'; return 42 ;;
    *) printf 'found\n' ;;
  esac
}
set +e
remote_image_exists rocm/example:ci_base
status=$?
set -e
printf 'status=%s\n' "$status"
""",
        env={
            "MODE": mode,
            "ROCM_IMAGE_LOOKUP_ATTEMPTS": "2",
            "ROCM_IMAGE_LOOKUP_RETRY_DELAY": "0",
            "TRACE": str(trace),
        },
    )
    assert result.stdout.strip() == f"status={expected_status}"
    assert len(trace.read_text().splitlines()) == expected_calls


def test_ci_base_lookup_failure_does_not_start_a_build() -> None:
    result = run_sourced(
        CI_BAKE,
        """
TARGET="ci-base-rocm-ci"
IMAGE_TAG="rocm/example:ci_base-content"
CI_BASE_CONTENT_HASH="content"
remote_image_exists() { return 2; }
maybe_skip_existing_image
""",
        check=False,
    )
    assert result.returncode == 2
    assert "Could not determine whether" in result.stderr


def test_ci_base_main_maps_exhausted_lookup_to_retryable_failure() -> None:
    result = run_sourced(
        CI_BAKE,
        """
init_config() { TARGET="ci-base-rocm-ci"; }
print_header() { :; }
validate_inputs() { :; }
load_ci_hcl() { :; }
init_bake_files() { :; }
compute_ci_base_hash_if_needed() { :; }
configure_ci_base_image_refs() { :; }
maybe_skip_existing_image() { return 2; }
setup_builder() { printf 'unexpected build\n'; }
main
""",
        check=False,
    )
    assert result.returncode == 1
    assert "unexpected build" not in result.stdout


def test_required_ci_base_metadata_fails_closed_in_buildkite() -> None:
    result = run_sourced(
        CI_BAKE,
        'set_buildkite_metadata "rocm-ci-base-build-required" "0"',
        env={"BUILDKITE": "true"},
        check=False,
    )
    assert result.returncode != 0
    assert "required Buildkite metadata" in result.stderr


@pytest.mark.parametrize(
    "env",
    [
        {"BUILDKITE_PULL_REQUEST": "48646", "CI_BASE_PUSH_STABLE_TAG": "1"},
        {
            "BUILDKITE_PULL_REQUEST": "false",
            "BUILDKITE_BRANCH": "main",
            "NIGHTLY": "1",
        },
    ],
)
def test_long_ci_base_build_never_writes_stable(env: dict[str, str]) -> None:
    result = run_sourced(
        CI_BAKE,
        'TARGET="ci-base-rocm-ci"\n'
        'CI_BASE_CONTENT_HASH="content"\n'
        'CI_BASE_IMAGE_TAG="rocm/example:ci_base"\n'
        "configure_ci_base_image_refs >/dev/null\n"
        "ci_base_output_refs",
        env=env | {"BUILDKITE_BUILD_ID": "uuid-123"},
    )
    outputs = set(result.stdout.splitlines())
    assert "rocm/example:ci_base-content" in outputs
    assert "rocm/example:ci_base-build-uuid-123" in outputs
    assert "rocm/example:ci_base" not in outputs


def test_ci_base_post_build_handoff_uses_build_scoped_source(tmp_path: Path) -> None:
    trace = tmp_path / "trace"
    result = run_sourced(
        CI_BAKE,
        """
TARGET="ci-base-rocm-ci"
CI_BASE_CONTENT_HASH="content"
CI_BASE_IMAGE_TAG="rocm/example:ci_base"
configure_ci_base_image_refs >/dev/null
resolve_image_digest() {
  printf 'resolve:%s\n' "$1" >> "$TRACE"
  printf '%s\n' "$DIGEST_A"
}
confirm_remote_image_push() { printf 'validated:%s\n' "$1" >> "$TRACE"; }
buildkite-agent() { printf 'metadata:%s=%s\n' "$3" "$4" >> "$TRACE"; }
publish_ci_base_handoff_ref
""",
        env={
            "BUILDKITE_BUILD_ID": "uuid-123",
            "DIGEST_A": DIGEST_A,
            "TRACE": str(trace),
        },
    )
    build_ref = "rocm/example:ci_base-build-uuid-123"
    handoff = f"rocm/example:ci_base-content@{DIGEST_A}"
    assert trace.read_text().splitlines() == [
        f"resolve:{build_ref}",
        f"validated:{handoff}",
        f"metadata:rocm-ci-base-image={handoff}",
    ]
    assert f"Published immutable ci_base handoff: {handoff}" in result.stdout


def test_ci_base_failed_bake_confirms_build_scoped_source(tmp_path: Path) -> None:
    trace = tmp_path / "trace"
    result = run_sourced(
        CI_BAKE,
        """
TARGET="ci-base-rocm-ci"
IMAGE_TAG="rocm/example:ci_base-content"
CI_BASE_IMAGE_TAG_BUILD_REF="rocm/example:ci_base-build-uuid-123"
BAKE_FILES=(-f docker/docker-bake-rocm.hcl)
BAKE_TARGETS=(ci-base-rocm-ci)
docker() { return 42; }
confirm_remote_image_push() { printf '%s\n' "$1" >> "$TRACE"; }
annotate_cache_export_warning() { :; }
run_bake
""",
        env={"TRACE": str(trace)},
    )
    assert trace.read_text().strip() == "rocm/example:ci_base-build-uuid-123"
    assert "Treating this as a non-fatal registry cache export failure" in result.stdout


def test_pr_reuses_stable_match_without_writing_stable(tmp_path: Path) -> None:
    stable = "rocm/example:ci_base"
    content = f"{stable}-content"
    commit = f"{stable}-deadbeef"
    immutable_stable = f"{stable}@{DIGEST_A}"
    handoff = f"{content}@{DIGEST_A}"
    trace = tmp_path / "trace"
    command = """
TARGET="ci-base-rocm-ci"
CI_BASE_CONTENT_HASH="content"
CI_BASE_IMAGE_TAG="$STABLE"
configure_ci_base_image_refs >/dev/null
remote_image_exists() { [[ "$1" == "$STABLE" ]]; }
resolve_image_digest() {
  printf 'resolve:%s\n' "$1" >> "$TRACE"
  printf '%s\n' "$DIGEST_A"
}
get_remote_image_label() {
  [[ "$1" == "$STABLE@$DIGEST_A" ]] || return 0
  case "$2" in
    vllm.ci_base.content_hash) printf 'content\n' ;;
    vllm.ci_base.metadata_version) printf '%s\n' "$DEFAULT_CI_BASE_METADATA_VERSION" ;;
  esac
}
docker() {
  [[ "$1 $2 $3 $4" == "buildx imagetools create -t" ]] || return 99
  printf 'retag:%s:%s\n' "$5" "$6" >> "$TRACE"
  [[ "${RETAG_FAIL:-0}" != 1 ]]
}
confirm_remote_image_push() {
  printf 'validated:%s\n' "$1" >> "$TRACE"
  [[ "${HANDOFF_FAIL:-0}" != 1 ]]
}
buildkite-agent() { printf 'metadata:%s=%s\n' "$3" "$4" >> "$TRACE"; }
if maybe_reuse_matching_ci_base_ref; then
  :
else
  printf 'rebuild\n' >> "$TRACE"
fi
"""
    env = {
        "BUILDKITE_COMMIT": "deadbeef",
        "BUILDKITE_PULL_REQUEST": "48646",
        "DIGEST_A": DIGEST_A,
        "STABLE": stable,
        "TRACE": str(trace),
    }

    run_sourced(CI_BAKE, command, env=env)
    events = trace.read_text().splitlines()
    assert {event for event in events if event.startswith("retag:")} == {
        f"retag:{content}:{immutable_stable}",
        f"retag:{commit}:{immutable_stable}",
    }
    assert f"metadata:rocm-ci-base-image={handoff}" in events
    assert [event for event in events if event.startswith("resolve:")] == [
        f"resolve:{stable}",
        f"resolve:{immutable_stable}",
    ]
    assert "rebuild" not in events

    for failure in ("RETAG_FAIL", "HANDOFF_FAIL"):
        trace.write_text("")
        run_sourced(CI_BAKE, command, env=env | {failure: "1"})
        events = trace.read_text().splitlines()
        assert "rebuild" in events
        assert not any(event.startswith("metadata:") for event in events)


def test_amd_pipeline_uses_local_checkout_for_ci_base() -> None:
    steps = {
        step["key"]: step for step in yaml.safe_load(AMD_PIPELINE.read_text())["steps"]
    }
    ensure_env = steps["ensure-ci-base-amd"]["env"]
    assert ensure_env["REMOTE_VLLM"] == "0"
    assert "VLLM_BRANCH" not in ensure_env

    remote = run_sourced(
        CI_BAKE,
        'TARGET="ci-base-rocm-ci"\ncompute_ci_base_hash_if_needed',
        env={"CI_BASE_CONTENT_FILES": "unused", "REMOTE_VLLM": "1"},
        check=False,
    )
    assert remote.returncode == 1
    assert "require REMOTE_VLLM=0" in remote.stderr

    promotion = steps["promote-stable-rocm-images-amd"]
    assert promotion["depends_on"] == ["image-build-amd"]
    assert promotion["if_condition"] == (
        "build.branch == pipeline.default_branch && build.pull_request.id == null"
    )
    assert promotion["concurrency"] == 1
    assert promotion["concurrency_group"] == "vllm/rocm/stable-image-promotion"
    assert steps["image-build-amd"]["depends_on"] == ["ensure-ci-base-amd"]


def test_stable_promotion_is_a_fast_noop_outside_trusted_main(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "metadata-calls"
    result = run_sourced(
        PROMOTE_STABLE,
        """
buildkite-agent() { printf '%s\n' "$*" >> "$TRACE"; return 1; }
main
""",
        env={
            "BUILDKITE": "true",
            "BUILDKITE_BRANCH": "feature",
            "BUILDKITE_PULL_REQUEST": "48646",
            "BUILDKITE_REPO": "https://github.com/vllm-project/vllm.git",
            "TRACE": str(trace),
        },
    )
    assert "outside a trusted main-branch build" in result.stdout
    assert not trace.exists()


@pytest.mark.parametrize(
    ("fail_ci_promotion", "latest_commit", "stable_state", "expected"),
    [
        ("0", "deadbeef", "previous", "promotes"),
        ("0", "cafebabe", "previous", "stale-commit"),
        ("0", "deadbeef", "current", "already-current"),
        ("0", "deadbeef", "missing", "bootstraps"),
        ("0", "deadbeef", "error", "stable-lookup-error"),
        ("1", "deadbeef", "previous", "rolls-back"),
    ],
)
def test_stable_promotion_rechecks_latest_main(
    tmp_path: Path,
    fail_ci_promotion: str,
    latest_commit: str,
    stable_state: str,
    expected: str,
) -> None:
    trace = tmp_path / "docker-calls"
    failure_marker = tmp_path / "ci-promotion-failed"
    base_stable_state = tmp_path / "base-stable-digest"
    ci_stable_state = tmp_path / "ci-stable-digest"
    if stable_state == "current":
        base_stable_state.write_text(DIGEST_A)
        ci_stable_state.write_text(DIGEST_C)
    elif stable_state == "previous":
        base_stable_state.write_text(DIGEST_D)
        ci_stable_state.write_text(DIGEST_E)
    else:
        base_stable_state.write_text(stable_state)
        ci_stable_state.write_text(stable_state)
    input_hash = "1" * 64
    content_hash = "2" * 64
    base_candidate = f"rocm/vllm-dev@{DIGEST_A}"
    base_canonical = f"rocm/vllm-dev:base-input-{input_hash}"
    ci_candidate = f"rocm/vllm-dev:ci_base-{content_hash}@{DIGEST_C}"
    content_files = (
        ".buildkite/scripts/ci-bake-rocm.sh "
        ".buildkite/scripts/rocm/build-ci-base.sh docker/ci-rocm.hcl"
    )
    base_config = json.dumps(
        {
            "config": {
                "Labels": {
                    "vllm.rocm_base.metadata_version": "2",
                    "vllm.rocm_base.input_hash": input_hash,
                    "vllm.rocm_base.image.canonical": base_canonical,
                    "vllm.rocm_base.image.stable": "rocm/vllm-dev:base",
                    "vllm.rocm_base.dockerfile": "docker/Dockerfile.rocm_base",
                    "vllm.rocm_base.base_image": f"rocm/parent@{DIGEST_B}",
                    "vllm.rocm_base.base_image_digest": DIGEST_B,
                }
            }
        }
    )
    ci_config = json.dumps(
        {
            "config": {
                "Labels": {
                    "vllm.ci_base.metadata_version": "2",
                    "vllm.ci_base.content_hash": content_hash,
                    "vllm.ci_base.content_files": content_files,
                    "vllm.ci_base.dockerfile": "docker/Dockerfile.rocm",
                    "vllm.ci_base.image.content": (
                        f"rocm/vllm-dev:ci_base-{content_hash}"
                    ),
                    "vllm.rocm.base_image": base_candidate,
                    "vllm.rocm.base_image_digest": DIGEST_A,
                }
            }
        }
    )
    result = run_sourced(
        PROMOTE_STABLE,
        """
buildkite-agent() {
  [[ "$1 $2" == "meta-data get" ]] || return 1
  case "$3" in
    rocm-base-image) printf '%s\n' "$BASE_CANDIDATE" ;;
    rocm-base-input-hash) printf '%s\n' "$BASE_INPUT_HASH" ;;
    rocm-base-canonical-tag) printf '%s\n' "$BASE_CANONICAL" ;;
    rocm-base-stable-tag) printf '%s\n' 'rocm/vllm-dev:base' ;;
    rocm-ci-base-image) printf '%s\n' "$CI_CANDIDATE" ;;
    rocm-ci-image-smoke-required|rocm-ci-image-smoked) printf '1\n' ;;
    *) return 1 ;;
  esac
}
docker() {
  printf '%s\n' "$*" >> "$TRACE"
  if [[ "$1 $2 $3" == "buildx imagetools create" ]]; then
    if [[ "$FAIL_CI_PROMOTION" == "1" \
        && "$6" == "rocm/vllm-dev:ci_base" \
        && "$7" == "$CI_CANDIDATE" \
        && ! -e "$FAILURE_MARKER" ]]; then
      touch "$FAILURE_MARKER"
      return 42
    fi
    case "$6" in
      rocm/vllm-dev:base) printf '%s\n' "${7##*@}" > "$BASE_STABLE_STATE" ;;
      rocm/vllm-dev:ci_base) printf '%s\n' "${7##*@}" > "$CI_STABLE_STATE" ;;
      *) return 95 ;;
    esac
    return 0
  fi
  [[ "$1 $2 $3" == "buildx imagetools inspect" ]] || return 99
  if [[ "$*" == *".Image"* ]]; then
    case "$4" in
      "$BASE_CANDIDATE") printf '%s\n' "$BASE_CONFIG" ;;
      "$CI_CANDIDATE") printf '%s\n' "$CI_CONFIG" ;;
      *) return 98 ;;
    esac
    return 0
  fi
  case "$4" in
    "$BASE_CANDIDATE") digest="$DIGEST_A" ;;
    "$CI_CANDIDATE") digest="$DIGEST_C" ;;
    'rocm/vllm-dev:base') digest="$(< "$BASE_STABLE_STATE")" ;;
    'rocm/vllm-dev:ci_base') digest="$(< "$CI_STABLE_STATE")" ;;
    "$PARENT_REF") digest="$DIGEST_B" ;;
    *) return 97 ;;
  esac
  case "$digest" in
    missing) printf 'manifest unknown: not found\n'; return 1 ;;
    error) printf 'connection reset\n'; return 42 ;;
  esac
  printf '"%s"\n' "$digest"
}
git() {
  case "$1" in
    check-ref-format|fetch) return 0 ;;
    rev-parse) printf '%s\n' "$LATEST_COMMIT" ;;
    show) printf 'ARG BASE_IMAGE=%s\n' "$PARENT_REF" ;;
    *) return 96 ;;
  esac
}
main
""",
        env={
            "BASE_CANDIDATE": base_candidate,
            "BASE_CANONICAL": base_canonical,
            "BASE_CONFIG": base_config,
            "BASE_INPUT_HASH": input_hash,
            "BASE_STABLE_STATE": str(base_stable_state),
            "BUILDKITE": "true",
            "BUILDKITE_BRANCH": "main",
            "BUILDKITE_COMMIT": "deadbeef",
            "BUILDKITE_PULL_REQUEST": "false",
            "BUILDKITE_REPO": "https://github.com/vllm-project/vllm.git",
            "CI_CANDIDATE": ci_candidate,
            "CI_CONFIG": ci_config,
            "CI_STABLE_STATE": str(ci_stable_state),
            "DIGEST_A": DIGEST_A,
            "DIGEST_B": DIGEST_B,
            "DIGEST_C": DIGEST_C,
            "FAILURE_MARKER": str(failure_marker),
            "FAIL_CI_PROMOTION": fail_ci_promotion,
            "LATEST_COMMIT": latest_commit,
            "PARENT_REF": "rocm/parent:latest",
            "ROCM_PROMOTION_IMAGE_LOOKUP_ATTEMPTS": "1",
            "ROCM_PROMOTION_IMAGE_LOOKUP_RETRY_DELAY": "0",
            "TRACE": str(trace),
        },
        check=False,
    )
    create_calls = [
        call
        for call in trace.read_text().splitlines()
        if call.startswith("buildx imagetools create")
    ]
    if expected in {"promotes", "bootstraps"}:
        assert result.returncode == 0
        assert create_calls == [
            "buildx imagetools create --prefer-index=false "
            f"-t rocm/vllm-dev:base {base_candidate}",
            "buildx imagetools create --prefer-index=false "
            f"-t rocm/vllm-dev:ci_base {ci_candidate}",
        ]
        assert base_stable_state.read_text().strip() == DIGEST_A
        assert ci_stable_state.read_text().strip() == DIGEST_C
        if expected == "bootstraps":
            assert "promotion will bootstrap it" in result.stdout
    elif expected == "stale-commit":
        assert result.returncode == 0
        assert not create_calls
        assert "build commit is no longer latest main" in result.stdout
    elif expected == "stable-lookup-error":
        assert result.returncode != 0
        assert not create_calls
        assert "existing stable ROCm image aliases" in result.stderr
    elif expected == "already-current":
        assert result.returncode == 0
        assert not create_calls
        assert "already match the selected candidates" in result.stdout
    else:
        assert result.returncode != 0
        assert create_calls == [
            "buildx imagetools create --prefer-index=false "
            f"-t rocm/vllm-dev:base {base_candidate}",
            "buildx imagetools create --prefer-index=false "
            f"-t rocm/vllm-dev:ci_base {ci_candidate}",
            "buildx imagetools create --prefer-index=false "
            f"-t rocm/vllm-dev:base rocm/vllm-dev@{DIGEST_D}",
            "buildx imagetools create --prefer-index=false "
            f"-t rocm/vllm-dev:ci_base rocm/vllm-dev@{DIGEST_E}",
        ]
        assert "attempting rollback" in result.stderr
        assert base_stable_state.read_text().strip() == DIGEST_D
        assert ci_stable_state.read_text().strip() == DIGEST_E


@pytest.mark.parametrize(
    ("handoff", "valid"),
    [(f"rocm/example@{DIGEST_A}", True), ("", False), ("rocm/example:base", False)],
)
def test_ci_base_build_requires_immutable_base_handoff(
    handoff: str, valid: bool
) -> None:
    result = run_bash(
        """
buildkite-agent() {
  [[ "$*" == "meta-data get rocm-base-image" ]] || return 1
  printf '%s\n' "$HANDOFF"
}
bash() { printf 'base=%s\npush_stable=%s\n' "$BASE_IMAGE" "$CI_BASE_PUSH_STABLE_TAG"; }
source "$1"
main
""",
        BUILD_CI_BASE,
        env={"BUILDKITE": "true", "HANDOFF": handoff},
        check=False,
    )
    assert (result.returncode == 0) is valid
    if valid:
        assert f"base={handoff}" in result.stdout
        assert "push_stable=0" in result.stdout
    else:
        assert "base=" not in result.stdout
        assert "handoff metadata is missing or invalid" in result.stderr


@pytest.mark.parametrize(
    ("ci_handoff", "valid"),
    [
        (f"rocm/example:ci_base-content@{DIGEST_B}", True),
        ("", False),
        ("rocm/example:ci_base-content", False),
    ],
)
def test_build_step_requires_immutable_ci_base_handoff(
    ci_handoff: str, valid: bool
) -> None:
    result = run_bash(
        """
buildkite-agent() {
  case "$*" in
    "meta-data get rocm-ci-base-image") printf '%s\n' "$CI_HANDOFF" ;;
    "meta-data set rocm-ci-image-smoke-required "*|\
    "meta-data set rocm-ci-image-smoked "*) return 0 ;;
    *) return 1 ;;
  esac
}
bash() { printf 'ci_base=%s\n' "$CI_BASE_IMAGE"; }
source "$1"
main
""",
        BUILD_TEST_IMAGE,
        env={
            "BUILDKITE": "true",
            "CI_HANDOFF": ci_handoff,
        },
        check=False,
    )
    assert (result.returncode == 0) is valid
    if valid:
        assert f"ci_base={ci_handoff}" in result.stdout
    else:
        assert "ci_base=" not in result.stdout
        assert "handoff metadata is missing or invalid" in result.stderr


@pytest.mark.parametrize(
    (
        "base_build_required",
        "ci_base_build_required",
        "branch",
        "pull_request",
        "target",
    ),
    [
        ("0", "0", "feature", "48646", "test-rocm-ci-with-artifacts"),
        ("1", "0", "feature", "48646", "test-rocm-ci-with-wheel"),
        ("0", "1", "feature", "48646", "test-rocm-ci-with-wheel"),
        ("", "0", "feature", "48646", "test-rocm-ci-with-wheel"),
        ("0", "", "feature", "48646", "test-rocm-ci-with-wheel"),
        ("unexpected", "0", "feature", "48646", "test-rocm-ci-with-wheel"),
        ("0", "0", "main", "false", "test-rocm-ci-with-wheel"),
    ],
)
def test_build_step_preserves_artifact_only_path(
    base_build_required: str,
    ci_base_build_required: str,
    branch: str,
    pull_request: str,
    target: str,
) -> None:
    result = run_bash(
        """
buildkite-agent() {
  case "$*" in
    "meta-data get rocm-ci-base-image") printf 'rocm/example@%s\n' "$DIGEST_B" ;;
    "meta-data get rocm-base-build-required") printf '%s\n' "$BASE_BUILD_REQUIRED" ;;
    "meta-data get rocm-ci-base-build-required")
      printf '%s\n' "$CI_BASE_BUILD_REQUIRED"
      ;;
    "meta-data set rocm-ci-image-smoke-required "*|\
    "meta-data set rocm-ci-image-smoked "*) return 0 ;;
    *) return 1 ;;
  esac
}
bash() { printf 'target=%s\n' "$2"; }
source "$1"
main
""",
        BUILD_TEST_IMAGE,
        env={
            "BASE_BUILD_REQUIRED": base_build_required,
            "BUILDKITE": "true",
            "BUILDKITE_BRANCH": branch,
            "BUILDKITE_PULL_REQUEST": pull_request,
            "BUILDKITE_REPO": "https://github.com/vllm-project/vllm.git",
            "CI_BASE_BUILD_REQUIRED": ci_base_build_required,
            "DIGEST_B": DIGEST_B,
            "ROCM_CI_ARTIFACT_ONLY": "1",
        },
    )
    assert f"target={target}" in result.stdout


@pytest.mark.parametrize(("smoke_required", "docker_calls"), [("0", 0), ("1", 1)])
def test_image_smoke_never_uses_a_stale_artifact_only_tag(
    tmp_path: Path, smoke_required: str, docker_calls: int
) -> None:
    trace = tmp_path / "trace"
    result = run_bash(
        """
buildkite-agent() {
  case "$*" in
    "meta-data get rocm-ci-image-smoke-required") printf '%s\n' "$SMOKE_REQUIRED" ;;
    "meta-data set rocm-ci-image-smoked 1") printf 'smoked\n' >> "$TRACE" ;;
    *) return 1 ;;
  esac
}
docker() { printf 'docker\n' >> "$TRACE"; }
source "$1"
main
""",
        SMOKE_TEST_IMAGE,
        env={
            "BUILDKITE": "true",
            "BUILDKITE_COMMIT": "deadbeef",
            "SMOKE_REQUIRED": smoke_required,
            "TRACE": str(trace),
        },
    )
    events = trace.read_text().splitlines() if trace.exists() else []
    assert events.count("docker") == docker_calls
    assert ("smoked" in events) is (smoke_required == "1")
    if smoke_required == "0":
        assert "no commit image to smoke-test" in result.stdout
