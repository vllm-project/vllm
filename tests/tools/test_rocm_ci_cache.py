# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
import regex as re
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_BAKE = REPO_ROOT / ".buildkite/scripts/ci-bake-rocm.sh"
BUILD_CI_BASE = REPO_ROOT / ".buildkite/scripts/rocm/build-ci-base.sh"
BUILD_TEST_IMAGE = REPO_ROOT / ".buildkite/scripts/rocm/build-test-image.sh"
ROCM_BASE_REFRESH = REPO_ROOT / ".buildkite/scripts/rocm/refresh-base-image.sh"
ROCM_BASE_DOCKERFILE = REPO_ROOT / "docker/Dockerfile.rocm_base"
ROCM_DOCKERFILE = REPO_ROOT / "docker/Dockerfile.rocm"
AMD_PIPELINE = REPO_ROOT / ".buildkite/hardware_tests/amd.yaml"
DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64
ISOLATED_ENV_VARS = frozenset(
    "BASE_IMAGE DOCKERHUB_CACHE_REPO FORCE_BUILD IMAGE_TAG NIGHTLY "  # noqa: SIM905
    "REMOTE_VLLM TARGET USE_SCCACHE VLLM_BAKE_FILE VLLM_BRANCH VLLM_REPO".split()
)


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
    result = subprocess.run(
        [
            "bash",
            "-c",
            "buildkite-agent() { return 127; }\n" + command,
            "rocm-ci-cache-test",
            *(str(arg) for arg in args),
        ],
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


def docker_stage(dockerfile: str, name: str) -> str:
    header = re.search(
        rf"^FROM .* AS {re.escape(name)}\s*$",
        dockerfile,
        re.IGNORECASE | re.MULTILINE,
    )
    assert header is not None, f"missing Docker stage: {name}"
    remainder = dockerfile[header.end() :]
    next_header = re.search(r"^FROM ", remainder, re.MULTILINE)
    end = header.end() + next_header.start() if next_header else len(dockerfile)
    return dockerfile[header.start() : end]


def test_rocm_base_stages_preserve_independent_cache_boundaries() -> None:
    dockerfile = ROCM_BASE_DOCKERFILE.read_text()
    pytorch = docker_stage(dockerfile, "build_pytorch")
    assert "PYTORCH_VISION" not in pytorch
    assert "PYTORCH_AUDIO" not in pytorch
    assert "FROM base AS build_pytorch_runtime" in dockerfile
    assert "from=build_pytorch" in docker_stage(dockerfile, "build_pytorch_runtime")
    assert "FROM build_pytorch_runtime AS build_torchvision" in dockerfile
    assert "FROM build_pytorch_runtime AS build_torchaudio" in dockerfile
    for aggregate in ("debs_wheel_release", "debs"):
        stage = docker_stage(dockerfile, aggregate)
        assert all(
            f"from={dependency}" in stage
            for dependency in ("build_pytorch", "build_torchvision", "build_torchaudio")
        )


def base_cache_config(env: Mapping[str, str]) -> tuple[str, str]:
    result = run_sourced(
        ROCM_BASE_REFRESH,
        "configure_rocm_base_layer_cache\n"
        'printf "%s\\n" "$ROCM_BASE_LAYER_CACHE_REF"\n'
        'printf "%s\\n" "${ROCM_BASE_CACHE_ARGS[@]}"',
        env={"ROCM_BASE_CACHE_REPO": "example/cache"} | dict(env),
    )
    cache_ref, *args = result.stdout.splitlines()
    return cache_ref, " ".join(args)


def test_rocm_base_cache_and_stable_alias_policy() -> None:
    trusted = {
        "BUILDKITE": "true",
        "BUILDKITE_BRANCH": "main",
        "BUILDKITE_PULL_REQUEST": "false",
        "BUILDKITE_REPO": "https://github.com/vllm-project/vllm.git",
    }
    trusted_ref, trusted_args = base_cache_config(trusted)
    assert trusted_ref == "example/cache:rocm-base-main"
    assert f"cache-to type=registry,ref={trusted_ref},mode=max" in trusted_args

    pr = {
        "BUILDKITE": "true",
        "BUILDKITE_BRANCH": "feature",
        "BUILDKITE_PULL_REQUEST": "48646",
        "BUILDKITE_REPO": "https://github.com/example/vllm.git",
    }
    pr_ref, pr_args = base_cache_config(pr)
    other_fork_ref, _ = base_cache_config(
        pr | {"BUILDKITE_REPO": "https://github.com/other/vllm.git"}
    )
    assert ":rocm-base-pr-48646-" in pr_ref
    assert pr_ref != other_fork_ref
    assert "cache-from type=registry,ref=example/cache:rocm-base-main" in pr_args
    assert f"cache-to type=registry,ref={pr_ref},mode=max" in pr_args

    disabled_ref, disabled_args = base_cache_config({"ROCM_BASE_NO_CACHE": "1"})
    forced_ref, forced_args = base_cache_config({"ROCM_BASE_REFRESH_FORCE": "1"})
    assert (disabled_ref, disabled_args) == ("disabled", "--no-cache")
    assert forced_ref != "disabled" and "--no-cache" not in forced_args

    alias_result = run_sourced(
        ROCM_BASE_REFRESH,
        """
git() {
  case "$1" in
    ls-remote) printf '%s\trefs/heads/main\n' "$REMOTE_TIP" ;;
    fetch) return 0 ;;
    diff) [[ "$SAME_BASE" == 1 ]] ;;
  esac
}
docker() { printf 'docker:%s\n' "$*"; }
for SAME_BASE in 1 0; do
  printf 'case=%s\n' "$SAME_BASE"
  tag_base_image_aliases "example/base:content@$DIGEST_A" \
    example/base:descriptive example/base:base
  printf 'updated=%s\n' "$ROCM_BASE_STABLE_TAG_UPDATED"
done
""",
        env=trusted
        | {
            "BUILDKITE_COMMIT": "a" * 40,
            "DIGEST_A": DIGEST_A,
            "REMOTE_TIP": "b" * 40,
        },
    )
    unchanged, changed = alias_result.stdout.split("case=0\n")
    assert "-t example/base:base" in unchanged and "updated=1" in unchanged
    assert "-t example/base:base" not in changed and "updated=0" in changed

    fail_closed = run_sourced(
        ROCM_BASE_REFRESH,
        """
set +e
buildkite-agent() { return 42; }
metadata_set required value
printf 'metadata=%s\n' "$?"
resolve_image_digest() { printf '%s\n' "$DIGEST_A"; }
docker() { printf 'wrong 2\n'; }
find_matching_base_content_ref expected 2 example/base:content
printf 'identity=%s\n' "$?"
""",
        env={"BUILDKITE": "true", "DIGEST_A": DIGEST_A},
    )
    assert {"metadata=42", "identity=2"} <= set(fail_closed.stdout.splitlines())


@pytest.mark.parametrize("reuse", [True, False])
def test_rocm_base_reuses_or_builds_content_ref(tmp_path: Path, reuse: bool) -> None:
    trace = tmp_path / "trace"
    built = tmp_path / "built"
    run_sourced(
        ROCM_BASE_REFRESH,
        """
resolve_image_digest() {
  case "$1" in
    rocm/dev-*) printf '%s\n' "$DIGEST_A" ;;
    example/base:base-v2-*)
      [[ "$REUSE" == 1 || -f "$BUILT" ]] || return 1
      printf '%s\n' "$DIGEST_B"
      ;;
    *) printf '%s\n' "$DIGEST_B" ;;
  esac
}
docker() {
  if [[ "$*" == *"imagetools inspect"* && "$*" == *"--format"* ]]; then
    printf '%s %s\n' "$base_hash" "$metadata_version"
    return
  fi
  printf 'docker:%s\n' "$*" >> "$TRACE"
  [[ "$*" != *"buildx build"* ]] || touch "$BUILT"
}
metadata_set() { printf 'metadata:%s=%s\n' "$1" "$2" >> "$TRACE"; }
build_base_image
""",
        env={
            "BUILDKITE": "true",
            "BUILDKITE_BRANCH": "feature",
            "BUILDKITE_BUILD_NUMBER": "123",
            "BUILDKITE_COMMIT": "deadbeef",
            "BUILDKITE_PULL_REQUEST": "48646",
            "BUILDKITE_REPO": "https://github.com/example/vllm.git",
            "BUILT": str(built),
            "DIGEST_A": DIGEST_A,
            "DIGEST_B": DIGEST_B,
            "REUSE": "1" if reuse else "0",
            "ROCM_BASE_IMAGE_REPO": "example/base",
            "TRACE": str(trace),
        },
    )
    events = trace.read_text().splitlines()
    builds = [event for event in events if "buildx build" in event]
    assert bool(builds) is not reuse
    if builds:
        assert "-t example/base:base-v2-pr-48646-" in builds[0]
        assert "vllm.rocm_base.content_hash=" in builds[0]
        assert "BASE_IMAGE=" in builds[0] and f"@{DIGEST_A}" in builds[0]
        assert (
            "cache-from type=registry,ref=rocm/vllm-ci-cache:rocm-base-main"
            in (builds[0])
        )
        assert "deadbeef" not in builds[0]
        assert "_bk_123" not in builds[0]
        assert "vllm.rocm_base.image.content" not in builds[0]
    alias = next(event for event in events if "imagetools create" in event)
    assert "-t example/base:base " not in alias
    handoff = next(
        event for event in events if event.startswith("metadata:rocm-base-image=")
    )
    assert handoff.endswith(f"@{DIGEST_B}")
    assert events[-1] == "metadata:rocm-base-refresh=1"


def test_ci_base_and_native_cache_identities(tmp_path: Path) -> None:
    files, stages, csrc_files, rust_files = run_sourced(
        CI_BAKE,
        'printf "%s\\n%s\\n%s\\n%s\\n" "$DEFAULT_CI_BASE_CONTENT_FILES" '
        '"$DEFAULT_CI_BASE_DOCKERFILE_STAGES" "$DEFAULT_ROCM_CSRC_CONTENT_FILES" '
        '"$DEFAULT_ROCM_RUST_CONTENT_FILES"',
    ).stdout.splitlines()
    assert {
        ".dockerignore",
        "requirements/test/rocm.txt",
        "docker/ci-rocm.hcl",
        "tools/install_protoc.sh",
    } <= set(files.split())
    assert {"base", "rust-toolchain", "build_nixl", "build_deepep", "ci_base"} <= set(
        stages.split()
    )
    assert {".dockerignore", "tools/build_rust.py"} <= set(csrc_files.split())
    assert {".dockerignore", "tools/build_rust.py", "tools/install_protoc.sh"} <= set(
        rust_files.split()
    )

    hcl = (REPO_ROOT / "docker/ci-rocm.hcl").read_text()
    target = hcl.split('target "ci-base-rocm-ci" {', 1)[1].split("\n}", 1)[0]
    cache_from = target.split("cache-from =", 1)[1].split("cache-to =", 1)[0]
    tags = next(line for line in target.splitlines() if "tags" in line)
    assert "CI_BASE_STABLE_CACHE_REF" in cache_from
    assert "CI_BASE_IMAGE_TAG_STABLE" not in cache_from
    assert "CI_BASE_IMAGE_TAG_STABLE" in tags
    assert "CI_BASE_STABLE_CACHE_REF" not in tags
    assert '"_ci-rocm"' not in target and '"_labels"' not in target
    assert "max_jobs = CI_MAX_JOBS" in target
    missing_hash = run_sourced(
        CI_BAKE,
        "TARGET=ci-base-rocm-ci\nconfigure_ci_base_image_refs",
        check=False,
    )
    assert missing_hash.returncode != 0

    override = tmp_path / "args.hcl"
    identity = run_sourced(
        CI_BAKE,
        """
docker() {
  [[ "$*" == *different* ]] && printf 'Digest: %s\n' "$DIGEST_B" \
    || printf 'Digest: %s\n' "$DIGEST_A"
}
CI_BASE_DOCKERFILE=docker/Dockerfile.rocm
CI_BASE_CONTENT_FILES=requirements/test/rocm.txt
CI_BASE_DOCKERFILE_STAGES=ci_base
CI_BASE_CONTENT_ARGS=BASE_IMAGE
for BASE_IMAGE in rocm/example:alias-a rocm/example:alias-b rocm/example:different; do
  compute_ci_base_content_hash_once
done
BASE_IMAGE=rocm/example:alias-a
pin_base_image
VLLM_BAKE_FILE=docker/docker-bake-rocm.hcl
ROCM_ARG_OVERRIDE_PATH="$1"
write_rocm_build_arg_override
""",
        override,
        env={"DIGEST_A": DIGEST_A, "DIGEST_B": DIGEST_B},
    )
    hashes = [
        line
        for line in identity.stdout.splitlines()
        if re.fullmatch(r"[0-9a-f]{64}", line)
    ]
    assert len(hashes) == 3 and hashes[0] == hashes[1] != hashes[2]
    assert f'BASE_IMAGE = "rocm/example:alias-a@{DIGEST_A}"' in override.read_text()

    rust = run_sourced(
        CI_BAKE,
        """
docker() { printf 'Digest: %s\n' "$DIGEST_A"; }
compute_content_hash() { printf 'files\n'; }
hash_dockerfile_stages() { printf 'stages\n'; }
VLLM_BAKE_FILE=docker/docker-bake-rocm.hcl
for spec in \
  '0 fork-a commit-a' '0 fork-b commit-b' \
  '1 fork-a commit-a' '1 fork-b commit-b'; do
  read -r REMOTE_VLLM repo VLLM_BRANCH <<< "$spec"
  VLLM_REPO="https://example.com/$repo.git"
  compute_rocm_rust_content_hash
done
""",
        env={"DIGEST_A": DIGEST_A},
    )
    rust_hashes = [
        line for line in rust.stdout.splitlines() if re.fullmatch(r"[0-9a-f]{64}", line)
    ]
    assert len(rust_hashes) == 4
    assert rust_hashes[0] == rust_hashes[1]
    assert rust_hashes[2] != rust_hashes[3]


@pytest.mark.parametrize("script", [CI_BAKE, ROCM_BASE_REFRESH])
def test_digest_resolution_retries_and_rejects_bad_output(
    tmp_path: Path, script: Path
) -> None:
    calls = tmp_path / "calls"
    recovered = run_sourced(
        script,
        """
docker() {
  printf 'call\n' >> "$CALLS"
  [[ "$(wc -l < "$CALLS")" != 1 ]] || return 42
  printf 'Digest: %s\n' "$DIGEST_A"
}
resolve_image_digest example/image:tag
""",
        env={
            "CALLS": str(calls),
            "DIGEST_A": DIGEST_A,
            "ROCM_IMAGE_DIGEST_ATTEMPTS": "2",
            "ROCM_IMAGE_DIGEST_RETRY_DELAY": "0",
        },
    )
    assert recovered.stdout.strip() == DIGEST_A

    malformed = run_sourced(
        script,
        "docker() { printf 'Digest: sha256:bad\\n'; }\n"
        "resolve_image_digest example/image:tag",
        env={"ROCM_IMAGE_DIGEST_ATTEMPTS": "1"},
        check=False,
    )
    assert malformed.returncode != 0


def test_pr_reuses_matching_stable_ci_base(tmp_path: Path) -> None:
    trace = tmp_path / "trace"
    stable = "rocm/example:ci_base"
    run_sourced(
        CI_BAKE,
        """
TARGET=ci-base-rocm-ci
CI_BASE_CONTENT_HASH=content
CI_BASE_IMAGE_TAG="$STABLE"
configure_ci_base_image_refs >/dev/null
remote_image_exists() { [[ "$1" == "$STABLE" ]]; }
resolve_image_digest() { printf '%s\n' "$DIGEST_A"; }
get_remote_image_label() {
  [[ "$1" == "$STABLE@$DIGEST_A" && "$2" == vllm.ci_base.content_hash ]] \
    && printf 'content\n'
}
remote_ci_base_metadata_is_current() { return 0; }
docker() {
  printf 'docker:%s\n' "$*" >> "$TRACE"
}
confirm_remote_image_push() { return 0; }
buildkite-agent() { printf 'metadata:%s=%s\n' "$3" "$4" >> "$TRACE"; }
maybe_reuse_matching_ci_base_ref || printf 'rebuild\n' >> "$TRACE"
""",
        env={
            "BUILDKITE_COMMIT": "deadbeef",
            "BUILDKITE_PULL_REQUEST": "48646",
            "DIGEST_A": DIGEST_A,
            "STABLE": stable,
            "TRACE": str(trace),
        },
    )
    events = trace.read_text().splitlines()
    retags = "\n".join(event for event in events if event.startswith("docker:"))
    assert f"-t {stable}-content {stable}@{DIGEST_A}" in retags
    assert f"-t {stable}-deadbeef {stable}@{DIGEST_A}" in retags
    assert f"-t {stable} " not in retags
    assert f"metadata:rocm-ci-base-image={stable}-content@{DIGEST_A}" in events
    assert "rebuild" not in events


def wrapper_result(
    script: Path, *, ci_handoff: str = "", base_handoff: str = "", refreshed: str = "0"
) -> subprocess.CompletedProcess[str]:
    return run_bash(
        """
buildkite-agent() {
  [[ "$1 $2" == 'meta-data get' ]] || return 1
  case "$3" in
    rocm-ci-base-image) printf '%s\n' "$CI_HANDOFF" ;;
    rocm-base-refresh) printf '%s\n' "$REFRESHED" ;;
    rocm-base-image) printf '%s\n' "$BASE_HANDOFF" ;;
    rocm-base-push-stable-tag) printf '0\n' ;;
  esac
}
bash() { printf 'bake:%s base=%s ci=%s\n' "$*" "${BASE_IMAGE:-}" "${CI_BASE_IMAGE:-}"; }
source "$1"
""",
        script,
        env={
            "BASE_HANDOFF": base_handoff,
            "BUILDKITE": "true",
            "CI_HANDOFF": ci_handoff,
            "REFRESHED": refreshed,
        },
        check=False,
    )


def test_amd_pipeline_and_immutable_handoffs() -> None:
    steps = {
        step["key"]: step for step in yaml.safe_load(AMD_PIPELINE.read_text())["steps"]
    }
    for key in ("ensure-ci-base-amd", "image-build-amd"):
        assert steps[key]["env"]["REMOTE_VLLM"] == "0"
        assert "VLLM_BRANCH" not in steps[key]["env"]

    pinned_base = f"rocm/example:base@{DIGEST_A}"
    pinned_ci = f"rocm/example:ci_base@{DIGEST_B}"
    base_build = wrapper_result(BUILD_CI_BASE, base_handoff=pinned_base, refreshed="1")
    assert base_build.returncode == 0
    assert f"base={pinned_base}" in base_build.stdout
    assert (
        wrapper_result(
            BUILD_CI_BASE, base_handoff="rocm/example:base", refreshed="1"
        ).returncode
        != 0
    )
    image_build = wrapper_result(
        BUILD_TEST_IMAGE,
        ci_handoff=pinned_ci,
        base_handoff=pinned_base,
        refreshed="1",
    )
    assert image_build.returncode == 0
    assert f"base={pinned_base} ci={pinned_ci}" in image_build.stdout
    assert (
        wrapper_result(BUILD_TEST_IMAGE, ci_handoff="rocm/example:ci_base").returncode
        != 0
    )
    assert (
        wrapper_result(
            BUILD_TEST_IMAGE,
            ci_handoff=pinned_ci,
            base_handoff="rocm/example:base",
            refreshed="1",
        ).returncode
        != 0
    )


def test_rocm_wheel_artifact_records_native_and_build_bases(
    tmp_path: Path,
) -> None:
    wheel_dir = tmp_path / "wheel-export"
    wheel_dir.mkdir()
    (wheel_dir / "vllm-test.whl").write_text("wheel")
    immutable_base = f"rocm/example:ci_base-content@{DIGEST_A}"
    run_sourced(
        CI_BAKE,
        """
buildkite-agent() { [[ "$1 $2" == 'artifact upload' ]]; }
TARGET=artifact
configure_ci_base_image_refs
upload_wheel_artifacts_if_present >/dev/null
""",
        cwd=tmp_path,
        env={
            "BUILDKITE_COMMIT": "deadbeef",
            "CI_BASE_IMAGE": immutable_base,
            "CI_BASE_IMAGE_TAG": "rocm/example:ci_base",
        },
    )
    metadata = tmp_path / "artifacts/vllm-rocm-install"
    assert (metadata / "native-base-image.txt").read_text().strip() == (
        "rocm/example:ci_base-deadbeef"
    )
    assert (metadata / "ci-base-image.txt").read_text().strip() == immutable_base

    dockerfile = ROCM_DOCKERFILE.read_text()
    export_stage = docker_stage(dockerfile, "export_vllm")
    assert "/.dockerignore" in export_stage
    assert "/tools/install_protoc.sh" in export_stage
