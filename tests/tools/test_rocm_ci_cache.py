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
BUILD_TEST_IMAGE = REPO_ROOT / ".buildkite/scripts/rocm/build-test-image.sh"
ROCM_BASE_REFRESH = REPO_ROOT / ".buildkite/scripts/rocm/refresh-base-image.sh"
ROCM_BASE_DOCKERFILE = REPO_ROOT / "docker/Dockerfile.rocm_base"
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


def docker_stage(dockerfile: str, stage_name: str) -> str:
    header = re.search(
        rf"^FROM(?:\s+--\S+)*\s+\S+\s+AS\s+{re.escape(stage_name)}\s*$",
        dockerfile,
        re.IGNORECASE | re.MULTILINE,
    )
    assert header is not None, f"missing Docker stage: {stage_name}"
    next_header = re.search(
        r"^FROM(?:\s+--\S+)*\s+\S+(?:\s+AS\s+\S+)?\s*$",
        dockerfile[header.end() :],
        re.IGNORECASE | re.MULTILINE,
    )
    end = header.end() + next_header.start() if next_header else len(dockerfile)
    return dockerfile[header.start() : end]


def test_rocm_base_component_stages_are_independently_cacheable() -> None:
    dockerfile = ROCM_BASE_DOCKERFILE.read_text()
    pytorch = docker_stage(dockerfile, "build_pytorch")
    pytorch_runtime = docker_stage(dockerfile, "build_pytorch_runtime")
    torchvision = docker_stage(dockerfile, "build_torchvision")
    torchaudio = docker_stage(dockerfile, "build_torchaudio")

    assert "PYTORCH_VISION" not in pytorch
    assert "PYTORCH_AUDIO" not in pytorch
    assert "cp /app/pytorch/dist/*.whl /app/install" in pytorch
    assert "from=build_pytorch" in pytorch_runtime
    assert "pip install /install/*.whl" in pytorch_runtime
    assert "FROM build_pytorch_runtime AS build_torchvision" in torchvision
    assert "FROM build_pytorch_runtime AS build_torchaudio" in torchaudio
    assert "pip install /install/*.whl" not in torchvision
    assert "pip install /install/*.whl" not in torchaudio
    assert "cp /app/vision/dist/*.whl /app/install" in torchvision
    assert "cp /app/audio/dist/*.whl /app/install" in torchaudio

    for aggregate in ("debs_wheel_release", "debs"):
        stage = docker_stage(dockerfile, aggregate)
        assert "from=build_pytorch" in stage
        assert "from=build_torchvision" in stage
        assert "from=build_torchaudio" in stage


def configured_rocm_base_cache(env: Mapping[str, str]) -> tuple[str, list[str]]:
    result = run_sourced(
        ROCM_BASE_REFRESH,
        "configure_rocm_base_layer_cache\n"
        'printf "ref=%s\\n" "$ROCM_BASE_LAYER_CACHE_REF"\n'
        'printf "arg=%s\\n" "${ROCM_BASE_CACHE_ARGS[@]}"',
        env={"ROCM_BASE_CACHE_REPO": "example/cache"} | dict(env),
    )
    lines = result.stdout.splitlines()
    return lines[0].removeprefix("ref="), [
        line.removeprefix("arg=") for line in lines[1:]
    ]


def test_rocm_base_registry_cache_is_scoped_by_trust() -> None:
    trusted_ref, trusted_args = configured_rocm_base_cache(
        {
            "BUILDKITE": "true",
            "BUILDKITE_BRANCH": "main",
            "BUILDKITE_PULL_REQUEST": "false",
            "BUILDKITE_REPO": "https://github.com/vllm-project/vllm.git",
        }
    )
    assert trusted_ref == "example/cache:rocm-base-main"
    assert trusted_args == [
        "--cache-from",
        "type=registry,ref=example/cache:rocm-base-main",
        "--cache-to",
        "type=registry,ref=example/cache:rocm-base-main,mode=max,ignore-error=true",
    ]

    pr_ref, pr_args = configured_rocm_base_cache(
        {
            "BUILDKITE": "true",
            "BUILDKITE_BRANCH": "feature",
            "BUILDKITE_PULL_REQUEST": "48646",
            "BUILDKITE_REPO": "https://github.com/example/vllm.git",
        }
    )
    assert re.fullmatch(r"example/cache:rocm-base-pr-48646-[0-9a-f]{12}", pr_ref)
    assert pr_args == [
        "--cache-from",
        f"type=registry,ref={pr_ref}",
        "--cache-from",
        "type=registry,ref=example/cache:rocm-base-main",
        "--cache-to",
        f"type=registry,ref={pr_ref},mode=max,ignore-error=true",
    ]

    preview_ref, preview_args = configured_rocm_base_cache(
        {
            "BUILDKITE": "true",
            "BUILDKITE_BRANCH": "main",
            "BUILDKITE_PULL_REQUEST": "false",
            "BUILDKITE_REPO": "https://github.com/example/vllm.git",
        }
    )
    assert preview_ref.startswith("example/cache:rocm-base-preview-main-")
    assert "type=registry,ref=example/cache:rocm-base-main" in preview_args
    assert preview_args[-1].startswith(
        f"type=registry,ref={preview_ref},mode=max,ignore-error=true"
    )


def test_rocm_base_registry_cache_can_be_disabled() -> None:
    cache_ref, cache_args = configured_rocm_base_cache({"ROCM_BASE_NO_CACHE": "1"})
    assert cache_ref == "disabled"
    assert cache_args == ["--no-cache"]

    forced_ref, forced_args = configured_rocm_base_cache(
        {"ROCM_BASE_REFRESH_FORCE": "1"}
    )
    assert forced_ref == "disabled"
    assert forced_args == ["--no-cache"]


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
    tags = next(
        line.strip() for line in target.splitlines() if line.strip().startswith("tags")
    )
    assert "CI_BASE_STABLE_CACHE_REF" in cache_from
    assert "CI_BASE_IMAGE_TAG_STABLE" not in cache_from
    assert "CI_BASE_IMAGE_TAG_STABLE" in tags
    assert "CI_BASE_STABLE_CACHE_REF" not in tags

    missing_hash = run_sourced(
        CI_BAKE,
        'TARGET="ci-base-rocm-ci"\nconfigure_ci_base_image_refs',
        check=False,
    )
    assert missing_hash.returncode != 0
    assert "ci_base builds require a content hash" in missing_hash.stderr


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
    ("env", "authorized"),
    [
        ({"BUILDKITE_PULL_REQUEST": "48646", "CI_BASE_PUSH_STABLE_TAG": "1"}, False),
        (
            {
                "BUILDKITE_PULL_REQUEST": "false",
                "BUILDKITE_BRANCH": "main",
                "NIGHTLY": "1",
            },
            True,
        ),
    ],
)
def test_stable_alias_authorization(env: dict[str, str], authorized: bool) -> None:
    result = run_sourced(
        CI_BAKE,
        'TARGET="ci-base-rocm-ci"\n'
        'CI_BASE_CONTENT_HASH="content"\n'
        'CI_BASE_IMAGE_TAG="rocm/example:ci_base"\n'
        "configure_ci_base_image_refs >/dev/null\n"
        "ci_base_output_refs",
        env=env,
    )
    outputs = set(result.stdout.splitlines())
    assert "rocm/example:ci_base-content" in outputs
    assert ("rocm/example:ci_base" in outputs) is authorized


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


@pytest.mark.parametrize(
    ("handoff", "valid"),
    [
        (f"rocm/example:ci_base-content@{DIGEST_A}", True),
        ("", False),
        ("rocm/example:ci_base-content", False),
    ],
)
def test_build_step_requires_immutable_handoff(handoff: str, valid: bool) -> None:
    result = run_bash(
        """
buildkite-agent() {
  [[ "$*" == "meta-data get rocm-ci-base-image" ]] || return 1
  printf '%s\n' "$HANDOFF"
}
bash() { printf 'selected=%s\n' "${CI_BASE_IMAGE:-local}"; }
source "$1"
""",
        BUILD_TEST_IMAGE,
        env={"BUILDKITE": "true", "HANDOFF": handoff},
        check=False,
    )
    assert (result.returncode == 0) is valid
    if valid:
        assert f"selected={handoff}" in result.stdout
    else:
        assert "selected=" not in result.stdout
        assert "handoff metadata is missing or invalid" in result.stderr
