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

    sccache_hashes = run_sourced(
        ROCM_BASE_REFRESH,
        """
ROCM_BASE_CONTENT_ARGS=SCCACHE_ENDPOINT
for SCCACHE_ENDPOINT in https://cache-a.example https://cache-b.example; do
  compute_base_content_hash 1 "$DIGEST_A"
done
""",
        env={"DIGEST_A": DIGEST_A},
    ).stdout.splitlines()
    assert len(sccache_hashes) == 2 and sccache_hashes[0] != sccache_hashes[1]


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


def test_rocm_base_cache_and_stable_alias_policy(tmp_path: Path) -> None:
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

    identity_calls = tmp_path / "identity-calls"
    fail_closed = run_sourced(
        ROCM_BASE_REFRESH,
        """
set +e
buildkite-agent() { return 42; }
metadata_set required value
printf 'metadata=%s\n' "$?"
resolve_image_digest() { printf '%s\n' "$DIGEST_A"; }
version="$DEFAULT_ROCM_BASE_METADATA_VERSION"
docker() {
  printf 'call\n' >> "$CALLS"
  if [[ "$(wc -l < "$CALLS")" == 1 ]]; then
    printf 'wrong %s\n' "$version"
  else
    printf 'expected %s\n' "$version"
  fi
}
match=$(find_matching_base_content_ref expected "$version" example/base:content)
printf 'recovered=%s calls=%s ref=%s\n' "$?" "$(wc -l < "$CALLS")" "$match"
: > "$CALLS"
docker() { printf 'call\n' >> "$CALLS"; printf 'wrong %s\n' "$version"; }
find_matching_base_content_ref expected "$version" example/base:content >/dev/null
printf 'identity=%s calls=%s\n' "$?" "$(wc -l < "$CALLS")"
""",
        env={
            "BUILDKITE": "true",
            "CALLS": str(identity_calls),
            "DIGEST_A": DIGEST_A,
            "ROCM_IMAGE_DIGEST_ATTEMPTS": "2",
            "ROCM_IMAGE_DIGEST_RETRY_DELAY": "0",
        },
    )
    lines = set(fail_closed.stdout.splitlines())
    assert "metadata=42" in lines
    assert f"recovered=0 calls=2 ref=example/base:content@{DIGEST_A}" in lines
    assert "identity=2 calls=2" in lines

    change_detection = run_sourced(
        ROCM_BASE_REFRESH,
        """
set +e
git() { return 128; }
git_diff_changed_base HEAD~1..HEAD
printf 'diff=%s\n' "$?"
ROCM_BASE_REFRESH_DIFF_UNAVAILABLE=1
rocm_base_changed
printf 'unavailable=%s\n' "$?"
""",
    ).stdout.splitlines()
    assert {"diff=2", "unavailable=0"} <= set(change_detection)

    stable_repair = run_sourced(
        ROCM_BASE_REFRESH,
        """
set +e
git() {
  case "$1" in
    rev-parse|diff) return 0 ;;
  esac
}
extract_arg_default() { printf 'example/parent:base\n'; }
resolve_image_digest() { printf '%s\n' "$DIGEST_A"; }
compute_base_content_hash() { printf 'expected\n'; }
find_matching_base_content_ref() { [[ "$MODE" == current ]]; }
for MODE in current stale; do
  rocm_base_changed
  printf '%s=%s\n' "$MODE" "$?"
done
""",
        env=trusted | {"BUILDKITE_COMMIT": "a" * 40, "DIGEST_A": DIGEST_A},
        check=False,
    ).stdout.splitlines()
    assert {"current=1", "stale=0"} <= set(stable_repair)


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
    assert "CI_BASE_IMAGE_TAG_STABLE" not in tags
    assert "CI_BASE_IMAGE_TAG_COMMIT_EXTRA" in tags
    assert "CI_BASE_STABLE_CACHE_REF" not in tags
    assert '"_ci-rocm"' not in target and '"_labels"' not in target
    assert "max_jobs = CI_MAX_JOBS" in target
    missing_hash = run_sourced(
        CI_BAKE,
        "TARGET=ci-base-rocm-ci\nconfigure_ci_base_image_refs",
        check=False,
    )
    assert missing_hash.returncode != 0

    metadata_output = run_sourced(
        CI_BAKE,
        """
compute_content_hash() { printf '%064d\n' 0; }
get_content_arg_names() { printf 'BASE_IMAGE\nNIXL_BRANCH\n'; }
resolve_dockerfile_arg_value() {
  if [[ "$2" == BASE_IMAGE ]]; then
    printf '%s\n' "$BASE_ALIAS"
  else
    printf 'value-%s\n' "$2"
  fi
}
resolve_image_digest() { printf '%s\n' "$DIGEST_A"; }
CI_BASE_CONTENT_HASH=content
CI_BASE_CONTENT_FILES=requirements/common.txt
CI_BASE_DOCKERFILE_STAGES='base ci_base'
for BASE_ALIAS in example/base:first example/base:second; do
  ci_base_metadata_pairs
  printf '%s\n' __END_METADATA__
done
""",
        env={
            "BUILDKITE_BUILD_ID": "volatile-build-id",
            "BUILDKITE_BUILD_NUMBER": "123",
            "BUILDKITE_COMMIT": "deadbeef",
            "DEEPEP_CACHE_KEY": "deepep-key",
            "DIGEST_A": DIGEST_A,
            "NIXL_CACHE_KEY": "nixl-key",
            "PYTORCH_ROCM_ARCH": "gfx942",
            "ROCSHMEM_CACHE_KEY": "rocshmem-key",
        },
    ).stdout
    rendered_metadata = [
        dict(line.split("\t", 1) for line in block.splitlines() if "\t" in line)
        for block in metadata_output.split("__END_METADATA__")
        if block.strip()
    ]
    assert len(rendered_metadata) == 2
    assert rendered_metadata[0] == rendered_metadata[1]
    metadata = rendered_metadata[0]
    required_metadata = set(
        """vllm.ci_base.metadata_version vllm.ci_base.content_hash
        vllm.ci_base.content_files_hash vllm.ci_base.content_files
        vllm.ci_base.content_args vllm.ci_base.dockerfile
        vllm.ci_base.dockerfile_stages vllm.rocm.base_image_digest
        vllm.rocm.pytorch_rocm_arch vllm.rocm.nic_backend
        vllm.rocm.nixl_commit vllm.rocm.nixl_cache_key
        vllm.rocm.rocshmem_commit vllm.rocm.rocshmem_cache_key
        vllm.rocm.deepep_commit vllm.rocm.deepep_cache_key""".split()  # noqa: SIM905
    )
    assert required_metadata <= metadata.keys()
    assert all(metadata[key] for key in required_metadata)
    assert metadata["vllm.rocm.base_image_digest"] == DIGEST_A
    assert {
        "vllm.ci_base.git_commit",
        "vllm.ci_base.git_branch",
        "vllm.ci_base.vllm_branch",
    }.isdisjoint(metadata)
    assert not any(
        key.startswith(("vllm.buildkite.", "vllm.ci_base.image.")) for key in metadata
    )

    ref_command = """
TARGET=ci-base-rocm-ci
CI_BASE_CONTENT_HASH=content
CI_BASE_IMAGE_TAG=example/base:ci_base
configure_cache_write_scope >/dev/null
configure_ci_base_image_refs >/dev/null
uses_rocm_csrc_cache() { return 0; }
uses_rocm_rust_cache() { return 0; }
compute_rocm_csrc_content_hash() { printf '%064d\n' 1; }
compute_rocm_rust_content_hash() { printf '%064d\n' 2; }
compute_rocm_csrc_content_hash_if_needed >/dev/null
compute_rocm_rust_content_hash_if_needed >/dev/null
NIXL_CACHE_KEY=nixl-key
printf 'scope=%s\n' "$ROCM_CACHE_WRITE_SUFFIX"
printf 'content=%s\n' "$CI_BASE_IMAGE_TAG_CONTENT_REF"
printf 'trusted=%s\n' "$CI_BASE_TRUSTED_CONTENT_REF"
printf 'csrc=%s\n' "$ROCM_CSRC_CONTENT_CACHE_REF"
printf 'csrc_trusted=%s\n' "$ROCM_CSRC_TRUSTED_CONTENT_CACHE_REF"
printf 'rust=%s\n' "$ROCM_RUST_CONTENT_CACHE_REF"
printf 'rust_trusted=%s\n' "$ROCM_RUST_TRUSTED_CONTENT_CACHE_REF"
printf 'dependency=%s\n' "$(dependency_cache_ref_for_target nixl-rocm-ci)"
printf 'dependency_trusted=%s\n' \
  "$(dependency_cache_ref_for_target nixl-rocm-ci trusted)"
printf '%s\n' CANDIDATES
ci_base_candidate_refs
printf '%s\n' OUTPUTS
ci_base_output_refs
"""
    trusted_env = {
        "BUILDKITE": "true",
        "BUILDKITE_BRANCH": "main",
        "BUILDKITE_COMMIT": "a" * 40,
        "BUILDKITE_PULL_REQUEST": "false",
        "BUILDKITE_REPO": "https://github.com/vllm-project/vllm.git",
    }
    pr_env = {
        "BUILDKITE": "true",
        "BUILDKITE_BRANCH": "feature",
        "BUILDKITE_COMMIT": "b" * 40,
        "BUILDKITE_PULL_REQUEST": "48646",
        "BUILDKITE_PULL_REQUEST_REPO": "https://github.com/example/vllm.git",
        "BUILDKITE_REPO": "https://github.com/vllm-project/vllm.git",
    }

    def image_refs(
        env: Mapping[str, str],
    ) -> tuple[dict[str, str], list[str], list[str]]:
        output = run_sourced(CI_BAKE, ref_command, env=env).stdout.splitlines()
        candidates_at = output.index("CANDIDATES")
        outputs_at = output.index("OUTPUTS")
        values = dict(line.split("=", 1) for line in output[:candidates_at])
        return values, output[candidates_at + 1 : outputs_at], output[outputs_at + 1 :]

    trusted_values, _, trusted_outputs = image_refs(trusted_env)
    pr_values, pr_candidates, pr_outputs = image_refs(pr_env)
    assert trusted_values["scope"] == ""
    assert trusted_values["content"] == trusted_values["trusted"]
    assert pr_values["scope"]
    assert pr_values["content"] != pr_values["trusted"]
    assert pr_values["trusted"] == trusted_values["trusted"]
    assert pr_values["trusted"] in pr_candidates
    assert pr_values["content"] in pr_outputs
    assert pr_values["trusted"] not in pr_outputs
    assert trusted_values["content"] in trusted_outputs
    for name in ("csrc", "rust", "dependency"):
        trusted_name = f"{name}_trusted"
        assert trusted_values[name] == trusted_values[trusted_name]
        assert pr_values[trusted_name] == trusted_values[trusted_name]
        assert pr_values[name] != pr_values[trusted_name]

    tag_limits = run_sourced(
        CI_BAKE,
        """
ROCM_CACHE_NAMESPACE=abcdefghijklm
BUILDKITE_BRANCH=$(printf 'branch%.0s' {1..20})
configure_cache_write_scope >/dev/null
ROCM_CACHE_BRANCH_TAG=$(compose_cache_branch_tag owner/repository "$BUILDKITE_BRANCH")
ROCSHMEM_CACHE_KEY=$(
  compose_dependency_cache_key "$(printf 'pin%.0s' {1..30})" material
)
TARGET=csrc-rocm-ci
compute_rocm_csrc_content_hash() { printf '%064d\n' 1; }
compute_rocm_csrc_content_hash_if_needed >/dev/null
branch_ref="example/cache:csrc-rocm-${ROCM_CACHE_NAMESPACE}"
branch_ref+="-branch-${ROCM_CACHE_BRANCH_TAG}${ROCM_CACHE_WRITE_SUFFIX}"
for ref in \
  "$ROCM_CSRC_CONTENT_CACHE_REF" \
  "$(dependency_cache_ref_for_target rocshmem-rocm-ci)" \
  "$branch_ref"; do
  tag="${ref##*:}"
  printf 'length=%s\n' "${#tag}"
done
set +e
ROCM_CACHE_NAMESPACE=abcdefghijklmn
configure_cache_write_scope >/dev/null 2>&1
printf 'invalid=%s\n' "$?"
""",
    ).stdout.splitlines()
    lengths = [int(line.removeprefix("length=")) for line in tag_limits[:-1]]
    assert lengths and max(lengths) <= 128
    assert tag_limits[-1] != "invalid=0"

    policy_command = """
git() { printf '%s\trefs/heads/main\n' "$REMOTE_TIP"; }
should_push_stable_ci_base_tag
"""

    def stable_policy(env: Mapping[str, str]) -> int:
        return run_sourced(CI_BAKE, policy_command, env=env, check=False).returncode

    promotion_env = trusted_env | {
        "CI_BASE_PUSH_STABLE_TAG": "1",
        "REMOTE_TIP": trusted_env["BUILDKITE_COMMIT"],
    }
    assert stable_policy(promotion_env) == 0
    assert stable_policy(promotion_env | {"REMOTE_TIP": "b" * 40}) != 0
    assert stable_policy(pr_env | {"CI_BASE_PUSH_STABLE_TAG": "1"}) != 0

    version_refs = run_sourced(
        CI_BAKE,
        """
TARGET=ci-base-rocm-ci
CI_BASE_CONTENT_HASH=content
CI_BASE_IMAGE_TAG=example/base:ci_base
for CI_BASE_METADATA_VERSION in 3 4; do
  CI_BASE_IMAGE_TAG=example/base:ci_base
  configure_ci_base_image_refs >/dev/null
  printf '%s\n' "$CI_BASE_IMAGE_TAG_CONTENT_REF"
done
""",
    ).stdout.splitlines()
    assert len(version_refs) == 2 and version_refs[0] != version_refs[1]

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

    csrc_dir = tmp_path / "csrc-hash"
    csrc_dir.mkdir()
    csrc_dockerfile = csrc_dir / "Dockerfile.rocm"
    csrc_bake = csrc_dir / "docker-bake-rocm.hcl"
    csrc_bake.write_text("")
    csrc_dockerfile.write_text(ROCM_DOCKERFILE.read_text())

    def csrc_hash() -> str:
        return run_sourced(
            CI_BAKE,
            "docker() { printf 'Digest: %s\\n' \"$DIGEST_A\"; }\n"
            "compute_rocm_csrc_content_hash",
            env={"DIGEST_A": DIGEST_A, "VLLM_BAKE_FILE": str(csrc_bake)},
        ).stdout.strip()

    original_csrc_hash = csrc_hash()
    csrc_dockerfile.write_text(
        csrc_dockerfile.read_text().replace(
            "ENV VLLM_TARGET_DEVICE=rocm",
            "ENV VLLM_TARGET_DEVICE=rocm-mutated",
            1,
        )
    )
    assert csrc_hash() != original_csrc_hash

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

    hash_repo = tmp_path / "hash-repo"
    content_dir = hash_repo / "content"
    content_dir.mkdir(parents=True)
    tracked = content_dir / "tracked.txt"
    tracked.write_text("first")
    run_bash("git init -q\ngit add content/tracked.txt", cwd=hash_repo)

    def content_hash() -> str:
        return run_sourced(
            CI_BAKE, "compute_content_hash content", cwd=hash_repo
        ).stdout.strip()

    original_hash = content_hash()
    (content_dir / "untracked.txt").write_text("residue")
    assert content_hash() == original_hash
    tracked.write_text("changed")
    assert content_hash() != original_hash


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

    if script == CI_BAKE:
        identity_calls = tmp_path / "identity-calls"
        identity = run_sourced(
            CI_BAKE,
            """
docker() {
  printf 'call\n' >> "$CALLS"
  if [[ "$(wc -l < "$CALLS")" == 1 ]]; then
    printf 'stale|%s|%064d|%s\n' "$DEFAULT_CI_BASE_METADATA_VERSION" 0 "$DIGEST_A"
  else
    printf 'content|%s|%064d|%s\n' "$DEFAULT_CI_BASE_METADATA_VERSION" 0 "$DIGEST_A"
  fi
}
CI_BASE_CONTENT_HASH=content
remote_ci_base_identity_is_current_with_retry example/base:content
printf 'calls=%s\n' "$(wc -l < "$CALLS")"
""",
            env={
                "CALLS": str(identity_calls),
                "CI_BASE_LABEL_ATTEMPTS": "2",
                "CI_BASE_LABEL_RETRY_DELAY": "0",
                "DIGEST_A": DIGEST_A,
            },
        )
        assert "calls=2" in identity.stdout.splitlines()

        probe_calls = tmp_path / "probe-calls"
        probe = run_sourced(
            CI_BAKE,
            """
docker() {
  printf 'call\n' >> "$CALLS"
  [[ "$(wc -l < "$CALLS")" != 1 ]]
}
remote_image_exists example/image:cache
printf 'calls=%s\n' "$(wc -l < "$CALLS")"
""",
            env={
                "CALLS": str(probe_calls),
                "ROCM_REGISTRY_PROBE_ATTEMPTS": "2",
                "ROCM_REGISTRY_PROBE_RETRY_DELAY": "0",
            },
        )
        assert "calls=2" in probe.stdout.splitlines()


def test_pr_reuses_matching_stable_ci_base(tmp_path: Path) -> None:
    trace = tmp_path / "trace"
    stable = "rocm/example:ci_base"
    run_sourced(
        CI_BAKE,
        """
TARGET=ci-base-rocm-ci
CI_BASE_CONTENT_HASH=content
CI_BASE_IMAGE_TAG="$STABLE"
configure_cache_write_scope >/dev/null
configure_ci_base_image_refs >/dev/null
resolve_image_digest() { printf '%s\n' "$DIGEST_A"; }
remote_ci_base_identity_is_current_with_retry() {
  [[ "$1" == "$CI_BASE_TRUSTED_CONTENT_REF@$DIGEST_A" ]]
}
docker() {
  printf 'docker:%s\n' "$*" >> "$TRACE"
}
confirm_remote_image_push() { return 0; }
buildkite-agent() { printf 'metadata:%s=%s\n' "$3" "$4" >> "$TRACE"; }
printf 'source:%s@%s\ncontent:%s\ncommit:%s\n' \
  "$CI_BASE_TRUSTED_CONTENT_REF" "$DIGEST_A" \
  "$CI_BASE_IMAGE_TAG_CONTENT_REF" "$CI_BASE_IMAGE_TAG_COMMIT_REF" >> "$TRACE"
maybe_reuse_matching_ci_base_ref || printf 'rebuild\n' >> "$TRACE"
""",
        env={
            "BUILDKITE_COMMIT": "deadbeef",
            "BUILDKITE_BRANCH": "feature",
            "BUILDKITE_PULL_REQUEST": "48646",
            "BUILDKITE_PULL_REQUEST_REPO": "https://github.com/example/vllm.git",
            "BUILDKITE_REPO": "https://github.com/vllm-project/vllm.git",
            "BUILDKITE": "true",
            "DIGEST_A": DIGEST_A,
            "STABLE": stable,
            "TRACE": str(trace),
        },
    )
    events = trace.read_text().splitlines()
    refs = {
        key: next(
            event.removeprefix(f"{key}:")
            for event in events
            if event.startswith(f"{key}:")
        )
        for key in ("source", "content", "commit")
    }
    retags = "\n".join(event for event in events if event.startswith("docker:"))
    assert "--prefer-index=false" in retags
    assert f"-t {refs['content']} {refs['source']}" in retags
    assert f"-t {refs['commit']} {refs['source']}" in retags
    assert f"-t {stable} " not in retags
    assert f"metadata:rocm-ci-base-image={refs['content']}@{DIGEST_A}" in events
    assert "rebuild" not in events

    failure_trace = tmp_path / "failure-trace"
    failures = run_sourced(
        CI_BAKE,
        """
find_matching_ci_base_ref() { printf 'example/base:source@%s\n' "$DIGEST_A"; }
refresh_ci_base_tags_from_ref() { [[ "$MODE" != retag ]]; }
validate_ci_base_output_refs() { return 0; }
promote_stable_ci_base_tag() { [[ "$MODE" != promotion ]]; }
publish_ci_base_handoff_ref() {
  [[ "$MODE" != handoff ]] || return 1
  buildkite-agent meta-data set rocm-ci-base-image "$1"
}
buildkite-agent() { printf 'metadata:%s=%s\n' "$3" "$4" >> "$TRACE"; }
for MODE in retag promotion handoff; do
  if ! maybe_reuse_matching_ci_base_ref; then
    printf 'rebuild:%s\n' "$MODE"
  fi
done
""",
        env={"DIGEST_A": DIGEST_A, "TRACE": str(failure_trace)},
    )
    assert set(failures.stdout.splitlines()) >= {
        "rebuild:retag",
        "rebuild:promotion",
        "rebuild:handoff",
    }
    assert not failure_trace.exists()

    bake_validation = run_sourced(
        CI_BAKE,
        """
set +e
TARGET=ci-base-rocm-ci
IMAGE_TAG=example/base:content
CI_BASE_IMAGE_TAG=example/base:content
CI_BASE_IMAGE_TAG_COMMIT_EXTRA=example/base:commit
BAKE_FILES=()
BAKE_TARGETS=(ci-base-rocm-ci)
docker() { return 41; }
refresh_ci_base_tags_from_ref() { return 0; }
confirm_remote_image_push() {
  [[ "$MODE" == all || "$1" != example/base:commit ]]
}
for MODE in partial all; do
  run_bake >/dev/null 2>&1
  printf '%s=%s\n' "$MODE" "$?"
done
""",
    ).stdout.splitlines()
    assert {"partial=41", "all=0"} <= set(bake_validation)


def wrapper_result(
    script: Path,
    *,
    ci_handoff: str = "",
    parent_handoff: str = "",
    base_handoff: str = "",
    refreshed: str = "0",
) -> subprocess.CompletedProcess[str]:
    return run_bash(
        """
buildkite-agent() {
  [[ "$1 $2" == 'meta-data get' ]] || return 1
  case "$3" in
    rocm-ci-base-image) printf '%s\n' "$CI_HANDOFF" ;;
    rocm-ci-base-parent-image) printf '%s\n' "$PARENT_HANDOFF" ;;
    rocm-base-refresh) printf '%s\n' "$REFRESHED" ;;
    rocm-base-image) printf '%s\n' "$BASE_HANDOFF" ;;
    rocm-base-push-stable-tag) printf '0\n' ;;
  esac
}
bash() {
  printf 'bake:%s base=%s ci=%s remote=%s branch=%s\n' \
    "$*" "${BASE_IMAGE:-}" "${CI_BASE_IMAGE:-}" \
    "${REMOTE_VLLM:-}" "${VLLM_BRANCH-unset}"
}
source "$1"
""",
        script,
        env={
            "BASE_HANDOFF": base_handoff,
            "BUILDKITE": "true",
            "CI_HANDOFF": ci_handoff,
            "PARENT_HANDOFF": parent_handoff,
            "REFRESHED": refreshed,
            "REMOTE_VLLM": "1",
            "VLLM_BRANCH": "hostile-remote-branch",
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
        parent_handoff=pinned_base,
        base_handoff=pinned_base,
        refreshed="1",
    )
    assert image_build.returncode == 0
    assert f"base={pinned_base} ci={pinned_ci}" in image_build.stdout
    assert "remote=0 branch=unset" in image_build.stdout
    no_refresh = wrapper_result(
        BUILD_TEST_IMAGE,
        ci_handoff=pinned_ci,
        parent_handoff=pinned_base,
    )
    assert no_refresh.returncode == 0
    assert f"base={pinned_base} ci={pinned_ci}" in no_refresh.stdout
    assert (
        wrapper_result(
            BUILD_TEST_IMAGE,
            ci_handoff="rocm/example:ci_base",
            parent_handoff=pinned_base,
        ).returncode
        != 0
    )
    assert wrapper_result(BUILD_TEST_IMAGE, ci_handoff=pinned_ci).returncode != 0
    assert (
        wrapper_result(
            BUILD_TEST_IMAGE,
            ci_handoff=pinned_ci,
            parent_handoff=pinned_base,
            base_handoff="rocm/example:base",
            refreshed="1",
        ).returncode
        != 0
    )

    parent_publish = run_sourced(
        CI_BAKE,
        """
buildkite-agent() { printf '%s=%s\n' "$3" "$4"; }
resolve_image_digest() { printf '%s\n' "$DIGEST_A"; }
compute_ci_base_content_hash() { printf 'content\n'; }
TARGET=ci-base-rocm-ci
CI_BASE_CONTENT_FILES=requirements/common.txt
BASE_IMAGE=rocm/example:base
compute_ci_base_hash_if_needed
""",
        env={"BUILDKITE": "true", "DIGEST_A": DIGEST_A},
    )
    assert f"rocm-ci-base-parent-image={pinned_base}" in parent_publish.stdout


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
    assert (metadata / "native-base-image.txt").read_text().strip() == immutable_base
    assert (metadata / "ci-base-image.txt").read_text().strip() == immutable_base

    dockerfile = ROCM_DOCKERFILE.read_text()
    dependencies = docker_stage(dockerfile, "build_vllm_dependencies")
    csrc = docker_stage(dockerfile, "csrc-build")
    build_vllm = docker_stage(dockerfile, "build_vllm")
    assert "requirements/rocm.txt" in dependencies
    assert "uv pip install --system" in dependencies
    assert "COPY --from=fetch_vllm ${COMMON_WORKDIR}/vllm " not in dependencies
    assert csrc.startswith("FROM build_vllm_dependencies AS csrc-build")
    assert "uv pip install --system -r requirements/rocm.txt" not in csrc
    assert build_vllm.startswith("FROM build_vllm_dependencies AS build_vllm")
    assert "COPY --from=fetch_vllm ${COMMON_WORKDIR}/vllm " in build_vllm
    assert "uv pip install --system -r requirements/rocm.txt" not in build_vllm

    export_stage = docker_stage(dockerfile, "export_vllm")
    assert "/.dockerignore" in export_stage
    assert "/tools/install_protoc.sh" in export_stage
