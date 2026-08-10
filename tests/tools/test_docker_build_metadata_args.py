# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import shlex
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / ".buildkite" / "scripts" / "docker-build-metadata-args.sh"
CI_BAKE_ROCM = REPO_ROOT / ".buildkite" / "scripts" / "ci-bake-rocm.sh"


def run_helper(
    *args: str,
    env: dict[str, str] | None = None,
    path: str | None = None,
) -> list[str]:
    helper_env = {"PATH": path or os.environ["PATH"]}
    if env:
        helper_env.update(env)
    result = subprocess.run(
        ["bash", str(HELPER), *args],
        check=True,
        env=helper_env,
        stdout=subprocess.PIPE,
        text=True,
    )
    return shlex.split(result.stdout)


def option_values(args: list[str], option: str) -> list[str]:
    return [args[i + 1] for i, arg in enumerate(args[:-1]) if arg == option]


def build_args(args: list[str]) -> dict[str, str]:
    values = {}
    for value in option_values(args, "--build-arg"):
        key, arg_value = value.split("=", 1)
        values[key] = arg_value
    return values


def init_context_test_repo(
    tmp_path: Path,
    *,
    reference_clone: bool = False,
) -> tuple[Path, Path, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    regular = repo / "regular file.txt"
    executable = repo / "executable.sh"
    link = repo / "regular-link"
    trailing_newline = repo / "trailing-newline\n"
    trailing_newline_link = repo / "trailing-newline-link"
    regular.write_text("regular\n")
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    link.symlink_to(regular.name)
    trailing_newline.write_text("odd filename\n")
    trailing_newline_link.symlink_to(trailing_newline.name)
    (repo / ".dockerignore").write_text(
        "/bake-config-build-*.json\n"
        "/wheel-export/\n"
        "/artifacts/vllm-rocm-install/\n"
        "docker-output/\n"
    )
    (repo / ".gitignore").write_text("ignored.bin\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "ci@example.com"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "CI"], cwd=repo, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=repo, check=True)
    subprocess.run(["git", "config", "tag.gpgsign", "false"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "update-index", "--chmod=-x", regular.name],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "update-index", "--chmod=+x", executable.name],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    subprocess.run(["git", "tag", "v1.0.0"], cwd=repo, check=True)
    (repo / "child.txt").write_text("child commit\n")
    subprocess.run(["git", "add", "child.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "child"], cwd=repo, check=True)
    if reference_clone:
        checkout = tmp_path / "reference-checkout"
        subprocess.run(
            ["git", "clone", "-q", "--reference", str(repo), str(repo), str(checkout)],
            check=True,
        )
        repo = checkout
        regular = repo / regular.name
        executable = repo / executable.name
        link = repo / link.name
    regular.chmod(0o664)
    executable.chmod(0o775)
    return repo, regular, executable, link


def run_context_preparation(
    repo: Path,
    script_tmp: Path,
    *,
    target: str = "ci-base-rocm-ci-with-deps",
) -> subprocess.CompletedProcess[str]:
    script_tmp.mkdir()
    command = f"""
set -euo pipefail
source {shlex.quote(str(CI_BAKE_ROCM))}
trap - EXIT
SCRIPT_TMP_DIR={shlex.quote(str(script_tmp))}
BUILD_CONTEXT_OVERRIDE_PATH="$SCRIPT_TMP_DIR/build-context-override.hcl"
BUILDKITE=true
BUILDKITE_COMMIT=$(git rev-parse HEAD)
REMOTE_VLLM=0
TARGET={shlex.quote(target)}
BAKE_FILES=(-f base.hcl)
prepare_ci_build_context
write_build_context_override
printf 'CONTEXT=%s\n' "$ROCM_BUILD_CONTEXT_ROOT"
printf 'CONTEXT_INDEX=%s\n' "$ROCM_BUILD_CONTEXT_INDEX"
printf 'OVERRIDE=%s\n' "$BUILD_CONTEXT_OVERRIDE_PATH"
printf 'BAKE_FILE_LAST=%s\n' "${{BAKE_FILES[-1]}}"
printf 'BAKE_ALLOW_ARGS=%s\n' "${{BAKE_ALLOW_ARGS[*]}}"
printf 'CONTEXT_HASH=%s\n' "$(compute_content_hash .)"
ROCM_BUILD_CONTEXT_ROOT=""
ROCM_BUILD_CONTEXT_INDEX=""
printf 'SOURCE_HASH=%s\n' "$(compute_content_hash .)"
"""
    return subprocess.run(
        ["bash", "-c", command],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )


def context_result_values(result: subprocess.CompletedProcess[str]) -> dict[str, str]:
    return dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )


def compute_prepared_context_hash(repo: Path, context: Path, index: Path) -> str:
    command = f"""
set -euo pipefail
source {shlex.quote(str(CI_BAKE_ROCM))}
trap - EXIT
ROCM_BUILD_CONTEXT_ROOT={shlex.quote(str(context))}
ROCM_BUILD_CONTEXT_INDEX={shlex.quote(str(index))}
compute_content_hash .
"""
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_release_metadata_args_prefer_pipeline_id() -> None:
    args = run_helper(
        "cu130-ubuntu2404",
        env={
            "BUILDKITE": "1",
            "BUILDKITE_COMMIT": "abc123",
            "BUILDKITE_PIPELINE_ID": "pipe-uuid",
            "BUILDKITE_PIPELINE_SLUG": "release",
            "BUILDKITE_BUILD_URL": "https://buildkite.example/vllm/builds/1",
            "RELEASE_VERSION": "v0.20.0",
        },
    )

    assert build_args(args) == {
        "VLLM_BUILD_COMMIT": "abc123",
        "VLLM_BUILD_PIPELINE": "pipe-uuid",
        "VLLM_BUILD_URL": "https://buildkite.example/vllm/builds/1",
        "VLLM_IMAGE_TAG": "vllm/vllm-openai:v0.20.0-cu130-ubuntu2404",
    }
    expected_tag = (
        "public.ecr.aws/q9t5s3a7/vllm-release-repo:"
        f"abc123-{os.uname().machine}-cu130-ubuntu2404"
    )
    assert option_values(args, "--tag") == [expected_tag]


def test_nightly_metadata_args_fall_back_to_pipeline_slug() -> None:
    args = run_helper(
        "ubuntu2404",
        env={
            "BUILDKITE": "1",
            "BUILDKITE_COMMIT": "def456",
            "BUILDKITE_PIPELINE_SLUG": "release",
            "BUILDKITE_BUILD_URL": "https://buildkite.example/vllm/builds/2",
            "NIGHTLY": "1",
        },
    )

    assert build_args(args) == {
        "VLLM_BUILD_COMMIT": "def456",
        "VLLM_BUILD_PIPELINE": "release",
        "VLLM_BUILD_URL": "https://buildkite.example/vllm/builds/2",
        "VLLM_IMAGE_TAG": "vllm/vllm-openai:nightly-def456-ubuntu2404",
    }
    expected_tag = (
        "public.ecr.aws/q9t5s3a7/vllm-release-repo:"
        f"def456-{os.uname().machine}-ubuntu2404"
    )
    assert option_values(args, "--tag") == [expected_tag]


def test_local_metadata_args_use_local_overrides() -> None:
    args = run_helper(
        env={
            "VLLM_IMAGE_TAG": "local/test:dev",
            "VLLM_BUILD_COMMIT": "localsha",
            "VLLM_BUILD_PIPELINE": "local-pipeline",
            "VLLM_BUILD_URL": "https://buildkite.example/local",
        },
    )

    assert build_args(args) == {
        "VLLM_BUILD_COMMIT": "localsha",
        "VLLM_BUILD_PIPELINE": "local-pipeline",
        "VLLM_BUILD_URL": "https://buildkite.example/local",
        "VLLM_IMAGE_TAG": "local/test:dev",
    }
    assert option_values(args, "--tag") == ["local/test:dev"]


def test_release_version_lookup_failure_falls_back_to_commit(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    buildkite_agent = fake_bin / "buildkite-agent"
    buildkite_agent.write_text("#!/bin/sh\nexit 1\n")
    buildkite_agent.chmod(0o755)

    args = run_helper(
        "cu129",
        env={
            "BUILDKITE": "1",
            "BUILDKITE_COMMIT": "fallback123",
            "BUILDKITE_PIPELINE_SLUG": "release",
        },
        path=f"{fake_bin}:{os.environ['PATH']}",
    )

    assert build_args(args)["VLLM_IMAGE_TAG"] == ("vllm/vllm-openai:vfallback123-cu129")


def test_vllm_openai_image_embeds_metadata_contract() -> None:
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text()

    for expected in (
        "ARG VLLM_BUILD_COMMIT",
        "ARG VLLM_BUILD_PIPELINE",
        "ARG VLLM_BUILD_URL",
        "ARG VLLM_IMAGE_TAG",
        "VLLM_BUILD_COMMIT=${VLLM_BUILD_COMMIT:-unknown}",
        "VLLM_BUILD_PIPELINE=${VLLM_BUILD_PIPELINE:-local}",
        "VLLM_BUILD_URL=${VLLM_BUILD_URL:-}",
        "VLLM_IMAGE_TAG=${VLLM_IMAGE_TAG:-local/vllm-openai:dev}",
        'ai.vllm.build.commit="${VLLM_BUILD_COMMIT}"',
        'ai.vllm.build.pipeline="${VLLM_BUILD_PIPELINE}"',
        'ai.vllm.build.url="${VLLM_BUILD_URL}"',
        'ai.vllm.image.tag="${VLLM_IMAGE_TAG}"',
    ):
        assert expected in dockerfile


def test_rocm_ci_base_bake_embeds_content_hash_label() -> None:
    bake_file = (REPO_ROOT / "docker" / "docker-bake-rocm.hcl").read_text()

    for expected in (
        'variable "CI_BASE_CONTENT_HASH"',
        'target "ci-base-rocm"',
        'target   = "ci_base"',
        '"vllm.ci_base.content_hash" = CI_BASE_CONTENT_HASH',
    ):
        assert expected in bake_file


def test_rocm_ci_base_metadata_inputs_cover_ci_base_files() -> None:
    ci_bake = (REPO_ROOT / ".buildkite" / "scripts" / "ci-bake-rocm.sh").read_text()

    for expected in (
        "requirements/common.txt",
        "requirements/rocm.txt",
        "requirements/test/rocm.txt",
        "docker/Dockerfile.rocm",
    ):
        assert expected in ci_bake


def test_rocm_ci_uses_owned_canonical_context_without_mutating_source(
    tmp_path: Path,
) -> None:
    repo, regular, executable, link = init_context_test_repo(tmp_path)

    result = run_context_preparation(repo, tmp_path / "script-tmp")

    assert result.returncode == 0, result.stderr
    values = context_result_values(result)
    context = Path(values["CONTEXT"])
    context_index = Path(values["CONTEXT_INDEX"])
    override = Path(values["OVERRIDE"])
    assert stat.S_IMODE(regular.stat().st_mode) == 0o664
    assert stat.S_IMODE(executable.stat().st_mode) == 0o775
    assert stat.S_IMODE((context / regular.name).stat().st_mode) == 0o644
    assert stat.S_IMODE((context / executable.name).stat().st_mode) == 0o755
    assert (context / link.name).is_symlink()
    assert (context / link.name).readlink() == Path(regular.name)
    assert not (context / ".git").exists()
    assert values["CONTEXT_HASH"] != values["SOURCE_HASH"]
    assert values["BAKE_FILE_LAST"] == str(override)
    assert values["BAKE_ALLOW_ARGS"] == f"--allow fs.read={context}"
    assert override.read_text() == (
        f'target "_common-rocm" {{\n  context = "{context}"\n}}\n'
    )
    trailing_newline = context / "trailing-newline\n"
    assert trailing_newline.read_text() == "odd filename\n"
    trailing_newline.write_text("changed odd filename\n")
    assert (
        compute_prepared_context_hash(repo, context, context_index)
        != values["CONTEXT_HASH"]
    )
    trailing_newline.write_text("odd filename\n")
    trailing_newline_link = context / "trailing-newline-link"
    assert trailing_newline_link.readlink() == Path("trailing-newline\n")
    trailing_newline_link.unlink()
    trailing_newline_link.symlink_to("trailing-newline")
    assert (
        compute_prepared_context_hash(repo, context, context_index)
        != values["CONTEXT_HASH"]
    )

    regular.chmod(0o644)
    executable.chmod(0o755)
    canonical_result = run_context_preparation(
        repo,
        tmp_path / "canonical-script-tmp",
    )
    assert canonical_result.returncode == 0, canonical_result.stderr
    canonical_values = context_result_values(canonical_result)
    assert canonical_values["CONTEXT_HASH"] == values["CONTEXT_HASH"]
    assert canonical_values["SOURCE_HASH"] == values["CONTEXT_HASH"]


def test_rocm_non_base_context_preserves_git_version_metadata(tmp_path: Path) -> None:
    repo, regular, executable, _ = init_context_test_repo(
        tmp_path,
        reference_clone=True,
    )
    source_alternates = repo / ".git" / "objects" / "info" / "alternates"
    assert source_alternates.is_file()
    subprocess.run(
        [
            "git",
            "update-ref",
            "refs/remotes/vllm-cache-upstream/main",
            "HEAD~1",
        ],
        cwd=repo,
        check=True,
    )
    source_remote_refs = subprocess.run(
        [
            "git",
            "for-each-ref",
            "--format=%(refname) %(objectname)",
            "refs/remotes",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    result = run_context_preparation(
        repo,
        tmp_path / "script-tmp",
        target="test-rocm-ci-with-artifacts",
    )

    assert result.returncode == 0, result.stderr
    context = Path(context_result_values(result)["CONTEXT"])
    source_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    context_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=context,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    context_status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=context,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    source_description = subprocess.run(
        ["git", "describe", "--tags", "--long", "--dirty"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    context_description = subprocess.run(
        ["git", "describe", "--tags", "--long", "--dirty"],
        cwd=context,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    context_remote_refs = subprocess.run(
        [
            "git",
            "for-each-ref",
            "--format=%(refname) %(objectname)",
            "refs/remotes",
        ],
        cwd=context,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert source_head == context_head
    assert context_status == ""
    assert source_description == context_description
    assert context_remote_refs == source_remote_refs
    assert source_alternates.is_file()
    assert not (context / ".git" / "objects" / "info" / "alternates").exists()
    assert stat.S_IMODE((context / regular.name).stat().st_mode) == 0o644
    assert stat.S_IMODE((context / executable.name).stat().st_mode) == 0o755
    git_file_modes = {
        stat.S_IMODE(path.stat().st_mode)
        for path in (context / ".git").rglob("*")
        if path.is_file()
    }
    assert git_file_modes == {0o644}


def test_rocm_ci_context_tolerates_docker_excluded_retry_debris(
    tmp_path: Path,
) -> None:
    repo, _, _, _ = init_context_test_repo(tmp_path)
    bake_config = repo / "bake-config-build-retry.json"
    wheel_export = repo / "wheel-export" / "stale.whl"
    artifact = repo / "artifacts" / "vllm-rocm-install" / "stale.tar.gz"
    bake_config.write_text("ignored by Docker\n")
    wheel_export.parent.mkdir()
    wheel_export.write_text("ignored by Docker\n")
    (wheel_export.parent / "empty").mkdir()
    artifact.parent.mkdir(parents=True)
    artifact.write_text("ignored by Docker\n")
    (artifact.parent / "empty").mkdir()

    excluded_result = run_context_preparation(
        repo,
        tmp_path / "excluded-script-tmp",
    )

    assert excluded_result.returncode == 0, excluded_result.stderr
    context = Path(context_result_values(excluded_result)["CONTEXT"])
    assert not (context / bake_config.name).exists()
    assert not (context / "wheel-export").exists()
    assert not (context / "artifacts").exists()


def test_rocm_ci_context_rejects_unapproved_untracked_inputs(tmp_path: Path) -> None:
    repo, _, _, _ = init_context_test_repo(tmp_path)

    docker_ignored_input = repo / "docker-output" / "unapproved.bin"
    docker_ignored_input.parent.mkdir()
    docker_ignored_input.write_text("Docker-ignored but not an approved retry output\n")

    docker_ignored_result = run_context_preparation(
        repo,
        tmp_path / "docker-ignored-script-tmp",
    )

    assert docker_ignored_result.returncode != 0
    assert "Untracked files cannot be omitted" in docker_ignored_result.stderr

    docker_ignored_input.unlink()
    docker_ignored_input.parent.rmdir()
    empty_input = repo / "empty-input"
    empty_input.mkdir()

    empty_result = run_context_preparation(
        repo,
        tmp_path / "empty-script-tmp",
    )

    assert empty_result.returncode != 0
    assert "Empty untracked directory cannot be omitted" in empty_result.stderr

    empty_input.rmdir()
    git_ignored_input = repo / "ignored.bin"
    git_ignored_input.write_text("ignored by Git but visible to Docker\n")

    git_ignored_result = run_context_preparation(
        repo,
        tmp_path / "git-ignored-script-tmp",
    )

    assert git_ignored_result.returncode != 0
    assert "Untracked files cannot be omitted" in git_ignored_result.stderr

    git_ignored_input.unlink()
    visible_input = repo / "visible-input.bin"
    visible_input.write_text("untracked build input\n")

    untracked_result = run_context_preparation(
        repo,
        tmp_path / "untracked-script-tmp",
    )

    assert untracked_result.returncode != 0
    assert "Untracked files cannot be omitted" in untracked_result.stderr


@pytest.mark.parametrize(
    ("policy_change", "expected_error"),
    [
        ("missing-rule", "Missing required CI retry-output exclusion"),
        ("negation", "Dockerignore negations"),
        ("dockerfile-specific", "Dockerfile-specific ignore file"),
    ],
)
def test_rocm_ci_context_requires_audited_retry_exclusions(
    tmp_path: Path,
    policy_change: str,
    expected_error: str,
) -> None:
    repo, _, _, _ = init_context_test_repo(tmp_path)
    dockerignore = repo / ".dockerignore"
    if policy_change == "missing-rule":
        dockerignore.write_text(
            dockerignore.read_text().replace("/wheel-export/\n", "")
        )
    elif policy_change == "negation":
        dockerignore.write_text(f"{dockerignore.read_text()}!wheel-export/keep.whl\n")
    else:
        dockerfile_ignore = repo / "docker" / "Dockerfile.rocm.dockerignore"
        dockerfile_ignore.parent.mkdir()
        dockerfile_ignore.write_text("wheel-export/\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-qm", f"{policy_change} policy"],
        cwd=repo,
        check=True,
    )
    stale_wheel = repo / "wheel-export" / "stale.whl"
    stale_wheel.parent.mkdir()
    stale_wheel.write_text("retry output\n")

    result = run_context_preparation(repo, tmp_path / "policy-script-tmp")

    assert result.returncode != 0
    assert expected_error in result.stderr


def test_rocm_ci_context_rejects_dirty_tracked_inputs(tmp_path: Path) -> None:
    repo, regular, _, _ = init_context_test_repo(tmp_path)

    regular.write_text("dirty input\n")
    dirty_result = run_context_preparation(
        repo,
        tmp_path / "dirty-script-tmp",
    )

    assert dirty_result.returncode != 0
    assert "Tracked worktree changes cannot be omitted" in dirty_result.stderr

    subprocess.run(["git", "add", regular.name], cwd=repo, check=True)
    staged_result = run_context_preparation(
        repo,
        tmp_path / "staged-script-tmp",
    )

    assert staged_result.returncode != 0
    assert "Staged changes cannot be omitted" in staged_result.stderr


def test_rocm_bake_calls_allow_the_owned_context(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_calls = tmp_path / "docker-calls"
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        "#!/bin/sh\n"
        "printf 'CALL\\0' >> \"$DOCKER_CALLS\"\n"
        'printf \'%s\\0\' "$@" >> "$DOCKER_CALLS"\n'
        "for arg do\n"
        '  if [ "$arg" = --print ]; then\n'
        "    printf '{}\\n'\n"
        "    break\n"
        "  fi\n"
        "done\n"
    )
    fake_docker.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["DOCKER_CALLS"] = str(docker_calls)
    bake_config = tmp_path / "bake-config.json"
    command = f"""
set -euo pipefail
source {shlex.quote(str(CI_BAKE_ROCM))}
trap - EXIT
BAKE_ALLOW_ARGS=(--allow "fs.read=/tmp/owned context")
BAKE_FILES=(-f base.hcl -f "context override.hcl")
BAKE_TARGETS=(probe)
DEPENDENCY_CACHE_TARGETS=()
BAKE_CONFIG_FILE={shlex.quote(str(bake_config))}
TARGET=probe
BUILDKITE=false
IMAGE_TAG=""
print_bake_config
run_bake
TARGET=ci-base-rocm-ci-with-deps
DEPENDENCY_CACHE_TARGETS=(nixl-rocm-ci)
dependency_cache_ref_for_target() {{ printf 'cache-ref\n'; }}
verify_dependency_cache_ref() {{ return 0; }}
seed_dependency_caches_if_needed
"""

    result = subprocess.run(
        ["bash", "-c", command],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    calls: list[list[str]] = []
    for field in docker_calls.read_bytes().split(b"\0"):
        if field == b"CALL":
            calls.append([])
        elif field:
            calls[-1].append(field.decode())
    assert len(calls) == 3
    common_args = [
        "buildx",
        "bake",
        "--allow",
        "fs.read=/tmp/owned context",
        "-f",
        "base.hcl",
        "-f",
        "context override.hcl",
    ]
    assert calls == [
        [*common_args, "--print", "probe"],
        [*common_args, "--progress", "plain", "probe"],
        [*common_args, "--progress", "plain", "nixl-rocm-ci"],
    ]
