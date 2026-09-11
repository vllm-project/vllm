# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import os
import shlex
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / ".buildkite" / "scripts" / "docker-build-metadata-args.sh"
ROCM_CI_BAKE = REPO_ROOT / ".buildkite" / "scripts" / "ci-bake-rocm.sh"
ROCM_IMAGE_SMOKE = REPO_ROOT / ".buildkite" / "scripts" / "rocm" / "smoke-test-image.sh"


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


def test_rust_build_cache_excludes_git_metadata() -> None:
    from vllm.platforms import current_platform

    dockerfile_names = ["Dockerfile", "Dockerfile.cpu"]
    if not current_platform.is_rocm():
        dockerfile_names.append("Dockerfile.xpu")
    for name in dockerfile_names:
        dockerfile = (REPO_ROOT / "docker" / name).read_text()
        cached_stage, exact_version_stage = dockerfile.split(
            "FROM rust-build-cache AS rust-build", maxsplit=1
        )
        exact_version_stage = exact_version_stage.split("\nFROM ", maxsplit=1)[0]
        cached_run = cached_stage.rsplit("RUN ", maxsplit=1)[1]

        assert 'SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0+docker.cache"' in cached_run
        assert "source=.git,target=.git" not in cached_run
        assert "source=.git,target=.git" in exact_version_stage
        assert 'SETUPTOOLS_SCM_PRETEND_METADATA="{dirty=false}"' in exact_version_stage
        assert "bash build_rust.sh" in exact_version_stage


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
    ci_bake = ROCM_CI_BAKE.read_text()

    for expected in (
        "requirements/common.txt",
        "requirements/rocm.txt",
        "requirements/test/rocm.txt",
        "docker/Dockerfile.rocm",
    ):
        assert expected in ci_bake


@pytest.mark.parametrize("dockerfile_name", ["Dockerfile.rocm", "Dockerfile.rock"])
def test_rocm_ci_smoke_runs_in_shared_buildkit_graph(dockerfile_name: str) -> None:
    dockerfile = (REPO_ROOT / "docker" / dockerfile_name).read_text()
    ci_hcl = (REPO_ROOT / "docker" / "ci-rocm.hcl").read_text()
    full_image_group = ci_hcl.split('group "test-rocm-ci-with-wheel"', maxsplit=1)[
        1
    ].split("}", maxsplit=1)[0]

    for expected in (
        "FROM test AS test_smoke",
        "smoke-test-image.sh --inside",
        "FROM scratch AS export_test_smoke",
        'target "smoke-test-rocm-ci"',
        'target     = "export_test_smoke"',
        'output     = ["type=local,dest=./build/rocm-smoke-export"]',
    ):
        assert expected in dockerfile or expected in ci_hcl
    assert '"smoke-test-rocm-ci"' in full_image_group
    assert 'target "smoke-test-rocm-ci"' in ROCM_CI_BAKE.read_text()


def prepare_rocm_smoke_test(
    tmp_path: Path,
    *,
    marker_id: str,
    build_id: str,
) -> tuple[dict[str, str], Path, Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_called = tmp_path / "docker-called"
    docker = fake_bin / "docker"
    docker.write_text('#!/bin/sh\ntouch "$FAKE_DOCKER_CALLED"\nexit 99\n')
    docker.chmod(0o755)

    marker = tmp_path / "build" / "rocm-smoke-export" / "vllm-smoke-ok"
    marker.parent.mkdir(parents=True)
    marker.write_text(f"{marker_id}\n")
    env = os.environ.copy()
    env.update(
        {
            "BUILDKITE_BUILD_ID": build_id,
            "FAKE_DOCKER_CALLED": str(docker_called),
            "PATH": f"{fake_bin}:{env['PATH']}",
        }
    )
    env.pop("ROCM_CI_ARTIFACT_ONLY", None)
    env.pop("VLLM_CI_SMOKE_IMAGE", None)
    return env, marker, docker, docker_called


def test_rocm_smoke_marker_avoids_host_image_pull(tmp_path: Path) -> None:
    env, marker, _, docker_called = prepare_rocm_smoke_test(
        tmp_path,
        marker_id="build-123",
        build_id="build-123",
    )

    result = subprocess.run(
        ["bash", str(ROCM_IMAGE_SMOKE)],
        check=True,
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        text=True,
    )

    assert "verified inside BuildKit" in result.stdout
    assert not docker_called.exists()
    assert not marker.exists()


def test_rocm_smoke_rejects_marker_from_another_build(tmp_path: Path) -> None:
    env, marker, _, docker_called = prepare_rocm_smoke_test(
        tmp_path,
        marker_id="previous-build",
        build_id="current-build",
    )

    result = subprocess.run(
        ["bash", str(ROCM_IMAGE_SMOKE)],
        check=False,
        cwd=tmp_path,
        env=env,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 1
    assert "previous-build, not current-build" in result.stderr
    assert not docker_called.exists()
    assert marker.exists()


def test_rocm_smoke_override_streams_current_checks_to_docker(
    tmp_path: Path,
) -> None:
    env, marker, docker, _ = prepare_rocm_smoke_test(
        tmp_path,
        marker_id="build-123",
        build_id="build-123",
    )
    docker_args = tmp_path / "docker-args"
    docker_stdin = tmp_path / "docker-stdin"
    docker.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$FAKE_DOCKER_ARGS"\n'
        'cat > "$FAKE_DOCKER_STDIN"\n'
    )
    env.update(
        {
            "FAKE_DOCKER_ARGS": str(docker_args),
            "FAKE_DOCKER_STDIN": str(docker_stdin),
            "IMAGE_TAG": "rocm/vllm-ci:built",
            "VLLM_CI_SMOKE_IMAGE": "rocm/vllm-ci:override",
        }
    )

    subprocess.run(
        ["bash", str(ROCM_IMAGE_SMOKE)],
        check=True,
        cwd=tmp_path,
        env=env,
    )

    assert "rocm/vllm-ci:override" in docker_args.read_text().splitlines()
    assert docker_args.read_text().splitlines()[-3:] == ["-s", "--", "--inside"]
    assert "run_smoke_checks()" in docker_stdin.read_text()
    assert marker.exists()


def test_rocm_git_fetch_disables_automatic_maintenance(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    git = fake_bin / "git"
    git.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    git.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; git_fetch_with_timeout --quiet origin HEAD',
            "bash",
            str(ROCM_CI_BAKE),
        ],
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        text=True,
    )

    assert result.stdout.splitlines() == [
        "fetch",
        "--no-auto-maintenance",
        "--quiet",
        "origin",
        "HEAD",
    ]


@pytest.mark.parametrize(
    ("base_stack", "stack", "use_rock", "rename"),
    [
        pytest.param(None, None, "0", False, id="default"),
        pytest.param("rocm", "rocm", "0", False, id="explicit-defaults"),
        pytest.param("rock", "rock", "0", False, id="custom-pair"),
        pytest.param(None, None, "1", False, id="rock-shorthand"),
        pytest.param("rock", None, "0", False, id="custom-base-only"),
        pytest.param(None, "rock", "0", False, id="custom-final-only"),
        pytest.param("rocm", "rocm", "1", False, id="explicit-overrides-shorthand"),
        pytest.param("rock", "rock", "0", True, id="renamed-rock"),
        pytest.param("rocm", "rocm", "0", True, id="renamed-rocm"),
    ],
)
def test_amd_stack_selection_preserves_handoff_and_protects_stable_images(
    tmp_path: Path,
    base_stack: str | None,
    stack: str | None,
    use_rock: str,
    rename: bool,
) -> None:
    """Any custom stack must preserve handoffs without promoting stable images."""
    selection = {"VLLM_USE_ROCK": use_rock}
    fallback_stack = "rock" if use_rock == "1" else "rocm"
    base_dockerfile = f"docker/Dockerfile.{base_stack or fallback_stack}_base"
    dockerfile = f"docker/Dockerfile.{stack or fallback_stack}"
    if rename:
        (tmp_path / "docker").symlink_to(REPO_ROOT / "docker", target_is_directory=True)
        for name, source in (("base", base_dockerfile), ("final", dockerfile)):
            (tmp_path / name).write_text((REPO_ROOT / source).read_text())
        base_dockerfile, dockerfile = "base", "final"
    if base_stack is not None:
        selection["CI_ROCM_DOCKERFILE_BASE"] = base_dockerfile
    if stack is not None:
        selection["CI_ROCM_DOCKERFILE"] = dockerfile
    custom = (base_dockerfile, dockerfile) != (
        "docker/Dockerfile.rocm_base",
        "docker/Dockerfile.rocm",
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
source "$1/.buildkite/scripts/ci-bake-rocm.sh"
init_config ci-base-rocm-ci-with-deps
validate_inputs
init_bake_files
configure_ci_base_image_refs
printf 'dockerfile=%s\n' "$CI_BASE_DOCKERFILE"
printf 'bake_files=%s\n' "${BAKE_FILES[*]}"
printf 'handoff=%s\n' "$CI_BASE_IMAGE_TAG_BUILD_REF"
if wants_stable_ci_base_tag; then echo ci_stable=yes; else echo ci_stable=no; fi
if using_custom_rocm_dockerfiles; then
    resolve_ci_base_dependency_targets
    printf 'targets=%s\n' "${BAKE_TARGETS[*]}"
fi
source "$1/.buildkite/scripts/rocm/refresh-base-image.sh"
configure_rocm_base_layer_cache
printf 'base=%s\ncache=%s\n' "$DOCKERFILE" "$ROCM_BASE_LAYER_CACHE_REF"
if should_push_stable_tag; then echo base_stable=yes; else echo base_stable=no; fi
""",
            "bash",
            str(REPO_ROOT),
        ],
        env={
            "PATH": os.environ["PATH"],
            **selection,
            "BUILDKITE": "true",
            "BUILDKITE_BUILD_ID": "test-build",
            "BUILDKITE_COMMIT": "a" * 40,
            "BUILDKITE_BRANCH": "main",
            "BUILDKITE_REPO": "https://github.com/vllm-project/vllm.git",
            "NIGHTLY": "1",
            "ROCM_BASE_PUSH_STABLE_TAG": "1",
            "CI_BASE_PUSH_STABLE_TAG": "1",
            "CI_BASE_CONTENT_HASH": "b" * 64,
            "BASE_IMAGE": "rocm/vllm-dev:base@sha256:" + "c" * 64,
        },
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path if rename else REPO_ROOT,
    )
    lines = result.stdout.splitlines()
    stable = "no" if custom else "yes"
    assert f"dockerfile={dockerfile}" in lines
    assert f"base={base_dockerfile}" in lines
    assert "handoff=rocm/vllm-dev:ci_base-build-test-build" in lines
    cache_prefix = "rocm-base"
    if custom:
        selection_hash = hashlib.sha256(
            f"{base_dockerfile}\n{dockerfile}\n".encode()
        ).hexdigest()[:12]
        cache_prefix += f"-{selection_hash}"
    assert f"cache=rocm/vllm-ci-cache:{cache_prefix}-main" in lines
    assert f"ci_stable={stable}" in lines
    assert f"base_stable={stable}" in lines
    bake_files = next(line for line in lines if line.startswith("bake_files="))
    assert ("*.cache-to=" in bake_files) == custom
    if custom:
        assert "targets=ci-base-rocm-ci" in lines


@pytest.mark.parametrize(
    "selection",
    [{"VLLM_USE_ROCK": "1"}, {"CI_ROCM_DOCKERFILE": "docker/Dockerfile.rock"}],
    ids=["rock-shorthand", "custom-final"],
)
@pytest.mark.parametrize(
    ("target", "base_image", "missing"),
    [
        ("ci-base-rocm-ci-with-deps", "", "BASE_IMAGE"),
        ("test-rocm-ci-with-wheel", "rock-base", "CI_BASE_IMAGE"),
    ],
)
def test_custom_rocm_build_requires_selected_base_handoffs(
    selection: dict[str, str], target: str, base_image: str, missing: str
) -> None:
    result = subprocess.run(
        ["bash", str(ROCM_CI_BAKE), target],
        env={
            "PATH": os.environ["PATH"],
            **selection,
            "BASE_IMAGE": base_image,
        },
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode != 0
    assert f"require {missing} from the selected" in result.stderr


@pytest.mark.parametrize("missing_file", [True, False], ids=["file", "stage"])
def test_custom_rocm_dockerfile_must_support_ci_targets(
    tmp_path: Path, missing_file: bool
) -> None:
    """Reject unusable Dockerfiles before fetching images or starting a build."""
    (tmp_path / "docker").symlink_to(REPO_ROOT / "docker", target_is_directory=True)
    if not missing_file:
        (tmp_path / "Dockerfile.custom").write_text("FROM scratch AS ci_base\n")
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; init_config ci-base-rocm-ci-with-deps; validate_inputs',
            "bash",
            str(ROCM_CI_BAKE),
        ],
        env={
            "PATH": os.environ["PATH"],
            "CI_ROCM_DOCKERFILE": "Dockerfile.custom",
            "BASE_IMAGE": "selected-base",
        },
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    if missing_file:
        assert "ROCm Dockerfile not found: Dockerfile.custom" in result.stderr
    else:
        assert "is missing required stage: test" in result.stderr


def test_custom_rocm_content_hash_covers_helper_stages(tmp_path: Path) -> None:
    """A custom helper stage and its build arguments must invalidate cached images."""
    (tmp_path / "Dockerfile.custom").write_text(
        "from scratch as extra_helper\n"
        "arg EXTRA_VERSION=one\n"
        'RUN echo "$EXTRA_VERSION"\n'
        "FROM extra_helper AS ci_base\n"
        "FROM ci_base AS test\n"
        "FROM scratch AS export_vllm\n"
        "FROM scratch AS export_test_smoke\n"
        "FROM scratch AS csrc-build\n"
        "FROM scratch AS rust-build\n"
    )
    (tmp_path / "input.txt").write_text("unchanged input\n")
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
set -euo pipefail
git init --quiet
git add Dockerfile.custom input.txt
tree=$(git write-tree)
commit=$(printf fixture | git -c user.name=Test -c user.email=test@example.com \\
    commit-tree "$tree")
git update-ref HEAD "$commit"
source "$1"
init_config ci-base-rocm-ci-with-deps
prepare_ci_build_context
configure_custom_rocm_stages
printf 'hash=%s\n' "$(compute_ci_base_content_hash)"
EXTRA_VERSION=two
printf 'hash=%s\n' "$(compute_ci_base_content_hash)"
sed -i 's/RUN echo/RUN printf/' "$ROCM_BUILD_CONTEXT_ROOT/Dockerfile.custom"
printf 'hash=%s\n' "$(compute_ci_base_content_hash)"
""",
            "bash",
            str(ROCM_CI_BAKE),
        ],
        env={
            "PATH": os.environ["PATH"],
            "BUILDKITE": "true",
            "CI_ROCM_DOCKERFILE": "Dockerfile.custom",
            "BASE_IMAGE": "selected-base",
            "CI_BASE_CONTENT_FILES": "input.txt",
        },
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    hashes = [line for line in result.stdout.splitlines() if line.startswith("hash=")]
    assert len(hashes) == len(set(hashes)) == 3


@pytest.mark.parametrize(
    ("selection", "skip", "error"),
    [
        ({"VLLM_USE_ROCK": "invalid"}, "0", "VLLM_USE_ROCK must be 0 or 1"),
        ({"VLLM_USE_ROCK": "1"}, "1", "require ROCM_BASE_REFRESH_SKIP=0"),
        (
            {"CI_ROCM_DOCKERFILE_BASE": "docker/Dockerfile.rock_base"},
            "1",
            "require ROCM_BASE_REFRESH_SKIP=0",
        ),
        (
            {"CI_ROCM_DOCKERFILE": "docker/Dockerfile.rock"},
            "1",
            "require ROCM_BASE_REFRESH_SKIP=0",
        ),
    ],
)
def test_custom_rocm_base_selection_rejects_unsafe_fallbacks(
    selection: dict[str, str], skip: str, error: str
) -> None:
    result = subprocess.run(
        ["bash", str(REPO_ROOT / ".buildkite/scripts/rocm/refresh-base-image.sh")],
        env={
            "PATH": os.environ["PATH"],
            **selection,
            "ROCM_BASE_REFRESH_SKIP": skip,
        },
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 2
    assert error in result.stderr
