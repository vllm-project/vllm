# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
    import torch

    from vllm.platforms import current_platform

    dockerfile_names = ["Dockerfile", "Dockerfile.cpu"]
    # CPU jobs can reuse ROCm artifacts, which omit the XPU Dockerfile.
    if not current_platform.is_rocm() and (
        not current_platform.is_cpu() or torch.version.hip is None
    ):
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
        assert "bash tools/build_rust.sh" in exact_version_stage


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


def test_rocm_ci_smoke_runs_in_shared_buildkit_graph() -> None:
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile.rocm").read_text()
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


def run_rocm_map_build(
    tmp_path: Path, *, target: str = "test-rocm-ci", **settings: str
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    # Exercise real orchestration/errexit; replace expensive build preparation
    # and external services. Cached exports launch another copy of this harness.
    harness = tmp_path / "bake.sh"
    harness.write_text(
        r"""
source "$1"
for fn in configure_ci_base_write_scope print_header validate_inputs load_ci_hcl \
    init_bake_files compute_ci_base_hash_if_needed configure_ci_base_image_refs \
    prepare_git_cache_metadata prepare_ci_build_context configure_custom_rocm_stages \
    extract_dependency_pins write_rocm_build_arg_override \
    compute_dependency_cache_keys \
    write_ci_base_label_override compute_rocm_csrc_content_hash_if_needed \
    compute_rocm_rust_content_hash_if_needed write_rocm_cache_override \
    write_build_context_override resolve_ci_base_dependency_targets print_bake_config \
    seed_dependency_caches_if_needed promote_stable_ci_base_tag \
    publish_ci_base_handoff_ref
do
    eval "$fn() { :; }"
done
init_config() {
    TARGET="$1"
    BAKE_TARGETS=("$TARGET")
    BAKE_ALLOW_ARGS=(--allow 'fs.read=/owned context')
    BAKE_FILES=(-f 'resolved args and caches.hcl' -f 'owned context.hcl')
    CI_HCL_PATH=ci.hcl VLLM_BAKE_FILE=bake.hcl BUILDER_NAME=builder
}
setup_builder() {
    if [[ "$TARGET" == kernel-symbol-map-rocm && "$FAILURE" == setup ]]; then
        false
    fi
    echo setup >> events
}
remote_image_exists() { [[ "$CACHED" == 1 ]]; }
get_remote_image_label() { echo "$BUILDKITE_COMMIT"; }
bash() { command bash "$0" "$@"; }
docker() {
    printf '%q ' "$@" >> calls
    printf '\n' >> calls
    if [[ "${*: -1}" == kernel-symbol-map-rocm ]]; then
        [[ ! -e kernel-symbol-map-rocm ]] || return 90
        [[ "$FAILURE" != export ]] || return 41
        [[ "$FAILURE" != missing ]] || return 0
        mkdir kernel-symbol-map-rocm
        if [[ "$FAILURE" == empty ]]; then
            touch kernel-symbol-map-rocm/kernel_symbol_map.rocm.json.gz
        else
            echo current > kernel-symbol-map-rocm/kernel_symbol_map.rocm.json.gz
        fi
        [[ "$FAILURE" != partial ]] || return 42
    else
        [[ "$FAILURE" != primary ]] || return 43
    fi
}
buildkite-agent() {
    printf '%q ' "$@" >> "$TEST_ROOT/calls"
    printf '\n' >> "$TEST_ROOT/calls"
    [[ "$(cat kernel_symbol_map.rocm.json.gz)" == current ]] || return 91
    [[ "$FAILURE" != upload ]] || return 44
}
main "$2"
"""
    )
    export = tmp_path / "kernel-symbol-map-rocm"
    export.mkdir()
    (export / "kernel_symbol_map.rocm.json.gz").write_text("stale")
    (tmp_path / "calls").touch()
    env = {
        "PATH": os.environ["PATH"],
        "TEST_ROOT": str(tmp_path),
        "VLLM_KERNEL_SYMBOL_MAP": "1",
        "BUILDKITE": "true",
        "BUILDKITE_COMMIT": "current-commit",
        "IMAGE_TAG": "test:current",
        "CACHED": "0",
        "FAILURE": "",
        **settings,
    }
    result = subprocess.run(
        ["bash", str(harness), str(ROCM_CI_BAKE), target],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    return result, [
        shlex.split(line) for line in (tmp_path / "calls").read_text().splitlines()
    ]


@pytest.mark.parametrize("cached", ["0", "1"])
@pytest.mark.parametrize(
    "failure", ["", "export", "partial", "missing", "empty", "upload"]
)
def test_rocm_optional_map_failures_preserve_image_success(
    tmp_path: Path, cached: str, failure: str
) -> None:
    result, calls = run_rocm_map_build(tmp_path, CACHED=cached, FAILURE=failure)
    assert result.returncode == 0, result.stderr
    targets = [call[-1] for call in calls if call[:2] == ["buildx", "bake"]]
    assert targets == (
        ["test-rocm-ci", "kernel-symbol-map-rocm"]
        if cached == "0"
        else ["kernel-symbol-map-rocm"]
    )
    if cached == "0":
        assert calls[0][:-1] == calls[1][:-1]
    uploads = [call for call in calls if call[:2] == ["artifact", "upload"]]
    assert uploads == (
        [["artifact", "upload", "kernel_symbol_map.rocm.json.gz"]]
        if failure in ("", "upload")
        else []
    )
    assert ("continuing without it" in result.stderr) == bool(failure)


def test_rocm_cached_map_setup_stops_on_failure_without_failing_image(
    tmp_path: Path,
) -> None:
    result, calls = run_rocm_map_build(tmp_path, CACHED="1", FAILURE="setup")
    assert result.returncode == 0, result.stderr
    assert "continuing without it" in result.stderr
    assert not calls
    assert not (tmp_path / "events").exists()


@pytest.mark.parametrize(
    ("target", "failure", "expected_status"),
    [
        ("test-rocm-ci", "primary", 43),
        ("kernel-symbol-map-rocm", "export", 41),
        ("kernel-symbol-map-rocm", "upload", 44),
        ("smoke-test-rocm-ci", "", 1),
        ("export-wheel-rocm", "", 1),
    ],
)
def test_rocm_required_build_failures_remain_fatal(
    tmp_path: Path, target: str, failure: str, expected_status: int
) -> None:
    result, calls = run_rocm_map_build(tmp_path, target=target, FAILURE=failure)
    assert result.returncode == expected_status, result.stderr
    assert [call[-1] for call in calls if call[:2] == ["buildx", "bake"]] == [target]
    assert "continuing without it" not in result.stderr


@pytest.mark.parametrize("cached", ["0", "1"])
@pytest.mark.parametrize("setting", ["VLLM_KERNEL_SYMBOL_MAP", "BAKE_PRINT_ONLY"])
def test_rocm_disabled_or_print_only_build_skips_map_export(
    tmp_path: Path, cached: str, setting: str
) -> None:
    settings = {setting: "0" if setting == "VLLM_KERNEL_SYMBOL_MAP" else "1"}
    result, calls = run_rocm_map_build(tmp_path, CACHED=cached, **settings)
    assert result.returncode == 0, result.stderr
    expected = (
        ["test-rocm-ci"]
        if cached == "0" and setting == "VLLM_KERNEL_SYMBOL_MAP"
        else []
    )
    assert [call[-1] for call in calls] == expected
    assert (
        tmp_path / "kernel-symbol-map-rocm/kernel_symbol_map.rocm.json.gz"
    ).read_text() == "stale"
