# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / ".buildkite" / "scripts" / "docker-build-metadata-args.sh"
ROCM_CI_BAKE = REPO_ROOT / ".buildkite" / "scripts" / "ci-bake-rocm.sh"
ROCM_IMAGE_SMOKE = REPO_ROOT / ".buildkite" / "scripts" / "rocm" / "smoke-test-image.sh"
ROCM_PROMOTION = (
    REPO_ROOT / ".buildkite" / "scripts" / "rocm" / "promote-stable-images.sh"
)
SMOKE_DIGEST = "sha256:" + "a" * 64
SMOKE_IMAGE = f"rocm/vllm-ci@{SMOKE_DIGEST}"


@pytest.fixture
def promotion_env(tmp_path: Path) -> dict[str, str]:
    """Record external calls; no registry or Git remote is contacted."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for command in ("git", "docker", "buildkite-agent"):
        stub = fake_bin / command
        stub.write_text(
            """#!/bin/bash
printf '%s %s\\n' "${0##*/}" "$*" >> "$CALL_LOG"
case "${0##*/}" in
    git)
        [[ "$1 $2" == 'ls-remote --exit-code' && "$#" == 4 ]] || exit 98
        [[ "$3" == 'https://github.com/vllm-project/vllm.git' ]] || exit 98
        [[ "$4" == refs/heads/main ]] || exit 98
        [[ -n "${REMOTE_TIP:-}" ]] || exit 1
        printf '%s\\trefs/heads/main\\n' "$REMOTE_TIP"
        ;;
    buildkite-agent)
        case "$*" in
            'meta-data get rocm-base-standard-config')
                printf '%s\\n' "${BASE_STANDARD:-1}" ;;
            'meta-data get rocm-ci-base-standard-config')
                printf '%s\\n' "${CI_STANDARD:-1}" ;;
            *) exit 97 ;;
        esac
        ;;
    *) exit 96 ;;
esac
"""
        )
        stub.chmod(0o755)
    return {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CALL_LOG": str(tmp_path / "calls"),
        "BUILDKITE": "true",
        "BUILDKITE_REPO": "https://github.com/vllm-project/vllm.git",
        "BUILDKITE_BRANCH": "main",
        "BUILDKITE_PULL_REQUEST": "false",
        "BUILDKITE_COMMIT": "a" * 40,
        "BUILDKITE_BUILD_ID": "standard-nightly",
        "NIGHTLY": "1",
        "REMOTE_TIP": "a" * 40,
    }


def run_promotion(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(ROCM_PROMOTION)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"NIGHTLY": ""},
        {"NIGHTLY": "0"},
        {"NIGHTLY": "true"},
        {"BUILDKITE": "false"},
        {"BUILDKITE_PULL_REQUEST": "50800"},
        {"BUILDKITE_PULL_REQUEST": ""},
        {"BUILDKITE_BRANCH": "feature", "ROCM_BASE_STABLE_BRANCH": "feature"},
        {
            "BUILDKITE_REPO": "https://github.com/fork/vllm.git",
            "CI_BASE_STABLE_REPO_SLUG": "fork/vllm",
        },
        {"CI_ROCM_DOCKERFILE": "docker/Dockerfile.rock"},
        {"ROCM_BASE_DOCKERFILE": "docker/Dockerfile.rock_base"},
        {"TORCH_NIGHTLY": "1"},
        {"PYTORCH_BRANCH": "experiment"},
        {"NIXL_BRANCH": ""},
        {"ROCM_BASE_PUSH_STABLE_TAG": "0"},
        {"ROCM_BASE_PUSH_STABLE_TAG": ""},
        {"CI_BASE_PUSH_STABLE_TAG": "0"},
        {"ROCM_BASE_IMAGE_REPO": "preview/vllm-dev"},
        {"CI_BASE_IMAGE_TAG": "rocm/vllm-dev:preview"},
    ],
)
def test_rocm_promotion_ineligible_build_never_contacts_registry_or_upstream(
    promotion_env, overrides
) -> None:
    result = run_promotion(promotion_env | overrides)
    assert result.returncode == 0, result.stderr
    assert "Skipping stable ROCm promotion" in result.stdout
    assert not Path(promotion_env["CALL_LOG"]).exists()


@pytest.mark.parametrize("tip", ["b" * 40, "", "invalid"])
def test_rocm_promotion_checks_fixed_upstream_before_loading_candidates(
    promotion_env, tip
) -> None:
    result = run_promotion(promotion_env | {"REMOTE_TIP": tip})
    assert result.returncode == (0 if len(tip) == 40 else 1), result.stderr
    assert Path(promotion_env["CALL_LOG"]).read_text().splitlines() == [
        "git ls-remote --exit-code "
        "https://github.com/vllm-project/vllm.git refs/heads/main"
    ]


@pytest.mark.parametrize("field", ["BASE_STANDARD", "CI_STANDARD"])
@pytest.mark.parametrize("value", ["0", "invalid"])
def test_rocm_promotion_requires_standard_configuration_from_each_producer(
    promotion_env, field, value
) -> None:
    result = run_promotion(promotion_env | {field: value})
    assert result.returncode == (0 if value == "0" else 1), result.stderr
    calls = Path(promotion_env["CALL_LOG"]).read_text().splitlines()
    assert all(
        call.startswith("git ") or call.endswith("-standard-config") for call in calls
    ), calls


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
    agent = fake_bin / "buildkite-agent"
    agent.write_text('#!/bin/sh\ntest "$1 $2" = "meta-data set"\n')
    agent.chmod(0o755)

    marker = tmp_path / "build" / "rocm-smoke-export" / "vllm-smoke-ok"
    marker.parent.mkdir(parents=True)
    marker.write_text(f"{marker_id}\n")
    env = os.environ.copy()
    env.update(
        {
            "BUILDKITE_BUILD_ID": build_id,
            "BUILDKITE": "false",
            "FAKE_DOCKER_CALLED": str(docker_called),
            "PATH": f"{fake_bin}:{env['PATH']}",
            "VLLM_CI_SMOKE_IMAGE": SMOKE_IMAGE,
        }
    )
    env.pop("ROCM_CI_ARTIFACT_ONLY", None)
    return env, marker, docker, docker_called


def test_rocm_smoke_marker_avoids_host_image_pull(tmp_path: Path) -> None:
    env, marker, _, docker_called = prepare_rocm_smoke_test(
        tmp_path,
        marker_id="build-123",
        build_id="build-123",
    )
    proof = marker.with_name("vllm-smoke-image")
    proof.write_text(f"build-123\n{SMOKE_IMAGE}\n{SMOKE_DIGEST}\n")

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
    assert marker.exists()


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
            "VLLM_CI_SMOKE_IMAGE": f"rocm/vllm-ci@sha256:{'b' * 64}",
        }
    )

    subprocess.run(
        ["bash", str(ROCM_IMAGE_SMOKE)],
        check=True,
        cwd=tmp_path,
        env=env,
    )

    assert env["VLLM_CI_SMOKE_IMAGE"] in docker_args.read_text().splitlines()
    assert docker_args.read_text().splitlines()[-3:] == ["-s", "--", "--inside"]
    assert "run_smoke_checks()" in docker_stdin.read_text()
    assert marker.exists()


@pytest.mark.parametrize(
    "proof_lines",
    [
        ["another-build", SMOKE_IMAGE, SMOKE_DIGEST],
        ["build-123", "rocm/vllm-ci:another-image", SMOKE_DIGEST],
        ["build-123", SMOKE_IMAGE, "sha256:" + "b" * 64],
        ["build-123", SMOKE_IMAGE],
    ],
    ids=["wrong-build", "wrong-image", "wrong-digest", "incomplete-proof"],
)
def test_rocm_smoke_rejects_image_proof_mismatch(
    tmp_path: Path, proof_lines: list[str]
) -> None:
    env, marker, _, docker_called = prepare_rocm_smoke_test(
        tmp_path, marker_id="build-123", build_id="build-123"
    )
    marker.with_name("vllm-smoke-image").write_text("\n".join(proof_lines) + "\n")

    result = subprocess.run(
        ["bash", str(ROCM_IMAGE_SMOKE)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "proof does not match" in result.stderr
    assert not docker_called.exists()


def test_rocm_smoke_rejects_proof_without_success_marker(tmp_path: Path) -> None:
    env, marker, _, docker_called = prepare_rocm_smoke_test(
        tmp_path, marker_id="build-123", build_id="build-123"
    )
    marker.with_name("vllm-smoke-image").write_text(
        f"build-123\n{SMOKE_IMAGE}\n{SMOKE_DIGEST}\n"
    )
    marker.unlink()

    result = subprocess.run(
        ["bash", str(ROCM_IMAGE_SMOKE)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "no success marker" in result.stderr
    assert not docker_called.exists()


@pytest.mark.parametrize("has_marker", [False, True])
def test_rocm_smoke_requires_pinned_image_run_without_digest_proof(
    tmp_path: Path, has_marker: bool
) -> None:
    env, marker, docker, docker_called = prepare_rocm_smoke_test(
        tmp_path, marker_id="build-123", build_id="build-123"
    )
    if not has_marker:
        marker.unlink()
    docker.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$FAKE_DOCKER_CALLED"\ncat >/dev/null\n'
    )

    subprocess.run(["bash", str(ROCM_IMAGE_SMOKE)], check=True, cwd=tmp_path, env=env)

    args = docker_called.read_text().splitlines()
    assert args[0] == "run"
    assert SMOKE_IMAGE in args
    assert "--network=none" in args


@pytest.mark.parametrize("marker_id", [None, "another-build", "build-123"])
def test_rocm_bake_export_binds_only_valid_marker_to_emitted_digest(
    tmp_path: Path, marker_id: str | None
) -> None:
    export_dir = tmp_path / "build" / "rocm-smoke-export"
    export_dir.mkdir(parents=True)
    if marker_id is not None:
        (export_dir / "vllm-smoke-ok").write_text(marker_id + "\n")
    metadata = tmp_path / "bake-metadata.json"
    metadata.write_text(
        json.dumps({"test-rocm-ci": {"containerimage.digest": SMOKE_DIGEST}})
    )
    proof = export_dir / "vllm-smoke-image"
    proof.write_text("stale-proof\n")
    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                'source "$1"; TARGET=test-rocm-ci-with-wheel; '
                'BAKE_METADATA_FILE="$2"; verify_rocm_smoke_export'
            ),
            "bash",
            str(ROCM_CI_BAKE),
            str(metadata),
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "BUILDKITE_BUILD_ID": "build-123",
            "IMAGE_TAG": "rocm/vllm-ci:build-123",
        },
        capture_output=True,
        text=True,
    )

    if marker_id == "build-123":
        assert result.returncode == 0, result.stderr
        assert proof.read_text().splitlines() == [
            "build-123",
            "rocm/vllm-ci:build-123",
            SMOKE_DIGEST,
        ]
    else:
        assert result.returncode == 1
        assert not proof.exists()


def test_rocm_bake_records_fresh_metadata_and_smoke_from_same_run(
    tmp_path: Path,
) -> None:
    export_dir = tmp_path / "build" / "rocm-smoke-export"
    export_dir.mkdir(parents=True)
    (export_dir / "vllm-smoke-ok").write_text("stale-marker\n")
    (export_dir / "vllm-smoke-image").write_text("stale-proof\n")
    metadata_dir = tmp_path / "temporary"
    metadata_dir.mkdir()
    (metadata_dir / "bake-metadata.json").write_text("stale-metadata\n")

    result = subprocess.run(
        [
            "bash",
            "-c",
            r"""
source "$1"
SCRIPT_TMP_DIR="$2"
TARGET=test-rocm-ci-with-wheel
BAKE_TARGETS=("$TARGET")
validate_ci_base_output_refs() { return 0; }
docker() {
    [[ "$1 $2" == 'buildx bake' ]] || return 97
    [[ ! -e ./build/rocm-smoke-export/vllm-smoke-ok ]] || return 97
    [[ ! -e ./build/rocm-smoke-export/vllm-smoke-image ]] || return 97
    while (($#)); do
        if [[ "$1" == --metadata-file ]]; then
            [[ ! -e "$2" ]] || return 97
            printf '{"test-rocm-ci":{"containerimage.digest":"%s"}}\n' \
                "$EXPECTED_DIGEST" > "$2"
            printf '%s\n' "$BUILDKITE_BUILD_ID" \
                > ./build/rocm-smoke-export/vllm-smoke-ok
            return 0
        fi
        shift
    done
    return 97
}
run_bake
verify_rocm_smoke_export
""",
            "bash",
            str(ROCM_CI_BAKE),
            str(metadata_dir),
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "BUILDKITE_BUILD_ID": "build-123",
            "IMAGE_TAG": "rocm/vllm-ci:build-123",
            "EXPECTED_DIGEST": SMOKE_DIGEST,
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (export_dir / "vllm-smoke-image").read_text().splitlines() == [
        "build-123",
        "rocm/vllm-ci:build-123",
        SMOKE_DIGEST,
    ]


@pytest.mark.parametrize(
    "metadata_content",
    [None, "invalid-json", "{}", '{"test-rocm-ci":{"containerimage.digest":"bad"}}'],
    ids=["missing", "malformed-json", "missing-target", "invalid-digest"],
)
def test_rocm_bake_without_complete_digest_requires_outer_smoke(
    tmp_path: Path, metadata_content: str | None
) -> None:
    export_dir = tmp_path / "build" / "rocm-smoke-export"
    export_dir.mkdir(parents=True)
    (export_dir / "vllm-smoke-ok").write_text("build-123\n")
    metadata = tmp_path / "bake-metadata.json"
    if metadata_content is not None:
        metadata.write_text(metadata_content)
    proof = export_dir / "vllm-smoke-image"
    proof.write_text("stale-proof\n")

    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                'source "$1"; TARGET=test-rocm-ci-with-wheel; '
                'BAKE_METADATA_FILE="$2"; verify_rocm_smoke_export'
            ),
            "bash",
            str(ROCM_CI_BAKE),
            str(metadata),
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "BUILDKITE_BUILD_ID": "build-123",
            "IMAGE_TAG": "rocm/vllm-ci:build-123",
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "host smoke will verify the pinned image" in result.stdout
    assert not proof.exists()


def test_rocm_smoke_streamed_script_reaches_argument_validation() -> None:
    result = subprocess.run(
        ["bash", "-s", "--", "--unknown-argument"],
        input=ROCM_IMAGE_SMOKE.read_text(),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Usage:" in result.stderr
    assert "unbound variable" not in result.stderr


@pytest.mark.parametrize(
    ("extra_env", "expected_image"),
    [
        (
            {
                "VLLM_CI_SMOKE_IMAGE": "rocm/vllm-ci:explicit",
                "IMAGE_TAG": "rocm/vllm-ci:local",
                "BUILDKITE_COMMIT": "local-commit",
            },
            "rocm/vllm-ci:explicit",
        ),
        ({"IMAGE_TAG": "rocm/vllm-ci:local"}, "rocm/vllm-ci:local"),
        ({"BUILDKITE_COMMIT": "local-commit"}, "rocm/vllm-ci:local-commit"),
        (
            {"BUILDKITE": "true", "IMAGE_TAG": "rocm/vllm-ci:ambient"},
            "rocm/vllm-ci:metadata",
        ),
    ],
    ids=["explicit-override", "local-image-tag", "local-commit", "ci-metadata"],
)
def test_rocm_smoke_image_selection_preserves_local_and_ci_contracts(
    tmp_path: Path, extra_env: dict[str, str], expected_image: str
) -> None:
    env, marker, docker, docker_called = prepare_rocm_smoke_test(
        tmp_path, marker_id="build-123", build_id="build-123"
    )
    marker.unlink()
    for name in ("VLLM_CI_SMOKE_IMAGE", "IMAGE_TAG", "BUILDKITE_COMMIT"):
        env.pop(name, None)
    env.update(extra_env)
    env.update({"EXPECTED_SMOKE_IMAGE": expected_image, "SMOKE_DIGEST": SMOKE_DIGEST})
    agent = docker.with_name("buildkite-agent")
    agent.write_text(
        "#!/bin/sh\n"
        'case "$1 $2 $3" in\n'
        '  "meta-data get rocm-ci-image-smoke-required") printf "1\\n";;\n'
        '  "meta-data get rocm-ci-image-smoke-ref") '
        'printf "%s\\n" "$EXPECTED_SMOKE_IMAGE";;\n'
        '  "meta-data set "*) :;;\n'
        "  *) exit 97;;\n"
        "esac\n"
    )
    docker.write_text(
        "#!/bin/sh\n"
        'if [ "$1 $2 $3" = "buildx imagetools inspect" ]; then\n'
        '  test "$4" = "$EXPECTED_SMOKE_IMAGE" || exit 97\n'
        '  printf "Digest: %s\\n" "$SMOKE_DIGEST"\n'
        'elif [ "$1" = run ]; then\n'
        '  printf "%s\\n" "$@" > "$FAKE_DOCKER_CALLED"\n'
        "  cat >/dev/null\n"
        "else exit 97; fi\n"
    )

    subprocess.run(["bash", str(ROCM_IMAGE_SMOKE)], check=True, cwd=tmp_path, env=env)

    assert SMOKE_IMAGE in docker_called.read_text().splitlines()


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


@pytest.mark.parametrize("helper", ["refresh", "validate"])
@pytest.mark.parametrize(
    "scenario", ["equivalent", "wrong-hash", "wrong-parent", "build-race"]
)
def test_rocm_ci_base_shared_content_races_preserve_build_identity(
    tmp_path: Path, helper: str, scenario: str
) -> None:
    """Concurrent equivalent cache writes cannot invalidate our pinned build."""
    result = subprocess.run(
        [
            "bash",
            "-c",
            r"""
source "$1"
TARGET=ci-base-rocm-ci
CI_BASE_IMAGE_TAG_BUILD_REF=ci:build
CI_BASE_IMAGE_TAG_CONTENT_REF=ci:content
IMAGE_TAG=ci:build
CI_BASE_IMAGE_TAG=ci:content
unexpected() { printf '%s\n' "$*" > unexpected-command; return 97; }
docker() {
    case "$1 ${2:-} ${3:-}" in
        'manifest inspect ci:build'|'manifest inspect ci:content') return 0 ;;
        'buildx imagetools create')
            [[ "$4 $5" == '--prefer-index=false -t' \
                && "$7" == "ci:content@$OWN_DIGEST" ]] || unexpected "$@"
            ;;
        'buildx imagetools inspect')
            case "$4" in
                ci:build|ci:content|"ci:build@$OWN_DIGEST"|"ci:content@$OWN_DIGEST"|\
                "ci:build@$OTHER_DIGEST"|"ci:content@$OTHER_DIGEST") ;;
                *) unexpected "$@"; return 97 ;;
            esac
            if [[ "${5:-}" == --format ]]; then
                local hash="$CI_BASE_CONTENT_HASH" parent="${BASE_IMAGE##*@}"
                if [[ "$4" == "ci:content@$OTHER_DIGEST" ]]; then
                    [[ "$SCENARIO" != wrong-hash ]] || hash="$OTHER_HASH"
                    [[ "$SCENARIO" != wrong-parent ]] || parent="$OTHER_DIGEST"
                fi
                printf '%s|3|%s|%s\n' "$hash" "$CI_BASE_CONTENT_HASH" "$parent"
            elif [[ "$4" == ci:build ]]; then
                local count=0 digest="$OWN_DIGEST"
                [[ ! -f inspections ]] || read -r count < inspections
                count=$((count + 1)); printf '%s\n' "$count" > inspections
                if [[ "$SCENARIO" == build-race ]] \
                    && { [[ "$HELPER" == refresh ]] || ((count > 1)); }; then
                    digest="$OTHER_DIGEST"
                fi
                printf 'Digest: %s\n' "$digest"
            else
                printf 'Digest: %s\n' "$OTHER_DIGEST"
            fi
            ;;
        *) unexpected "$@" ;;
    esac
}
if [[ "$HELPER" == refresh ]]; then
    refresh_ci_base_tags_from_ref "ci:content@$OWN_DIGEST"
else
    validate_ci_base_output_refs
fi
""",
            "bash",
            str(ROCM_CI_BAKE),
        ],
        cwd=tmp_path,
        env={
            "PATH": os.environ["PATH"],
            "HELPER": helper,
            "SCENARIO": scenario,
            "OWN_DIGEST": "sha256:" + "a" * 64,
            "OTHER_DIGEST": "sha256:" + "b" * 64,
            "CI_BASE_CONTENT_HASH": "c" * 64,
            "OTHER_HASH": "d" * 64,
            "BASE_IMAGE": "base@sha256:" + "e" * 64,
            "CI_BASE_LABEL_ATTEMPTS": "1",
            "CI_BASE_LABEL_RETRY_DELAY": "0",
            "ROCM_REGISTRY_PROBE_ATTEMPTS": "1",
        },
        capture_output=True,
        text=True,
    )
    assert not (tmp_path / "unexpected-command").exists(), result.stderr
    assert result.returncode == (0 if scenario == "equivalent" else 2), result.stderr
