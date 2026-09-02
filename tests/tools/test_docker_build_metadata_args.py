# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import shlex
import subprocess
from pathlib import Path

import regex as re
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / ".buildkite" / "scripts" / "docker-build-metadata-args.sh"
ROCM_CI_BAKE = REPO_ROOT / ".buildkite" / "scripts" / "ci-bake-rocm.sh"
ROCM_IMAGE_SMOKE = REPO_ROOT / ".buildkite" / "scripts" / "rocm" / "smoke-test-image.sh"
ROCM_BASE_WHEEL_CACHE = (
    REPO_ROOT / ".buildkite" / "scripts" / "cache-rocm-base-wheels.sh"
)


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


def test_rocm_ci_base_stage_hash_covers_dependency_graph() -> None:
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile.rocm").read_text()
    ci_bake = ROCM_CI_BAKE.read_text()
    global_args: dict[str, str] = {}
    stage_dependencies: dict[str, set[str]] = {}
    current_stage: str | None = None

    def expand_args(value: str) -> str:
        return re.sub(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}",
            lambda match: global_args.get(match.group(1), match.group(0)),
            value,
        )

    for raw_line in dockerfile.splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        arg_match = re.match(r"ARG\s+([A-Za-z_][A-Za-z0-9_]*)(?:=(.*))?$", line)
        if arg_match and current_stage is None and arg_match.group(2) is not None:
            global_args[arg_match.group(1)] = arg_match.group(2).strip().strip('"')

        from_match = re.match(r"FROM\s+(\S+)(?:\s+AS\s+(\S+))?", line, re.I)
        if from_match:
            parent = expand_args(from_match.group(1))
            current_stage = from_match.group(2)
            if current_stage is not None:
                stage_dependencies[current_stage] = {parent}
            continue

        if current_stage is not None:
            for dependency in re.findall(r"(?:--from=|,from=)([^,\s\\]+)", line):
                stage_dependencies[current_stage].add(expand_args(dependency))

    stage_names = set(stage_dependencies)
    for dependencies in stage_dependencies.values():
        dependencies.intersection_update(stage_names)

    ci_base_closure: set[str] = set()
    pending = ["ci_base"]
    while pending:
        stage = pending.pop()
        if stage in ci_base_closure:
            continue
        ci_base_closure.add(stage)
        pending.extend(stage_dependencies[stage])

    configured_stages = re.search(
        r'^DEFAULT_CI_BASE_DOCKERFILE_STAGES="([^"]+)"$', ci_bake, re.M
    )
    assert configured_stages is not None
    assert set(configured_stages.group(1).split()) == ci_base_closure


def requirement_names(path: Path, seen: set[Path] | None = None) -> set[str]:
    seen = seen or set()
    path = path.resolve()
    if path in seen:
        return set()
    seen.add(path)

    names = set()
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if not line:
            continue
        if line.startswith(("-r ", "--requirement ")):
            names.update(
                requirement_names(path.parent / line.split(maxsplit=1)[1], seen)
            )
        elif not line.startswith("-"):
            names.add(canonicalize_name(Requirement(line).name))
    return names


def test_rocm_test_lock_covers_runtime_requirements() -> None:
    requirements = REPO_ROOT / "requirements"
    lock = requirements / "test" / "rocm.txt"

    assert requirement_names(requirements / "rocm.txt") <= requirement_names(lock)
    locked_requirements = (
        Requirement(line)
        for line in lock.read_text().splitlines()
        if line and not line.startswith((" ", "#", "-"))
    )
    assert all(
        str(requirement.specifier).startswith("==")
        for requirement in locked_requirements
    )


def test_rocm_base_wheel_cache_key_covers_declared_inputs(tmp_path: Path) -> None:
    inputs = []
    for name in ("Dockerfile.rocm_base", ".dockerignore", "rocm.txt", "pipeline.yaml"):
        path = tmp_path / name
        path.write_text(f"{name}: original\n")
        inputs.append(path)

    env = os.environ.copy()
    env.update(
        {
            "ROCM_BASE_DOCKERFILE": str(inputs[0]),
            "ROCM_BASE_CONTENT_FILES": " ".join(map(str, inputs)),
            "ROCM_BASE_PARENT_IMAGE": "rocm/example:mutable",
            "ROCM_BASE_PARENT_DIGEST": f"sha256:{'1' * 64}",
        }
    )

    def cache_key(**overrides: str) -> str:
        command_env = env | overrides
        return subprocess.run(
            ["bash", str(ROCM_BASE_WHEEL_CACHE), "key"],
            check=True,
            cwd=REPO_ROOT,
            env=command_env,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()

    baseline = cache_key()
    for path in inputs:
        original = path.read_text()
        path.write_text(f"{original}changed\n")
        assert cache_key() != baseline
        path.write_text(original)
    for overrides in (
        {"ROCM_BASE_PARENT_DIGEST": f"sha256:{'2' * 64}"},
        {"ROCM_BASE_PLATFORM": "linux/arm64"},
        {"SCCACHE_BUCKET_NAME": "different-bucket"},
        {"ROCM_BASE_WHEEL_TARGET": "different-target"},
    ):
        assert cache_key(**overrides) != baseline


def docker_arg_value(dockerfile: str, name: str) -> str:
    prefix = f"ARG {name}="
    line = next(line for line in dockerfile.splitlines() if line.startswith(prefix))
    return line.removeprefix(prefix).split("#", maxsplit=1)[0].strip().strip('"')


def test_rocm_image_dependency_inputs_are_closed() -> None:
    rocm_base = (REPO_ROOT / "docker" / "Dockerfile.rocm_base").read_text()
    rocm_ci = (REPO_ROOT / "docker" / "Dockerfile.rocm").read_text()
    torchcodec = (REPO_ROOT / "tools" / "install_torchcodec_rocm.sh").read_text()

    for dockerfile, commit_arg_names in (
        (
            rocm_base,
            (
                "TRITON_BRANCH",
                "PYTORCH_BRANCH",
                "PYTORCH_VISION_BRANCH",
                "PYTORCH_AUDIO_BRANCH",
                "FA_BRANCH",
                "AITER_BRANCH",
                "MORI_BRANCH",
                "ROCPROFILER_SDK_COMMIT",
                "ROCM_RUNTIME_COMMIT",
            ),
        ),
        (
            rocm_ci,
            (
                "ROCM_TRITON_KERNELS_COMMIT",
                "NIXL_BRANCH",
                "UCX_BRANCH",
                "LMCACHE_REF",
                "ROCSHMEM_BRANCH",
                "DEEPEP_BRANCH",
                "RDMA_CORE_COMMIT",
            ),
        ),
    ):
        for name in commit_arg_names:
            value = docker_arg_value(dockerfile, name)
            assert len(value) == 40
            int(value, 16)
    for dockerfile, checksum_arg_names in (
        (
            rocm_base,
            (
                "GET_PIP_SHA256",
                "SCCACHE_DOWNLOAD_SHA256",
                "TRITON_LLVM_SHA256",
                "ROCPROFILER_SDK_PR_7796_SHA256",
                "ROCPROFILER_SDK_PR_7924_SHA256",
            ),
        ),
        (rocm_ci, ("UV_DOWNLOAD_SHA256", "RUSTUP_INIT_SHA256")),
    ):
        for name in checksum_arg_names:
            value = docker_arg_value(dockerfile, name)
            assert len(value) == 64
            int(value, 16)

    for expected in (
        "source=requirements/test/rocm.txt",
        "pip wheel --no-build-isolation --no-deps",
        "TRITON_OFFLINE_BUILD=1 TRITON_BUILD_PROTON=OFF",
        "FLASH_ATTENTION_FORCE_BUILD=TRUE",
        "AITER_USE_SYSTEM_TRITON=1",
        "pip check",
    ):
        assert expected in rocm_base
    for expected in (
        "--build-constraints /tmp/rocm-test-reqs.txt",
        "--no-build-isolation",
        "uv pip install --system --no-cache --no-deps --no-build-isolation",
        "python3 -m pip install --no-cache-dir --force-reinstall --no-deps",
        "uv pip install --system --no-deps /lmcache_install/*.whl",
        "uv pip check --system",
    ):
        assert expected in rocm_ci
    assert "revision = e5fada43131d251e9c4786b04263ce98b6767ba5" in rocm_ci
    assert docker_arg_value(rocm_ci, "RUST_TOOLCHAIN_VERSION") == "1.95.0"
    torchcodec_commit = (
        next(
            line
            for line in torchcodec.splitlines()
            if line.startswith("TORCHCODEC_COMMIT=")
        )
        .split(":-", maxsplit=1)[1]
        .split("}", maxsplit=1)[0]
    )
    assert len(torchcodec_commit) == 40
    int(torchcodec_commit, 16)
    assert "TORCHCODEC_WHEEL_CACHE" not in torchcodec
    assert "pip wheel . --no-cache-dir --no-build-isolation --no-deps" in torchcodec


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
