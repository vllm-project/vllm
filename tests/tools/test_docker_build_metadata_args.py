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
ROCM_REFRESH_BASE = (
    REPO_ROOT / ".buildkite" / "scripts" / "rocm" / "refresh-base-image.sh"
)
ROCM_RELEASE_PIPELINE = REPO_ROOT / ".buildkite" / "release-pipeline.yaml"
ROCM_RELEASE_BUILD = REPO_ROOT / ".buildkite" / "scripts" / "build-rocm-base-wheels.sh"
ROCM_BUILD_LOCK = REPO_ROOT / "requirements" / "build" / "rocm.txt"
ROCM_TEST_INPUT = REPO_ROOT / "requirements" / "test" / "rocm.in"
TORCHCODEC_INSTALLER = REPO_ROOT / "tools" / "install_torchcodec_rocm.sh"


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


def dockerfile_stage_dependencies(dockerfile: str) -> dict[str, set[str]]:
    global_args: dict[str, str] = {}
    stage_dependencies: dict[str, set[str]] = {}
    current_stage: str | None = None

    def expand_args(value: str) -> str:
        return re.sub(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)",
            lambda match: global_args.get(
                match.group(1) or match.group(2), match.group(0)
            ),
            value,
        )

    for raw_line in dockerfile.splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        arg_match = re.match(r"ARG\s+([A-Za-z_][A-Za-z0-9_]*)(?:=(.*))?$", line, re.I)
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
            for dependency in re.findall(r"(?:--from=|,from=)([^,\s\\]+)", line, re.I):
                stage_dependencies[current_stage].add(expand_args(dependency))

    stage_names = set(stage_dependencies)
    for dependencies in stage_dependencies.values():
        dependencies.intersection_update(stage_names)
    return stage_dependencies


def dockerfile_stage_closure(
    stage_dependencies: dict[str, set[str]], root: str
) -> set[str]:
    closure: set[str] = set()
    pending = [root]
    while pending:
        stage = pending.pop()
        if stage in closure:
            continue
        closure.add(stage)
        pending.extend(stage_dependencies[stage])
    return closure


def test_rocm_stage_hashes_cover_dependency_graphs() -> None:
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile.rocm").read_text()
    ci_bake = ROCM_CI_BAKE.read_text()
    stage_dependencies = dockerfile_stage_dependencies(dockerfile)

    for variable, root in (
        ("DEFAULT_CI_BASE_DOCKERFILE_STAGES", "ci_base"),
        ("DEFAULT_ROCM_CSRC_DOCKERFILE_STAGES", "csrc-build"),
        ("DEFAULT_ROCM_RUST_DOCKERFILE_STAGES", "rust-build"),
    ):
        configured_stages = re.search(rf'^{variable}="([^"]+)"$', ci_bake, re.M)
        assert configured_stages is not None
        assert set(configured_stages.group(1).split()) == dockerfile_stage_closure(
            stage_dependencies, root
        )


def test_dockerfile_stage_parser_expands_both_arg_forms() -> None:
    dockerfile = """\
ARG ROOT=scratch
ARG SHARED=shared
from ${ROOT} AS shared
RUN echo shared
FROM scratch AS braced
COPY --from=${SHARED} /in /out
FROM scratch AS unbraced
RUN --mount=type=bind,from=$SHARED,target=/in echo unbraced
"""
    dependencies = dockerfile_stage_dependencies(dockerfile)

    assert dependencies["braced"] == {"shared"}
    assert dependencies["unbraced"] == {"shared"}


def test_rocm_stage_hasher_treats_from_case_insensitively(tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "from scratch AS first\n"
        "RUN echo first\n"
        "FrOm scratch as second\n"
        "RUN echo second\n"
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; hash_dockerfile_stages "$2" first',
            "bash",
            str(ROCM_CI_BAKE),
            str(dockerfile),
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )

    assert result.stdout.splitlines() == [
        "from scratch AS first",
        "RUN echo first",
    ]


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


def locked_requirements(path: Path) -> dict[str, Requirement]:
    requirements = (
        Requirement(line)
        for line in path.read_text().splitlines()
        if line and not line.startswith((" ", "#", "-"))
    )
    return {
        canonicalize_name(requirement.name): requirement for requirement in requirements
    }


def test_rocm_locks_cover_their_inputs() -> None:
    requirements = REPO_ROOT / "requirements"
    build_input = requirements / "build" / "rocm.in"
    build_lock = requirements / "build" / "rocm.txt"
    test_input = requirements / "test" / "rocm.in"
    test_lock = requirements / "test" / "rocm.txt"

    assert requirement_names(build_input) <= requirement_names(build_lock)
    assert requirement_names(test_input) <= requirement_names(test_lock)
    build_requirements = locked_requirements(build_lock)
    test_requirements = locked_requirements(test_lock)
    for runtime_or_test_only in ("datasets", "peft", "pytest", "pytest-asyncio"):
        assert runtime_or_test_only not in build_requirements
    # TorchAudio's pinned requirements.txt installs SoundFile while building.
    assert "soundfile" in build_requirements
    assert str(test_requirements["datasets"].specifier) == "==3.6.0"
    assert str(build_requirements["numpy"].specifier) == "==2.2.6"
    assert build_requirements.keys() <= test_requirements.keys()
    for name, build_requirement in build_requirements.items():
        assert build_requirement == test_requirements[name]
    for lock in (build_lock, test_lock):
        assert all(
            str(requirement.specifier).startswith("==")
            for requirement in locked_requirements(lock).values()
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
            "ROCM_BASE_PYTORCH_ROCM_ARCH": "gfx90a",
            "ROCM_BASE_USE_SCCACHE": "1",
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
    assert cache_key() == baseline
    assert cache_key(SCCACHE_ENDPOINT="https://cache.example.invalid") == baseline
    for path in inputs:
        original = path.read_text()
        path.write_text(f"{original}changed\n")
        assert cache_key() != baseline
        path.write_text(original)
    for overrides in (
        {"ROCM_BASE_PARENT_IMAGE": "rocm/example:other"},
        {"ROCM_BASE_PARENT_DIGEST": f"sha256:{'2' * 64}"},
        {"ROCM_BASE_PLATFORM": "linux/arm64"},
        {"ROCM_BASE_IMAGE_TARGET": "different-image-target"},
        {"ROCM_BASE_WHEEL_TARGET": "different-wheel-target"},
        {"ROCM_BASE_USE_SCCACHE": "0"},
        {"SCCACHE_DOWNLOAD_URL": "https://example.invalid/sccache.tar.gz"},
        {"SCCACHE_VERSION": "v9.9.9"},
        {"SCCACHE_DOWNLOAD_SHA256": "2" * 64},
        {"SCCACHE_BUCKET_NAME": "different-bucket"},
        {"SCCACHE_REGION_NAME": "different-region"},
        {"SCCACHE_S3_NO_CREDENTIALS": "1"},
        {"ROCM_BASE_PYTORCH_ROCM_ARCH": "gfx942"},
    ):
        assert cache_key(**overrides) != baseline


def test_rocm_base_cache_omitted_sccache_matches_disabled(tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile.rocm_base"
    dockerfile.write_text(
        "ARG BASE_IMAGE=rocm/example:base\n"
        "ARG USE_SCCACHE\n"
        "ARG PYTORCH_ROCM_ARCH=gfx90a\n"
    )
    env = os.environ.copy()
    env.update(
        {
            "ROCM_BASE_DOCKERFILE": str(dockerfile),
            "ROCM_BASE_CONTENT_FILES": str(dockerfile),
            "ROCM_BASE_PARENT_DIGEST": f"sha256:{'1' * 64}",
        }
    )

    def cache_key(**overrides: str) -> str:
        return subprocess.run(
            ["bash", str(ROCM_BASE_WHEEL_CACHE), "key"],
            check=True,
            cwd=REPO_ROOT,
            env=env | overrides,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()

    omitted = cache_key()
    assert cache_key(ROCM_BASE_USE_SCCACHE="0") == omitted
    assert cache_key(ROCM_BASE_USE_SCCACHE="1") != omitted


def shell_words_assignment(path: Path, name: str) -> list[str]:
    match = re.search(rf'^{name}="([^"]*)"$', path.read_text(), re.M)
    assert match is not None
    value = match.group(1).replace("${DOCKERFILE}", "docker/Dockerfile.rocm_base")
    return shlex.split(value)


def test_rocm_base_cache_production_content_lists_are_synchronized() -> None:
    expected = [
        "docker/Dockerfile.rocm_base",
        ".dockerignore",
        "requirements/build/rocm.txt",
    ]
    release_match = re.search(
        r'export ROCM_BASE_CONTENT_FILES="([^"]+)"',
        ROCM_RELEASE_BUILD.read_text(),
    )
    assert release_match is not None

    assert shell_words_assignment(ROCM_BASE_WHEEL_CACHE, "DEFAULT_CONTENT_FILES") == (
        expected
    )
    assert (
        shell_words_assignment(ROCM_REFRESH_BASE, "DEFAULT_ROCM_BASE_CONTENT_FILES")
        == expected
    )
    assert shlex.split(release_match.group(1)) == expected


def test_rocm_release_does_not_expose_sccache_download_url_as_build_arg() -> None:
    release_build = ROCM_RELEASE_BUILD.read_text()

    assert "unset ROCM_BASE_PARENT_DIGEST SCCACHE_DOWNLOAD_URL" in release_build
    assert "--build-arg SCCACHE_DOWNLOAD_URL" not in release_build
    assert ".buildkite/scripts/build-rocm-base-wheels.sh" in (
        ROCM_RELEASE_PIPELINE.read_text()
    )


def test_rocm_native_cache_inputs_use_narrow_build_lock() -> None:
    ci_bake = ROCM_CI_BAKE.read_text()
    ci_base_files = shell_words_assignment(
        ROCM_CI_BAKE, "DEFAULT_CI_BASE_CONTENT_FILES"
    )
    csrc_files = shell_words_assignment(ROCM_CI_BAKE, "DEFAULT_ROCM_CSRC_CONTENT_FILES")
    rust_files = shell_words_assignment(ROCM_CI_BAKE, "DEFAULT_ROCM_RUST_CONTENT_FILES")

    assert "requirements/build/rocm.txt" in ci_base_files
    assert "requirements/test/rocm.txt" in ci_base_files
    for content_files in (csrc_files, rust_files):
        assert "requirements/build/rocm.txt" in content_files
        assert "requirements/test/rocm.txt" not in content_files
    for runtime_requirements in (
        "requirements/common.txt",
        "requirements/rocm.txt",
    ):
        assert runtime_requirements not in csrc_files
    assert "requirements/build/rocm.txt" in ci_bake


def test_rocm_metadata_changes_trigger_the_relevant_ci_lanes() -> None:
    docker_area = (REPO_ROOT / ".buildkite" / "test_areas" / "docker.yaml").read_text()
    cpu_sources, amd_mirror_sources = docker_area.split(
        "  mirror:\n    amd:", maxsplit=1
    )
    amd_pipeline = (REPO_ROOT / ".buildkite" / "test-amd.yaml").read_text()
    rocm_config = (REPO_ROOT / ".buildkite" / "ci_config_rocm.yaml").read_text()
    required_sources = (
        ".buildkite/scripts/build-rocm-base-wheels.sh",
        "tests/tools/test_rocm_release_build.py",
        ".buildkite/scripts/rocm/refresh-base-image.sh",
        ".buildkite/scripts/rocm/smoke-test-image.sh",
        "docker/ci-rocm.hcl",
        "docker/docker-bake-rocm.hcl",
        "requirements/build/rocm.in",
        "requirements/build/rocm.txt",
    )

    for source in required_sources:
        assert source in cpu_sources
        assert source in amd_mirror_sources
        assert source in amd_pipeline
    for source in required_sources[-2:]:
        assert source in rocm_config


def test_rocm_cache_helpers_reject_empty_content_lists(tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("ARG BASE_IMAGE=rocm/example:base\nARG USE_SCCACHE=0\n")
    env = os.environ.copy()
    env.update(
        {
            "ROCM_BASE_DOCKERFILE": str(dockerfile),
            "ROCM_BASE_CONTENT_FILES": "   ",
            "ROCM_BASE_PARENT_DIGEST": f"sha256:{'1' * 64}",
            "ROCM_BASE_PYTORCH_ROCM_ARCH": "gfx90a",
        }
    )
    cache_result = subprocess.run(
        ["bash", str(ROCM_BASE_WHEEL_CACHE), "key"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    refresh_result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; compute_base_content_hash 0 "$2" gfx90a linux/amd64',
            "bash",
            str(ROCM_REFRESH_BASE),
            f"sha256:{'1' * 64}",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert cache_result.returncode != 0
    assert "content" in cache_result.stderr.lower()
    assert refresh_result.returncode != 0
    assert "content" in refresh_result.stderr.lower()


def test_rocm_forced_base_refresh_still_exports_layer_cache() -> None:
    env = os.environ.copy()
    env["ROCM_BASE_REFRESH_FORCE"] = "1"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; configure_rocm_base_layer_cache; '
            'printf "%s\\n" "${ROCM_BASE_CACHE_ARGS[@]}"',
            "bash",
            str(ROCM_REFRESH_BASE),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    args = result.stdout.splitlines()

    assert "--no-cache" in args
    assert "--cache-to" in args
    assert "--cache-from" not in args


def test_rocm_base_preview_image_refs_are_source_scoped() -> None:
    def preview_ref(repo: str) -> str:
        env = os.environ.copy()
        env.update(
            {
                "BUILDKITE": "false",
                "BUILDKITE_BRANCH": "feature",
                "BUILDKITE_REPO": repo,
            }
        )
        return subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; scoped_base_content_ref "$2" 3',
                "bash",
                str(ROCM_REFRESH_BASE),
                "a" * 64,
            ],
            check=True,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()

    first = preview_ref("https://github.com/example/one.git")
    second = preview_ref("https://github.com/example/two.git")
    assert first != second
    assert first.endswith("a" * 64)
    assert second.endswith("a" * 64)


def test_rocm_dependency_cache_writes_are_source_scoped() -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
source "$1"
DOCKERHUB_CACHE_REPO=example/cache
NIXL_CACHE_KEY=key
CI_BASE_WRITE_SCOPE=preview-owner
BUILDKITE_BUILD_ID=build-one
trusted_dependency_cache_ref_for_target nixl-rocm-ci
dependency_cache_ref_for_target nixl-rocm-ci
BUILDKITE_BUILD_ID=build-two
dependency_cache_ref_for_target nixl-rocm-ci
CI_BASE_WRITE_SCOPE=
dependency_cache_ref_for_target nixl-rocm-ci
""",
            "bash",
            str(ROCM_CI_BAKE),
        ],
        check=True,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        text=True,
    )
    trusted, first_write, second_write, trusted_write = result.stdout.splitlines()

    assert trusted == "example/cache:nixl-rocm-key"
    assert first_write.startswith("example/cache:nixl-rocm-preview-owner-")
    assert second_write.startswith("example/cache:nixl-rocm-preview-owner-")
    assert first_write != second_write
    assert trusted_write == trusted


def test_rocm_bake_values_are_escaped_as_literal_hcl_strings() -> None:
    # Build arguments, metadata and cache lists must not evaluate templates or
    # let quotes/newlines change the generated Bake configuration structure.
    for value, escaped in (
        ("gfx90a;gfx942", "gfx90a;gfx942"),
        ('quote"\\path\n\r\t', 'quote\\"\\\\path\\n\\r\\t'),
        ("${1 + 1} %{if true}value%{endif}", "$${1 + 1} %%{if true}value%%{endif}"),
    ):
        result = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; hcl_escape_string "$2"; printf "\\n"; '
                'write_hcl_string_list_entries "  " "$2"',
                "bash",
                str(ROCM_CI_BAKE),
                value,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.stdout == f'{escaped}\n  "{escaped}",\n'


def test_rocm_wrappers_reject_secret_endpoints_without_logging_them() -> None:
    # Source the wrappers to exercise their early guard without Docker or AWS.
    for helper in (ROCM_CI_BAKE, ROCM_REFRESH_BASE):
        for endpoint in (
            "https://user:secret@cache.example",
            "https://cache.example?token=secret",
            "https://cache.example#secret",
            "https://cache.example\nsecret",
        ):
            result = subprocess.run(
                ["bash", "-c", 'source "$1"', "bash", str(helper)],
                env=os.environ | {"SCCACHE_ENDPOINT": endpoint},
                capture_output=True,
                text=True,
            )
            assert result.returncode != 0
            assert "SCCACHE_ENDPOINT must not contain" in result.stderr
            assert endpoint not in result.stdout + result.stderr

        for endpoint in ("", "http://localhost:9000", "https://cache.example"):
            subprocess.run(
                ["bash", "-c", 'source "$1"', "bash", str(helper)],
                env=os.environ | {"SCCACHE_ENDPOINT": endpoint},
                check=True,
            )


def test_rocm_commit_image_without_revision_is_rebuilt() -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
source "$1"
TARGET=test-rocm-ci
IMAGE_TAG=example/image:commit
BUILDKITE_COMMIT=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
remote_image_exists() { return 0; }
get_remote_image_label() { return 0; }
maybe_skip_existing_image
echo continued-to-build
""",
            "bash",
            str(ROCM_CI_BAKE),
        ],
        check=True,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        text=True,
    )

    assert "found revision: <missing>" in result.stdout
    assert result.stdout.splitlines()[-1] == "continued-to-build"


def test_rocm_force_disables_layers_for_non_base_targets() -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
source "$1"
TARGET=test-rocm-ci
FORCE_BUILD=1
BAKE_TARGETS=(test-rocm-ci)
BAKE_ALLOW_ARGS=()
BAKE_FILES=()
docker() { printf '%s\\n' "$*"; }
run_bake
""",
            "bash",
            str(ROCM_CI_BAKE),
        ],
        check=True,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        text=True,
    )

    assert "buildx bake --no-cache" in result.stdout


def docker_arg_value(dockerfile: str, name: str) -> str:
    prefix = f"ARG {name}="
    line = next(line for line in dockerfile.splitlines() if line.startswith(prefix))
    return line.removeprefix(prefix).split("#", maxsplit=1)[0].strip().strip('"')


def test_rocm_image_dependency_inputs_are_immutable() -> None:
    rocm_base = (REPO_ROOT / "docker" / "Dockerfile.rocm_base").read_text()
    rocm_ci = (REPO_ROOT / "docker" / "Dockerfile.rocm").read_text()

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
                "DECORD_COMMIT",
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
        (
            rocm_ci,
            (
                "UV_DOWNLOAD_SHA256",
                "RUSTUP_INIT_SHA256",
                "ROCM_TRITON_KERNELS_SHA256",
            ),
        ),
    ):
        for name in checksum_arg_names:
            value = docker_arg_value(dockerfile, name)
            assert len(value) == 64
            int(value, 16)


def test_rocm_source_builds_use_the_narrow_constraints() -> None:
    rocm_base = (REPO_ROOT / "docker" / "Dockerfile.rocm_base").read_text()
    rocm_ci = (REPO_ROOT / "docker" / "Dockerfile.rocm").read_text()

    assert "source=requirements/test/rocm.txt" not in rocm_base
    assert "source=requirements/build/rocm.txt" in rocm_base
    assert "COPY requirements/build/rocm.txt" in rocm_ci
    build_dependencies = rocm_ci.split(
        "FROM base AS build_vllm_dependencies", maxsplit=1
    )[1].split("FROM base AS rocm-triton-kernels", maxsplit=1)[0]
    assert "--requirement /tmp/rocm-build-constraints.txt" in build_dependencies
    assert "-r requirements/rocm.txt" not in build_dependencies
    assert "/requirements/common.txt" not in build_dependencies
    assert "/requirements/rocm.txt" not in build_dependencies
    assert "# Native-extension-only build." in build_dependencies
    assert "'-r common.txt' > requirements/rocm.txt" in build_dependencies
    assert (
        "COPY --from=export_vllm /requirements/test/rocm.txt /tmp/rocm-constraints.txt"
    ) in rocm_ci
    for expected in (
        "pip wheel --no-build-isolation --no-deps",
        "TRITON_OFFLINE_BUILD=1 TRITON_BUILD_PROTON=OFF",
        "FLASH_ATTENTION_FORCE_BUILD=TRUE",
        "AITER_USE_SYSTEM_TRITON=1",
        "pip check",
    ):
        assert expected in rocm_base

    # These two ABI-sensitive extensions must independently opt out of binary
    # wheels; combining their flags would allow one regression to mask another.
    assert "--no-binary arctic-inference" in rocm_ci
    assert "--no-binary fastsafetensors" in rocm_ci
    assert "--build-constraints /tmp/lmcache-build-constraints.txt" in rocm_ci


def test_rocm_native_build_graph_only_serializes_aiter_on_triton() -> None:
    rocm_base = (REPO_ROOT / "docker" / "Dockerfile.rocm_base").read_text()
    dependencies = dockerfile_stage_dependencies(rocm_base)

    assert dependencies["build_pytorch_runtime"] == {"base", "build_pytorch"}
    assert dependencies["build_pytorch_triton_runtime"] == {
        "build_pytorch_runtime",
        "build_triton",
    }
    for stage in (
        "build_torchvision",
        "build_torchaudio",
        "build_mori",
        "build_fa",
    ):
        assert dependencies[stage] == {"build_pytorch_runtime"}
    assert dependencies["build_aiter"] == {"build_pytorch_triton_runtime"}


def test_rocm_runtime_dependency_guards_are_enforced() -> None:
    rocm_base = (REPO_ROOT / "docker" / "Dockerfile.rocm_base").read_text()
    rocm_ci = (REPO_ROOT / "docker" / "Dockerfile.rocm").read_text()
    rocm_base_final = rocm_base.split("FROM base AS final", maxsplit=1)[1]

    for dockerfile in (rocm_base, rocm_ci):
        assert "ARG USE_SCCACHE\n" in dockerfile.split("FROM ", maxsplit=1)[0]

    assert (
        "apt-get purge -y software-properties-common "
        "python3-software-properties python3-gi"
    ) in rocm_base
    assert "PyGObject==" not in rocm_base
    assert "python3 -m pip check" in rocm_base
    assert "&& pip check" in rocm_base
    assert (
        "pip install --constraint /tmp/rocm-constraints.txt pyyaml /install/*.whl"
    ) in rocm_base_final
    assert "pyyaml" in locked_requirements(ROCM_BUILD_LOCK)

    assert (
        'test "$(rustc --version | awk \'{print $2}\')" = "${RUST_TOOLCHAIN_VERSION}"'
    ) in rocm_ci
    assert "revision = e5fada43131d251e9c4786b04263ce98b6767ba5" in rocm_ci
    assert "uv pip check --system" in rocm_ci


def test_rocm_decord_and_numpy_runtime_match_the_source_builds() -> None:
    rocm_ci = (REPO_ROOT / "docker" / "Dockerfile.rocm").read_text()
    numpy_pin = str(
        locked_requirements(ROCM_BUILD_LOCK)["numpy"].specifier
    ).removeprefix("==")

    for expected in (
        "FROM ci_base_system AS build_decord",
        "git -C /tmp/decord checkout -q --detach FETCH_HEAD",
        'test "$(git -C /tmp/decord rev-parse HEAD)" = "${DECORD_COMMIT}"',
        "git -C /tmp/decord submodule update --init --recursive --depth 1",
        "-DUSE_CUDA=OFF",
        "python3 -m pip wheel --no-cache-dir --no-build-isolation --no-deps",
        "python3 -m pip install --no-cache-dir --force-reinstall --no-deps",
        "python3 -c \"import decord; print('decord', decord.__version__)\"",
    ):
        assert expected in rocm_ci

    assert numpy_pin == "2.2.6"
    assert 'grep -Fxc "numpy<=${NUMPY_VERSION}"' in rocm_ci
    assert f"numpy=={numpy_pin}" not in rocm_ci
    assert '-e "s/^numpy' not in rocm_ci
    assert "import numpy; print(numpy.__version__)" in rocm_ci
    assert '"${NUMPY_VERSION}"' in rocm_ci


def test_rocm_release_lmcache_installs_its_runtime_extras() -> None:
    rocm_ci = (REPO_ROOT / "docker" / "Dockerfile.rocm").read_text()
    rocm_test_roots = {
        line.split("#", maxsplit=1)[0].strip()
        for line in ROCM_TEST_INPUT.read_text().splitlines()
    }
    final_lmcache = rocm_ci.split(
        "FROM final_common AS final_lmcache_true", maxsplit=1
    )[1].split("FROM final_lmcache_${INSTALL_LMCACHE} AS final", maxsplit=1)[0]

    assert "--constraint /tmp/rocm-constraints.txt" in final_lmcache
    for dependency in (
        "aiofile",
        "aiofiles",
        "awscrt",
        "cupy-rocm-7-0",
        "google-cloud-bigtable",
        "opentelemetry-exporter-prometheus",
        "sortedcontainers",
    ):
        assert dependency in final_lmcache
        assert dependency in rocm_test_roots
    assert "uv pip check --system" in final_lmcache


def test_torchcodec_build_contract_is_abi_safe() -> None:
    rocm_ci = (REPO_ROOT / "docker" / "Dockerfile.rocm").read_text()
    torchcodec = TORCHCODEC_INSTALLER.read_text()

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
    assert 'TORCHCODEC_FORCE_REBUILD="${TORCHCODEC_FORCE_REBUILD:-0}"' in torchcodec
    assert 'case "$TORCHCODEC_FORCE_REBUILD" in' in torchcodec
    assert '--force-reinstall --no-deps "$BUILT_WHEEL"' in torchcodec
    assert "TORCHCODEC_FORCE_REBUILD=1" in rocm_ci
    assert "TORCHCODEC_CONSTRAINTS=/tmp/rocm-build-constraints.txt" in rocm_ci


def test_torchcodec_working_install_skips_before_any_build(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python_calls = tmp_path / "python-calls"
    python = fake_bin / "python3"
    python.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$FAKE_PYTHON_CALLS"\n'
        'test "$1" = "-c"\n'
        'test "$2" = "from torchcodec.decoders import VideoDecoder"\n'
    )
    python.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "FAKE_PYTHON_CALLS": str(python_calls),
            "PATH": f"{fake_bin}:{env['PATH']}",
            "TORCHCODEC_FORCE_REBUILD": "0",
        }
    )
    result = subprocess.run(
        ["bash", str(TORCHCODEC_INSTALLER)],
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        text=True,
    )

    assert "already installed and working" in result.stdout
    assert python_calls.read_text().splitlines() == [
        "-c from torchcodec.decoders import VideoDecoder"
    ]


def test_torchcodec_force_rebuild_rejects_invalid_values() -> None:
    env = os.environ.copy()
    env["TORCHCODEC_FORCE_REBUILD"] = "sometimes"
    result = subprocess.run(
        ["bash", str(TORCHCODEC_INSTALLER)],
        env=env,
        stdout=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 2
    assert "must be 0 or 1" in result.stdout


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
