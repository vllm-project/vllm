# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import shlex
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / ".buildkite" / "scripts" / "docker-build-metadata-args.sh"
CI_BAKE = REPO_ROOT / ".buildkite" / "scripts" / "ci-bake-rocm.sh"
ROCM_BASE_REFRESH = (
    REPO_ROOT / ".buildkite" / "scripts" / "rocm" / "refresh-base-image.sh"
)
ROCM_CI_HCL = REPO_ROOT / "docker" / "ci-rocm.hcl"
AMD_HARDWARE_PIPELINE = REPO_ROOT / ".buildkite" / "hardware_tests" / "amd.yaml"


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


def run_ci_bake_shell(
    tmp_path: Path,
    command: str,
    *,
    env: Mapping[str, str | None] | None = None,
) -> str:
    helper_env = os.environ.copy()
    for key, value in (env or {}).items():
        if value is None:
            helper_env.pop(key, None)
        else:
            helper_env[key] = value

    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "$1"\n{command}',
            "ci-bake-rocm-test",
            str(CI_BAKE),
        ],
        check=True,
        cwd=REPO_ROOT,
        env=helper_env,
        capture_output=True,
        text=True,
    )
    return result.stdout


def compute_ci_base_hash(
    tmp_path: Path,
    dockerfile: Path,
    content_file: Path,
    *,
    env: dict[str, str],
) -> str:
    return run_ci_bake_shell(
        tmp_path,
        "compute_ci_base_content_hash_once",
        env={
            "CI_BASE_CONTENT_FILES": str(content_file),
            "CI_BASE_DOCKERFILE": str(dockerfile),
            "CI_BASE_DOCKERFILE_STAGES": "input",
            **env,
        },
    ).strip()


def write_rocm_input_dockerfile(tmp_path: Path) -> tuple[Path, Path]:
    docker_dir = tmp_path / "docker"
    docker_dir.mkdir()
    dockerfile = docker_dir / "Dockerfile.rocm"
    dockerfile.write_text(
        "FROM scratch AS input\n"
        "ARG REMOTE_VLLM\n"
        "ARG VLLM_REPO\n"
        "ARG VLLM_BRANCH\n"
        "ARG NIXL_BRANCH\n"
        'RUN echo "$REMOTE_VLLM $VLLM_REPO $VLLM_BRANCH $NIXL_BRANCH"\n'
    )
    bake_file = docker_dir / "docker-bake-rocm.hcl"
    bake_file.write_text("")
    return dockerfile, bake_file


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
    default_files = next(
        line
        for line in ci_bake.splitlines()
        if line.startswith("DEFAULT_CI_BASE_CONTENT_FILES=")
    )

    for expected in (
        "requirements/common.txt",
        "requirements/rocm.txt",
        "requirements/test/rocm.txt",
        "docker/Dockerfile.rocm",
        "docker/ci-rocm.hcl",
        "docker/docker-bake-rocm.hcl",
        "rust-toolchain.toml",
        "tools/install_protoc.sh",
        ".buildkite/scripts/ci-bake-rocm.sh",
    ):
        assert expected in ci_bake
    assert "docker/Dockerfile.rocm_base" not in default_files


def test_rocm_hash_walkers_force_c_locale() -> None:
    for script in (CI_BAKE, ROCM_BASE_REFRESH):
        contents = script.read_text()
        assert "| LC_ALL=C sort -z" in contents
        assert "-name __pycache__" in contents
        assert "! -name '*.py[cod]'" in contents


def test_rocm_content_hash_matches_docker_visible_inputs(tmp_path: Path) -> None:
    content_dir = tmp_path / "content"
    cache_dir = content_dir / "__pycache__"
    content_dir.mkdir()
    cache_dir.mkdir()
    source = content_dir / "input.py"
    source.write_text("VALUE = 1\n")

    first_hash = run_ci_bake_shell(
        tmp_path,
        f'compute_content_hash "{content_dir}"',
    ).strip()
    (cache_dir / "input.cpython-312.pyc").write_bytes(b"generated")
    (content_dir / "stray.pyc").write_bytes(b"generated")
    dist_dir = content_dir / "dist"
    dist_dir.mkdir()
    (dist_dir / "generated.whl").write_bytes(b"generated")
    cached_hash = run_ci_bake_shell(
        tmp_path,
        f'compute_content_hash "{content_dir}"',
    ).strip()
    source.chmod(0o755)
    executable_hash = run_ci_bake_shell(
        tmp_path,
        f'compute_content_hash "{content_dir}"',
    ).strip()
    source.chmod(0o644)
    symlink = content_dir / "input-link.py"
    symlink.symlink_to("input.py")
    symlink_hash = run_ci_bake_shell(
        tmp_path,
        f'compute_content_hash "{content_dir}"',
    ).strip()
    symlink.unlink()
    source.write_text("VALUE = 2\n")
    changed_hash = run_ci_bake_shell(
        tmp_path,
        f'compute_content_hash "{content_dir}"',
    ).strip()

    assert cached_hash == first_hash
    assert executable_hash != first_hash
    assert symlink_hash != first_hash
    assert changed_hash != first_hash


def test_rocm_ci_base_local_hash_ignores_checkout_coordinates(
    tmp_path: Path,
) -> None:
    dockerfile, _ = write_rocm_input_dockerfile(tmp_path)
    content_file = tmp_path / "rust-toolchain.toml"
    content_file.write_text('channel = "1.90.0"\n')

    inputs = {
        "REMOTE_VLLM": "0",
        "VLLM_REPO": "https://github.com/vllm-project/vllm.git",
        "VLLM_BRANCH": "first-commit",
        "NIXL_BRANCH": "nixl-main",
    }
    first_hash = compute_ci_base_hash(
        tmp_path,
        dockerfile,
        content_file,
        env=inputs,
    )
    relocated_hash = compute_ci_base_hash(
        tmp_path,
        dockerfile,
        content_file,
        env=inputs
        | {
            "VLLM_REPO": "https://github.com/fork/vllm.git",
            "VLLM_BRANCH": "second-commit",
        },
    )
    remote_hash = compute_ci_base_hash(
        tmp_path,
        dockerfile,
        content_file,
        env=inputs | {"REMOTE_VLLM": "1"},
    )
    changed_arg_hash = compute_ci_base_hash(
        tmp_path,
        dockerfile,
        content_file,
        env=inputs | {"NIXL_BRANCH": "nixl-next"},
    )
    content_file.write_text('channel = "1.91.0"\n')
    changed_file_hash = compute_ci_base_hash(
        tmp_path,
        dockerfile,
        content_file,
        env=inputs,
    )

    assert relocated_hash == first_hash
    assert remote_hash != first_hash
    assert changed_arg_hash != first_hash
    assert changed_file_hash != first_hash


def test_rocm_stage_parser_accepts_lowercase_from_and_crlf(tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_bytes(
        b"from scratch AS wanted\r\n"
        b"ARG INPUT\r\n"
        b"RUN echo $INPUT\r\n"
        b"FROM scratch AS ignored\r\n"
        b"RUN echo ignored\r\n"
    )

    output = run_ci_bake_shell(
        tmp_path,
        (
            f'hash_dockerfile_stages "{dockerfile}" "wanted"\n'
            'printf "args:\\n"\n'
            f'discover_dockerfile_stage_args "{dockerfile}" "wanted"'
        ),
    )

    assert "\r" not in output
    assert "from scratch AS wanted" in output
    assert "RUN echo $INPUT" in output
    assert "ignored" not in output
    assert output.endswith("args:\nINPUT\n")


def test_rocm_build_arg_override_keeps_checkout_coordinates(
    tmp_path: Path,
) -> None:
    _, bake_file = write_rocm_input_dockerfile(tmp_path)
    override_file = tmp_path / "rocm-arg-override.hcl"
    inputs = {
        "REMOTE_VLLM": "0",
        "VLLM_REPO": "https://github.com/vllm-project/vllm.git",
        "VLLM_BRANCH": "first-commit",
        "NIXL_BRANCH": "nixl-main",
    }
    output = run_ci_bake_shell(
        tmp_path,
        (
            f'VLLM_BAKE_FILE="{bake_file}"\n'
            f'ROCM_ARG_OVERRIDE_PATH="{override_file}"\n'
            'CI_BASE_DOCKERFILE_STAGES="input"\n'
            "write_rocm_build_arg_override >/dev/null\n"
            f'cat "{override_file}"'
        ),
        env=inputs,
    )
    assert 'REMOTE_VLLM = "0"' in output
    assert 'VLLM_REPO = "https://github.com/vllm-project/vllm.git"' in output
    assert 'VLLM_BRANCH = "first-commit"' in output


def test_rocm_build_uses_the_base_image_digest_that_was_hashed(
    tmp_path: Path,
) -> None:
    docker_dir = tmp_path / "docker"
    docker_dir.mkdir()
    dockerfile = docker_dir / "Dockerfile.rocm"
    dockerfile.write_text(
        "ARG BASE_IMAGE=rocm/example:base\nFROM ${BASE_IMAGE} AS base\n"
    )
    bake_file = docker_dir / "docker-bake-rocm.hcl"
    bake_file.write_text("")
    override_file = tmp_path / "rocm-arg-override.hcl"
    digest = "sha256:" + "a" * 64

    output = run_ci_bake_shell(
        tmp_path,
        (
            f'CI_BASE_DOCKERFILE="{dockerfile}"\n'
            f'VLLM_BAKE_FILE="{bake_file}"\n'
            f'ROCM_ARG_OVERRIDE_PATH="{override_file}"\n'
            'CI_BASE_DOCKERFILE_STAGES="base"\n'
            f'resolve_image_digest() {{ printf "%s\\n" "{digest}"; }}\n'
            "prime_base_image_digest_cache >/dev/null\n"
            "write_rocm_build_arg_override >/dev/null\n"
            f'cat "{override_file}"'
        ),
    )

    assert f'BASE_IMAGE = "rocm/example:base@{digest}"' in output


def test_rocm_base_refresh_builds_from_its_resolved_digest() -> None:
    refresh_script = ROCM_BASE_REFRESH.read_text()

    assert 'base_image_pinned="${base_image_arg%@*}@${base_image_digest}"' in (
        refresh_script
    )
    assert '--build-arg "BASE_IMAGE=${base_image_pinned}"' in refresh_script


@pytest.mark.parametrize(
    ("environment", "expected_push_ref"),
    [
        (
            {
                "BUILDKITE_PULL_REQUEST": "49516",
                "BUILDKITE_BRANCH": "feature",
                "CI_BASE_PUSH_STABLE_TAG": "1",
                "NIGHTLY": "1",
            },
            "",
        ),
        (
            {
                "BUILDKITE_PULL_REQUEST": "false",
                "BUILDKITE_BRANCH": "main",
                "CI_BASE_PUSH_STABLE_TAG": None,
                "NIGHTLY": "1",
            },
            "rocm/example:ci_base",
        ),
    ],
)
def test_rocm_ci_base_reads_stable_cache_without_unauthorized_push(
    tmp_path: Path,
    environment: dict[str, str | None],
    expected_push_ref: str,
) -> None:
    output = run_ci_bake_shell(
        tmp_path,
        (
            'TARGET="ci-base-rocm-ci"\n'
            "configure_ci_base_image_refs >/dev/null\n"
            'printf "cache=%s\\npush=%s\\nlookup=%s\\noutputs=%s\\n" '
            '"${CI_BASE_STABLE_CACHE_REF}" '
            '"${CI_BASE_IMAGE_TAG_STABLE:-}" '
            '"$(ci_base_candidate_refs | paste -sd, -)" '
            '"$(ci_base_output_refs | paste -sd, -)"'
        ),
        env={
            **environment,
            "BUILDKITE_COMMIT": "0123456789abcdef",
            "CI_BASE_CONTENT_HASH": "content-hash",
            "CI_BASE_IMAGE_TAG": "rocm/example:ci_base",
        },
    )
    refs = dict(line.split("=", 1) for line in output.splitlines())

    assert refs["cache"] == "rocm/example:ci_base"
    assert refs["push"] == expected_push_ref
    assert "rocm/example:ci_base" in refs["lookup"].split(",")
    assert ("rocm/example:ci_base" in refs["outputs"].split(",")) == bool(
        expected_push_ref
    )


def test_rocm_ci_base_rejects_an_empty_content_hash(tmp_path: Path) -> None:
    output = run_ci_bake_shell(
        tmp_path,
        (
            'TARGET="ci-base-rocm-ci"\n'
            'CI_BASE_CONTENT_HASH=""\n'
            "if configure_ci_base_image_refs 2>/dev/null; then\n"
            '  printf "unexpected-success\\n"\n'
            "else\n"
            '  printf "rejected\\n"\n'
            "fi"
        ),
        env={
            "BUILDKITE_PULL_REQUEST": "48646",
            "CI_BASE_IMAGE_TAG": "rocm/example:ci_base",
        },
    )

    assert output.strip() == "rejected"


def test_rocm_ci_base_stable_cache_is_read_only_in_bake() -> None:
    hcl = ROCM_CI_HCL.read_text()
    tags_line = next(
        line
        for line in hcl.splitlines()
        if line.lstrip().startswith("tags") and "CI_BASE_IMAGE_TAG_STABLE" in line
    )

    assert (
        'CI_BASE_STABLE_CACHE_REF != "" ? '
        '"type=registry,ref=${CI_BASE_STABLE_CACHE_REF}" : ""'
    ) in hcl
    assert (
        'CI_BASE_IMAGE_TAG_STABLE != "" ? '
        '"type=registry,ref=${CI_BASE_IMAGE_TAG_STABLE}" : ""'
    ) not in hcl
    assert "CI_BASE_IMAGE_TAG_STABLE" in tags_line
    assert "CI_BASE_STABLE_CACHE_REF" not in tags_line


def test_rocm_ci_base_finds_an_exact_stable_image_after_a_stale_primary(
    tmp_path: Path,
) -> None:
    output = run_ci_bake_shell(
        tmp_path,
        (
            'CI_BASE_CONTENT_HASH="expected"\n'
            'CI_BASE_METADATA_VERSION="2"\n'
            'IMAGE_TAG="rocm/example:primary"\n'
            'CI_BASE_IMAGE_TAG="rocm/example:primary"\n'
            'CI_BASE_IMAGE_TAG_CONTENT_EXTRA="rocm/example:content"\n'
            'CI_BASE_STABLE_CACHE_REF="rocm/example:stable"\n'
            "remote_image_exists() { return 0; }\n"
            "get_remote_image_label() {\n"
            '  if [[ "$1" == "rocm/example:stable" ]]; then\n'
            '    printf "expected\\n"\n'
            "  else\n"
            '    printf "stale\\n"\n'
            "  fi\n"
            "}\n"
            "remote_ci_base_metadata_is_current() {\n"
            '  [[ "$1" == "rocm/example:stable" ]]\n'
            "}\n"
            "find_matching_ci_base_ref"
        ),
    )

    assert output.strip() == "rocm/example:stable"


def test_rocm_ci_base_refresh_never_writes_its_stable_cache_source(
    tmp_path: Path,
) -> None:
    output = run_ci_bake_shell(
        tmp_path,
        (
            'CI_BASE_CONTENT_HASH="expected"\n'
            'IMAGE_TAG="rocm/example:commit"\n'
            'CI_BASE_IMAGE_TAG="rocm/example:commit"\n'
            'CI_BASE_IMAGE_TAG_CONTENT_EXTRA="rocm/example:content"\n'
            'CI_BASE_IMAGE_TAG_STABLE=""\n'
            'CI_BASE_STABLE_CACHE_REF="rocm/example:stable"\n'
            'get_remote_image_label() { printf "stale\\n"; }\n'
            'docker() { printf "docker:%s\\n" "$*"; }\n'
            'refresh_ci_base_tags_from_ref "rocm/example:stable"'
        ),
    )

    assert "create -t rocm/example:commit rocm/example:stable" in output
    assert "create -t rocm/example:content rocm/example:stable" in output
    assert "create -t rocm/example:stable" not in output


def test_rocm_merge_base_does_not_trust_a_fork_origin(tmp_path: Path) -> None:
    output = run_ci_bake_shell(
        tmp_path,
        (
            "git() {\n"
            '  case "$*" in\n'
            '    "rev-parse --is-shallow-repository") printf "false\\n" ;;\n'
            '    "remote get-url origin") '
            'printf "https://github.com/example-fork/vllm.git\\n" ;;\n'
            '    "merge-base HEAD refs/remotes/origin/main") '
            'printf "wrong-local-base\\n" ;;\n'
            '    "merge-base HEAD refs/remotes/vllm-cache-upstream/main") '
            'printf "correct-upstream-base\\n" ;;\n'
            "    *) return 1 ;;\n"
            "  esac\n"
            "}\n"
            "git_fetch_for_cache() { return 0; }\n"
            'TARGET="test-rocm-ci"\n'
            'PARENT_COMMIT="parent"\n'
            'ROCM_CACHE_BRANCH_TAG="branch"\n'
            'ROCM_CACHE_UPSTREAM_BRANCH_TAG="upstream"\n'
            "prepare_git_cache_metadata >/dev/null\n"
            'printf "%s\\n" "$VLLM_MERGE_BASE_COMMIT"'
        ),
        env={
            "BUILDKITE_PULL_REQUEST": "48646",
            "BUILDKITE_PULL_REQUEST_BASE_BRANCH": "main",
            "BUILDKITE_REPO": "https://github.com/vllm-project/vllm.git",
        },
    )

    assert output.strip() == "correct-upstream-base"


def test_rocm_ci_base_uses_local_sparse_inputs() -> None:
    pipeline = yaml.safe_load(AMD_HARDWARE_PIPELINE.read_text())
    steps = {step["key"]: step for step in pipeline["steps"]}

    ensure_env = steps["ensure-ci-base-amd"]["env"]
    image_env = steps["image-build-amd"]["env"]

    assert ensure_env["REMOTE_VLLM"] == "0"
    assert "VLLM_BRANCH" not in ensure_env
    assert image_env["REMOTE_VLLM"] == "1"
    assert image_env["VLLM_BRANCH"] == "$BUILDKITE_COMMIT"
