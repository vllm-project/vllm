# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import shlex
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
import regex as re
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_BAKE = REPO_ROOT / ".buildkite" / "scripts" / "ci-bake-rocm.sh"
ROCM_BASE_REFRESH = (
    REPO_ROOT / ".buildkite" / "scripts" / "rocm" / "refresh-base-image.sh"
)
ROCM_TEST_IMAGE = REPO_ROOT / ".buildkite" / "scripts" / "rocm" / "build-test-image.sh"
AMD_HARDWARE_PIPELINE = REPO_ROOT / ".buildkite" / "hardware_tests" / "amd.yaml"


def run_shell(
    command: str,
    *args: Path,
    env: Mapping[str, str | None] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    helper_env = os.environ.copy()
    for key in tuple(helper_env):
        if key.startswith(("BUILDKITE", "CI_BASE_", "ROCM_")):
            helper_env.pop(key)
    for key in (
        "BAKE_PRINT_ONLY",
        "BASE_IMAGE",
        "FORCE_BUILD",
        "IMAGE_TAG",
        "NIGHTLY",
        "REMOTE_VLLM",
        "TARGET",
        "VLLM_BAKE_FILE",
        "VLLM_REPO",
        "VLLM_BRANCH",
    ):
        helper_env.pop(key, None)
    for key, value in (env or {}).items():
        if value is None:
            helper_env.pop(key, None)
        else:
            helper_env[key] = value

    result = subprocess.run(
        [
            "bash",
            "-c",
            "buildkite-agent() { return 127; }\n" + command,
            "rocm-ci-cache-test",
            *(str(arg) for arg in args),
        ],
        check=False,
        cwd=REPO_ROOT,
        env=helper_env,
        capture_output=True,
        text=True,
    )
    if check:
        result.check_returncode()
    return result


def run_sourced_shell(
    script: Path,
    command: str,
    *,
    env: Mapping[str, str | None] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return run_shell(
        f'source "$1"\n{command}',
        script,
        env=env,
        check=check,
    )


def run_ci_bake_shell(
    command: str,
    *,
    env: Mapping[str, str | None] | None = None,
) -> str:
    return run_sourced_shell(CI_BAKE, command, env=env).stdout


def write_fake_docker(tmp_path: Path, body: str) -> tuple[Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True)
    call_log = tmp_path / "docker-calls"
    docker = fake_bin / "docker"
    docker.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    docker.chmod(0o755)
    return fake_bin, call_log


def compute_ci_base_hash(
    dockerfile: Path,
    content_file: Path,
    *,
    env: dict[str, str],
) -> str:
    return run_ci_bake_shell(
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


def dockerfile_instructions(source: str) -> list[str]:
    instructions: list[str] = []
    current = ""

    for raw_line in source.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        current = f"{current} {line}".strip()
        if current.endswith("\\"):
            current = current[:-1].rstrip()
            continue
        instructions.append(current)
        current = ""
    assert not current, "unterminated Dockerfile instruction"
    return instructions


def test_ci_base_bake_embeds_content_hash_label() -> None:
    bake_file = (REPO_ROOT / "docker" / "docker-bake-rocm.hcl").read_text()

    for expected in (
        'variable "CI_BASE_CONTENT_HASH"',
        'target "ci-base-rocm"',
        'target   = "ci_base"',
        '"vllm.ci_base.content_hash" = CI_BASE_CONTENT_HASH',
    ):
        assert expected in bake_file


def test_content_hash_tracks_build_inputs_and_fails_closed(
    tmp_path: Path,
) -> None:
    content_dir = tmp_path / "content"
    cache_dir = content_dir / "__pycache__"
    content_dir.mkdir()
    cache_dir.mkdir()
    source = content_dir / "input.py"
    source.write_text("VALUE = 1\n")

    first_hash = run_ci_bake_shell(f'compute_content_hash "{content_dir}"').strip()
    (cache_dir / "input.cpython-312.pyc").write_bytes(b"generated")
    (content_dir / "stray.pyc").write_bytes(b"generated")
    dist_dir = content_dir / "dist"
    dist_dir.mkdir()
    (dist_dir / "generated.whl").write_bytes(b"generated")
    ignored_hash = run_ci_bake_shell(f'compute_content_hash "{content_dir}"').strip()

    source.chmod(0o755)
    executable_hash = run_ci_bake_shell(f'compute_content_hash "{content_dir}"').strip()
    source.chmod(0o644)
    symlink = content_dir / "input-link.py"
    symlink.symlink_to("input.py")
    symlink_hash = run_ci_bake_shell(f'compute_content_hash "{content_dir}"').strip()
    symlink.unlink()
    source.write_text("VALUE = 2\n")
    changed_hash = run_ci_bake_shell(f'compute_content_hash "{content_dir}"').strip()

    assert ignored_hash == first_hash
    assert executable_hash != first_hash
    assert symlink_hash != first_hash
    assert changed_hash != first_hash

    for script in (CI_BAKE, ROCM_BASE_REFRESH):
        result = run_sourced_shell(
            script,
            'list_content_files() { return 42; }\ncompute_content_hash "tests"',
            check=False,
        )
        assert result.returncode == 42
        assert "failed to enumerate content files under tests" in result.stderr


def test_ci_base_hash_inputs_cover_selected_stage_dependencies() -> None:
    contract = run_ci_bake_shell(
        'printf "files=%s\\ndockerfile=%s\\nstages=%s\\n" '
        '"${DEFAULT_CI_BASE_CONTENT_FILES}" '
        '"${DEFAULT_CI_BASE_DOCKERFILE}" '
        '"${DEFAULT_CI_BASE_DOCKERFILE_STAGES}"\n'
        "hash_dockerfile_stages "
        '"${DEFAULT_CI_BASE_DOCKERFILE}" '
        '"${DEFAULT_CI_BASE_DOCKERFILE_STAGES}"'
    )
    files_line, dockerfile_line, stages_line, *selected_lines = contract.splitlines()
    content_paths = files_line.removeprefix("files=").split()
    selected_stages = set(stages_line.removeprefix("stages=").split())
    assert dockerfile_line == "dockerfile=docker/Dockerfile.rocm"
    assert set(content_paths) == {
        ".dockerignore",
        ".buildkite/scripts/ci-bake-rocm.sh",
        ".buildkite/scripts/rocm/build-ci-base.sh",
        ".buildkite/scripts/rocm/cache-utils.sh",
        "docker/ci-rocm.hcl",
        "docker/docker-bake-rocm.hcl",
        "requirements/common.txt",
        "requirements/rocm.txt",
        "requirements/test/rocm.txt",
        "rust-toolchain.toml",
        "tests/vllm_test_utils",
        "tools/install_protoc.sh",
        "tools/install_torchcodec_rocm.sh",
    }
    assert selected_stages == {
        "base",
        "build_deepep",
        "build_nixl",
        "build_rocshmem",
        "ci_base",
        "mori_base",
        "rust-toolchain",
        "rust-toolchain-input",
        "rust_toolchain_input_0",
        "rust_toolchain_input_1",
    }
    selected_instructions = dockerfile_instructions("\n".join(selected_lines))
    all_instructions = dockerfile_instructions(
        (REPO_ROOT / "docker" / "Dockerfile.rocm").read_text()
    )
    all_stages = {
        match.group(1)
        for instruction in all_instructions
        if (match := re.search(r"\s+AS\s+(\S+)$", instruction, re.IGNORECASE))
    }
    referenced_stages: set[str] = set()
    local_sources: set[str] = set()

    for instruction in selected_instructions:
        if instruction.upper().startswith("FROM "):
            base = shlex.split(instruction)[1]
            if base in all_stages:
                referenced_stages.add(base)
            elif "${" in base and not base.startswith("${"):
                prefix, suffix = base.split("${", 1)[0], base.split("}", 1)[1]
                referenced_stages.update(
                    stage
                    for stage in all_stages
                    if stage.startswith(prefix) and stage.endswith(suffix)
                )
        referenced_stages.update(
            ref
            for ref in re.findall(r"(?:--from=|,from=)([^,\s]+)", instruction)
            if ref in all_stages
        )
        instruction_type = instruction.split(maxsplit=1)[0].upper()
        if instruction_type not in {"ADD", "COPY"} or "--from=" in instruction:
            continue
        copy_args = [
            arg for arg in shlex.split(instruction)[1:] if not arg.startswith("--")
        ]
        local_sources.update(copy_args[:-1])

    uncovered_sources = {
        source
        for source in local_sources
        if not any(
            source.rstrip("/") == path.rstrip("/")
            or source.startswith(path.rstrip("/") + "/")
            for path in content_paths
        )
    }
    assert not referenced_stages - selected_stages, (
        f"unhashed ci_base stage dependencies: {referenced_stages - selected_stages}"
    )
    assert not uncovered_sources, (
        f"unhashed ci_base context inputs: {uncovered_sources}"
    )


@pytest.mark.parametrize("script", (CI_BAKE, ROCM_BASE_REFRESH))
def test_image_digest_accepts_pins_and_retries_transient_lookups(
    script: Path,
    tmp_path: Path,
) -> None:
    digest = "sha256:" + "a" * 64
    script_tmp = tmp_path / script.parent.name
    fake_bin, call_log = write_fake_docker(
        script_tmp,
        f"""
count=0
if [[ -f "${{FAKE_DOCKER_CALL_LOG}}" ]]; then
    read -r count < "${{FAKE_DOCKER_CALL_LOG}}"
fi
count=$((count + 1))
printf '%s\\n' "${{count}}" > "${{FAKE_DOCKER_CALL_LOG}}"
if ((count < 3)); then
    echo "transient registry failure ${{count}}" >&2
    exit 42
fi
printf 'Name: rocm/example:base\\nDigest: {digest}\\n'
""",
    )
    result = run_sourced_shell(
        script,
        (
            f'printf "pinned=%s\\n" "$(resolve_image_digest '
            f'"rocm/example:base@{digest}")"\n'
            'printf "resolved=%s\\n" "$(resolve_image_digest '
            '"rocm/example:base")"'
        ),
        env={
            "FAKE_DOCKER_CALL_LOG": str(call_log),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "ROCM_IMAGE_DIGEST_ATTEMPTS": "4",
            "ROCM_IMAGE_DIGEST_RETRY_DELAY": "0",
            "ROCM_IMAGE_DIGEST_RETRY_MAX_DELAY": "0",
        },
    )

    assert result.stdout.splitlines() == [f"pinned={digest}", f"resolved={digest}"]
    assert call_log.read_text().strip() == "3"
    assert "attempt 1/4" in result.stderr
    assert "attempt 2/4" in result.stderr


@pytest.mark.parametrize("script", (CI_BAKE, ROCM_BASE_REFRESH))
def test_image_digest_rejects_invalid_and_terminal_results(
    script: Path,
    tmp_path: Path,
) -> None:
    valid_but_failed = "sha256:" + "b" * 64
    script_tmp = tmp_path / script.parent.name
    fake_bin, call_log = write_fake_docker(
        script_tmp,
        f"""
count=0
if [[ -f "${{FAKE_DOCKER_CALL_LOG}}" ]]; then
    read -r count < "${{FAKE_DOCKER_CALL_LOG}}"
fi
count=$((count + 1))
printf '%s\\n' "${{count}}" > "${{FAKE_DOCKER_CALL_LOG}}"
if ((count == 1)); then
    printf 'Digest: sha256:not-a-digest\\n'
    exit 0
fi
printf 'Digest: {valid_but_failed}\\n'
exit 42
""",
    )
    result = run_sourced_shell(
        script,
        'resolve_image_digest "rocm/example:base"',
        env={
            "FAKE_DOCKER_CALL_LOG": str(call_log),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "ROCM_IMAGE_DIGEST_ATTEMPTS": "2",
            "ROCM_IMAGE_DIGEST_RETRY_DELAY": "0",
            "ROCM_IMAGE_DIGEST_RETRY_MAX_DELAY": "0",
        },
        check=False,
    )

    assert result.returncode == 1
    assert call_log.read_text().strip() == "2"
    assert "after 2 attempts (last exit status 42)" in result.stderr
    assert valid_but_failed in result.stderr


def test_local_and_remote_checkouts_have_the_right_hash_identity(
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

    local_hash = compute_ci_base_hash(dockerfile, content_file, env=inputs)
    relocated_local_hash = compute_ci_base_hash(
        dockerfile,
        content_file,
        env=inputs
        | {
            "VLLM_REPO": "https://github.com/fork/vllm.git",
            "VLLM_BRANCH": "second-commit",
        },
    )
    remote_hash = compute_ci_base_hash(
        dockerfile,
        content_file,
        env=inputs | {"REMOTE_VLLM": "1"},
    )
    relocated_remote_hash = compute_ci_base_hash(
        dockerfile,
        content_file,
        env=inputs
        | {
            "REMOTE_VLLM": "1",
            "VLLM_REPO": "https://github.com/fork/vllm.git",
            "VLLM_BRANCH": "second-commit",
        },
    )
    changed_arg_hash = compute_ci_base_hash(
        dockerfile,
        content_file,
        env=inputs | {"NIXL_BRANCH": "nixl-next"},
    )

    assert relocated_local_hash == local_hash
    assert remote_hash != local_hash
    assert relocated_remote_hash != remote_hash
    assert changed_arg_hash != local_hash


def test_resolved_base_digest_is_shared_by_identity_and_build_override(
    tmp_path: Path,
) -> None:
    docker_dir = tmp_path / "docker"
    docker_dir.mkdir()
    dockerfile = docker_dir / "Dockerfile.rocm"
    dockerfile.write_text(
        "ARG BASE_IMAGE=rocm/example:base\nFROM ${BASE_IMAGE} AS base\n"
    )
    content_file = tmp_path / "content.txt"
    content_file.write_text("cache input\n")
    bake_file = docker_dir / "docker-bake-rocm.hcl"
    bake_file.write_text("")
    override_file = tmp_path / "rocm-arg-override.hcl"
    digest = "sha256:" + "a" * 64
    moving_digest = "sha256:" + "b" * 64
    fake_bin, call_log = write_fake_docker(
        tmp_path,
        f"""
count=0
if [[ -f "${{FAKE_DOCKER_CALL_LOG}}" ]]; then
    read -r count < "${{FAKE_DOCKER_CALL_LOG}}"
fi
count=$((count + 1))
printf '%s\\n' "${{count}}" > "${{FAKE_DOCKER_CALL_LOG}}"
if ((count == 1)); then
    printf 'Digest: {digest}\\n'
else
    printf 'Digest: {moving_digest}\\n'
fi
""",
    )

    output = run_ci_bake_shell(
        (
            f'CI_BASE_DOCKERFILE="{dockerfile}"\n'
            f'CI_BASE_CONTENT_FILES="{content_file}"\n'
            f'VLLM_BAKE_FILE="{bake_file}"\n'
            f'ROCM_ARG_OVERRIDE_PATH="{override_file}"\n'
            'CI_BASE_DOCKERFILE_STAGES="base"\n'
            "prime_base_image_digest_cache >/dev/null\n"
            f'hash_dockerfile_arg_values "{dockerfile}" BASE_IMAGE\n'
            "ci_base_metadata_pairs\n"
            "write_rocm_build_arg_override >/dev/null\n"
            f'cat "{override_file}"'
        ),
        env={
            "FAKE_DOCKER_CALL_LOG": str(call_log),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        },
    )

    assert call_log.read_text().strip() == "1"
    assert f"arg:BASE_IMAGE.digest={digest}" in output
    assert f"vllm.rocm.base_image_digest\t{digest}" in output
    assert f'BASE_IMAGE = "rocm/example:base@{digest}"' in output
    assert moving_digest not in output


@pytest.mark.parametrize(
    ("content_hash", "environment", "expected"),
    [
        (
            "content-hash",
            {
                "BUILDKITE_PULL_REQUEST": "48646",
                "BUILDKITE_BRANCH": "feature",
                "CI_BASE_PUSH_STABLE_TAG": "1",
                "NIGHTLY": "1",
            },
            {
                "status": "ok",
                "primary": "rocm/example:ci_base-content-hash",
                "stable": "",
                "stable_output": "0",
                "stable_candidate": "1",
            },
        ),
        (
            "content-hash",
            {
                "BUILDKITE_PULL_REQUEST": "false",
                "BUILDKITE_BRANCH": "main",
                "NIGHTLY": "1",
            },
            {
                "status": "ok",
                "primary": "rocm/example:ci_base-content-hash",
                "stable": "rocm/example:ci_base",
                "stable_output": "1",
                "stable_candidate": "1",
            },
        ),
        (
            "",
            {"BUILDKITE_PULL_REQUEST": "48646"},
            {"status": "rejected"},
        ),
    ],
)
def test_stable_tag_policy_and_required_content_hash(
    content_hash: str,
    environment: dict[str, str | None],
    expected: dict[str, str],
) -> None:
    result = run_sourced_shell(
        CI_BAKE,
        (
            'TARGET="ci-base-rocm-ci"\n'
            f"CI_BASE_CONTENT_HASH={shlex.quote(content_hash)}\n"
            'CI_BASE_IMAGE_TAG="rocm/example:ci_base"\n'
            "if configure_ci_base_image_refs >/dev/null; then\n"
            '  outputs="$(ci_base_output_refs)"\n'
            '  candidates="$(ci_base_candidate_refs)"\n'
            '  printf "status=ok\\nprimary=%s\\nstable=%s\\n" '
            '"${CI_BASE_IMAGE_TAG}" "${CI_BASE_IMAGE_TAG_STABLE:-}"\n'
            '  grep -qxF "rocm/example:ci_base" <<< "${outputs}" '
            '&& echo "stable_output=1" || echo "stable_output=0"\n'
            '  grep -qxF "rocm/example:ci_base" <<< "${candidates}" '
            '&& echo "stable_candidate=1" || echo "stable_candidate=0"\n'
            "else\n"
            '  echo "status=rejected"\n'
            "fi"
        ),
        env={
            **environment,
            "BUILDKITE_COMMIT": "0123456789abcdef",
        },
    )
    actual = dict(line.split("=", 1) for line in result.stdout.splitlines())

    assert actual == expected


def test_candidate_lookup_pins_multiarch_image_before_validation(
    tmp_path: Path,
) -> None:
    root_digest = "sha256:" + "a" * 64
    child_digest = "sha256:" + "b" * 64
    mutable_ref = "rocm/example:cache"
    root_ref = f"{mutable_ref}@{root_digest}"
    child_ref = f"{mutable_ref}@{child_digest}"
    fake_bin, call_log = write_fake_docker(
        tmp_path,
        f"""
printf '%s\\n' "$*" >> "${{FAKE_DOCKER_CALL_LOG}}"
if [[ "$1 $2" == "manifest inspect" ]]; then
    exit 0
fi
if [[ "$1 $2 $3" != "buildx imagetools inspect" ]]; then
    exit 42
fi
raw=0
image_ref=""
for arg in "$@"; do
    [[ "${{arg}}" == "--raw" ]] && raw=1
    [[ "${{arg}}" == rocm/example:* ]] && image_ref="${{arg}}"
done
if ((raw == 0)); then
    printf 'Name: {mutable_ref}\\nDigest: {root_digest}\\n'
    exit 0
fi
case "${{image_ref}}" in
    "{root_ref}")
        printf '%s\\n' \
            '{{"manifests":[{{"digest":"{child_digest}","platform":{{"os":"linux","architecture":"amd64"}}}}]}}'
        ;;
    "{child_ref}")
        printf '%s\\n' \
            '{{"annotations":{{"vllm.ci_base.content_hash":"expected","vllm.ci_base.metadata_version":"2"}}}}'
        ;;
    *)
        echo "unexpected docker ref: ${{image_ref}}" >&2
        exit 42
        ;;
esac
""",
    )
    output = run_ci_bake_shell(
        (
            'CI_BASE_CONTENT_HASH="expected"\n'
            'CI_BASE_METADATA_VERSION="2"\n'
            f'CI_BASE_STABLE_CACHE_REF="{mutable_ref}"\n'
            "find_matching_ci_base_ref"
        ),
        env={
            "FAKE_DOCKER_CALL_LOG": str(call_log),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        },
    )
    calls = call_log.read_text()

    assert output.strip() == root_ref
    assert child_ref in calls
    assert f"{root_ref}@{child_digest}" not in calls


def test_retag_failure_is_terminal_and_stops_later_updates() -> None:
    result = run_sourced_shell(
        CI_BAKE,
        (
            'CI_BASE_CONTENT_HASH="expected"\n'
            'IMAGE_TAG="rocm/example:commit"\n'
            'CI_BASE_IMAGE_TAG="rocm/example:commit"\n'
            'CI_BASE_IMAGE_TAG_CONTENT_EXTRA="rocm/example:content"\n'
            'get_remote_image_label() { printf "stale\\n"; }\n'
            'docker() { printf "docker:%s\\n" "$*"; return 42; }\n'
            'refresh_ci_base_tags_from_ref "rocm/example:source"'
        ),
        check=False,
    )

    assert result.returncode == 1
    assert "create -t rocm/example:commit" in result.stdout
    assert "create -t rocm/example:content" not in result.stdout
    assert "Failed to update ci_base tag rocm/example:commit" in result.stderr


def test_amd_pipeline_uses_local_inputs_then_remote_commit_checkout() -> None:
    pipeline = yaml.safe_load(AMD_HARDWARE_PIPELINE.read_text())
    steps = {step["key"]: step for step in pipeline["steps"]}
    ensure_env = steps["ensure-ci-base-amd"]["env"]
    image_env = steps["image-build-amd"]["env"]

    assert ensure_env["REMOTE_VLLM"] == "0"
    assert "VLLM_BRANCH" not in ensure_env
    assert image_env["REMOTE_VLLM"] == "1"
    assert image_env["VLLM_BRANCH"] == "$BUILDKITE_COMMIT"
    for step in steps.values():
        assert step["env"]["BUILDKIT_PROGRESS"] == "tty"
        assert step["env"]["BUILDKIT_TTY_LOG_LINES"] == "1"
        assert step["artifact_paths"] == ["build/buildkit-logs/**/*"]


def test_ci_base_handoff_is_content_addressed_across_build_steps(
    tmp_path: Path,
) -> None:
    metadata_file = tmp_path / "buildkite-metadata"
    digest = "sha256:" + "c" * 64
    content_ref = "rocm/example:ci_base-content-hash"
    immutable_ref = f"{content_ref}@{digest}"
    producer = run_sourced_shell(
        CI_BAKE,
        (
            "set_buildkite_metadata() {\n"
            '  printf "%s\\t%s\\n" "$1" "$2" >> "${FAKE_METADATA_FILE}"\n'
            "}\n"
            "set_required_buildkite_metadata() {\n"
            '  printf "%s\\t%s\\n" "$1" "$2" >> "${FAKE_METADATA_FILE}"\n'
            "}\n"
            'TARGET="ci-base-rocm-ci"\n'
            'CI_BASE_CONTENT_HASH="content-hash"\n'
            'CI_BASE_IMAGE_TAG="rocm/example:ci_base"\n'
            "configure_ci_base_image_refs >/dev/null\n"
            "if awk -F '\\t' "
            "'$1 == \"rocm-ci-base-image\" { found=1 } END { exit !found }' "
            '"${FAKE_METADATA_FILE}"; then\n'
            '  echo "mutable handoff published before final validation" >&2\n'
            "  exit 42\n"
            "fi\n"
            f'resolve_image_digest() {{ printf "%s\\n" "{digest}"; }}\n'
            "get_remote_image_label_with_retry() {\n"
            '  printf "%s\\n" "${CI_BASE_CONTENT_HASH}"\n'
            "}\n"
            "remote_ci_base_metadata_is_current_with_retry() { return 0; }\n"
            "publish_ci_base_handoff_ref >/dev/null"
        ),
        env={
            "BUILDKITE_COMMIT": "0123456789abcdef",
            "BUILDKITE_PULL_REQUEST": "48646",
            "BUILDKITE_BRANCH": "feature",
            "FAKE_METADATA_FILE": str(metadata_file),
        },
    )
    producer.check_returncode()

    consumer = run_shell(
        (
            "buildkite-agent() {\n"
            '  [[ "$1 $2" == "meta-data get" ]] || return 42\n'
            "  awk -F '\\t' -v key=\"$3\" '$1 == key { value=$2 } "
            'END { if (value != "") print value }\' "${FAKE_METADATA_FILE}"\n'
            "}\n"
            "bash() {\n"
            '  printf "selected=%s\\n" "${CI_BASE_IMAGE:-}"\n'
            "}\n"
            'source "$1"'
        ),
        ROCM_TEST_IMAGE,
        env={"FAKE_METADATA_FILE": str(metadata_file)},
    )
    selected = next(
        line.removeprefix("selected=")
        for line in consumer.stdout.splitlines()
        if line.startswith("selected=")
    )
    metadata = dict(
        line.split("\t", 1) for line in metadata_file.read_text().splitlines()
    )

    assert metadata["rocm-ci-base-image-content"] == content_ref
    assert metadata["rocm-ci-base-image"] == immutable_ref
    assert selected == immutable_ref


def test_ci_base_handoff_fails_if_required_metadata_write_fails() -> None:
    digest = "sha256:" + "d" * 64
    result = run_sourced_shell(
        CI_BAKE,
        (
            'TARGET="ci-base-rocm-ci"\n'
            'CI_BASE_CONTENT_HASH="content-hash"\n'
            'CI_BASE_IMAGE_TAG_CONTENT_REF="rocm/example:ci_base-content-hash"\n'
            f'resolve_image_digest() {{ printf "%s\\n" "{digest}"; }}\n'
            'get_remote_image_label_with_retry() { printf "content-hash\\n"; }\n'
            "remote_ci_base_metadata_is_current_with_retry() { return 0; }\n"
            "set_required_buildkite_metadata() { return 42; }\n"
            'set_buildkite_metadata() { echo "optional metadata was set"; }\n'
            "if publish_ci_base_handoff_ref; then\n"
            "  exit 0\n"
            "else\n"
            "  rc=$?\n"
            '  exit "${rc}"\n'
            "fi"
        ),
        check=False,
    )

    assert result.returncode == 1
    assert "optional metadata was set" not in result.stdout
    assert "Could not publish required ci_base handoff metadata" in result.stderr


@pytest.mark.parametrize(
    "handoff_ref",
    ("", "rocm/example:ci_base-content-hash"),
)
def test_buildkite_consumer_rejects_missing_or_mutable_handoff(
    handoff_ref: str,
) -> None:
    result = run_shell(
        (
            "buildkite-agent() {\n"
            '  [[ "$1 $2" == "meta-data get" ]] || return 42\n'
            f"  printf '%s\\n' {shlex.quote(handoff_ref)}\n"
            "}\n"
            'bash() { echo "unexpected image build"; }\n'
            'source "$1"'
        ),
        ROCM_TEST_IMAGE,
        env={"BUILDKITE": "true"},
        check=False,
    )

    assert result.returncode == 1
    assert "unexpected image build" not in result.stdout
    assert "Required ROCm ci_base handoff metadata is missing" in result.stderr
