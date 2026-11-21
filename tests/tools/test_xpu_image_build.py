# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import regex as re

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".buildkite/image_build/image_build_xpu.sh"
DIGEST = "sha256:" + "a" * 64
OTHER_DIGEST = "sha256:" + "b" * 64


@pytest.fixture
def build_context(tmp_path: Path) -> Path:
    (tmp_path / "docker").mkdir()
    shutil.copyfile(ROOT / "docker/Dockerfile.xpu", tmp_path / "docker/Dockerfile.xpu")

    tools = tmp_path / "bin"
    tools.mkdir()
    (tools / "aws").write_text("#!/bin/sh\nexit 0\n")
    (tools / "sleep").write_text('#!/bin/sh\necho "sleep $*" >> "$CALLS"\n')
    (tmp_path / "calls").touch()
    (tools / "docker").write_text(
        """#!/bin/bash
echo "$*" >> "$CALLS"
if [[ $1 == buildx && $2 == imagetools && $3 == inspect ]]; then
  if [[ $4 == *:xpu-base-* ]]; then
    attempt=$(grep -c '^buildx imagetools inspect .*:xpu-base-' "$CALLS")
    if [[ $attempt -le $FAIL_BASE_ATTEMPTS ]]; then
      echo "503 Service Unavailable" >&2
      exit 1
    fi
    [[ $BASE_EXISTS == 1 || -f $PUSHED ]] || { echo "manifest unknown" >&2; exit 1; }
  else
    [[ $FINAL_EXISTS == 1 ]] || { echo "manifest unknown" >&2; exit 1; }
  fi
  if [[ -f $PUSHED ]]; then
    echo "Digest: $OTHER_DIGEST"
  else
    echo "Digest: $DIGEST"
  fi
elif [[ $1 == buildx && $2 == build ]]; then
  [[ $FAIL_BASE_BUILD != 1 ]] || { echo "base build failed" >&2; exit 1; }
  touch "$PUSHED"
  while [[ $# -gt 0 ]]; do
    if [[ $1 == --metadata-file ]]; then
      if [[ $PUSH_NO_DIGEST == 1 ]]; then
        echo '{}' > "$2"
      else
        printf '{"containerimage.descriptor":{"digest":"%s"},\\n' "$OTHER_DIGEST" > "$2"
        printf '"containerimage.digest":"%s"\\n}\\n' "$DIGEST" >> "$2"
      fi
      break
    fi
    shift
  done
elif [[ $1 == build && $FAIL_FINAL_BUILD == 1 ]]; then
  echo "final build failed" >&2
  exit 1
fi
"""
    )
    for tool in tools.iterdir():
        tool.chmod(0o755)
    annotate = tmp_path / ".buildkite/scripts/annotate-image-build.sh"
    annotate.parent.mkdir(parents=True)
    annotate.write_text("#!/bin/sh\nexit 0\n")
    annotate.chmod(0o755)
    return tmp_path


def run_build(
    context: Path, **env: str
) -> tuple[subprocess.CompletedProcess[str], str]:
    run_env = {
        **os.environ,
        "PATH": f"{context / 'bin'}:{os.environ['PATH']}",
        "CALLS": str(context / "calls"),
        "PUSHED": str(context / "pushed"),
        "DIGEST": DIGEST,
        "OTHER_DIGEST": OTHER_DIGEST,
        "BASE_EXISTS": "0",
        "FINAL_EXISTS": "0",
        "FAIL_BASE_ATTEMPTS": "0",
        "FAIL_BASE_BUILD": "0",
        "FAIL_FINAL_BUILD": "0",
        "PUSH_NO_DIGEST": "0",
        "BUILDKITE_PULL_REQUEST": "42",
        "BUILDKITE_BRANCH": "feature",
        "BUILDKITE_PIPELINE_DEFAULT_BRANCH": "main",
        **env,
    }
    result = subprocess.run(
        ["bash", str(SCRIPT), "registry.example", "ci", "commit"],
        cwd=context,
        env=run_env,
        capture_output=True,
        text=True,
    )
    return result, (context / "calls").read_text()


def test_base_hit_is_pinned_and_skips_base_build(build_context: Path) -> None:
    result, calls = run_build(build_context, BASE_EXISTS="1")
    assert result.returncode == 0, result.stderr
    assert (
        "--set vllm-openai.contexts.vllm-base=docker-image://registry.example/ci:"
        in calls
    )
    assert f"@{DIGEST}" in calls
    assert "buildx build" not in calls
    assert "buildx bake --file docker/docker-bake-xpu.hcl --load" in calls
    assert "--set vllm-openai.tags=registry.example/ci:commit-xpu" in calls
    assert "push registry.example/ci:commit-xpu" in calls


def test_existing_final_image_skips_base_lookup(build_context: Path) -> None:
    result, calls = run_build(build_context, FINAL_EXISTS="1")
    assert result.returncode == 0, result.stderr
    assert "xpu-base-" not in calls
    assert "buildx bake" not in calls
    assert "\nbuild " not in calls


@pytest.mark.parametrize("base_exists", ["0", "1"])
def test_main_checks_base_even_when_final_image_exists(
    build_context: Path, base_exists: str
) -> None:
    result, calls = run_build(
        build_context,
        FINAL_EXISTS="1",
        BASE_EXISTS=base_exists,
        BUILDKITE_PULL_REQUEST="false",
        BUILDKITE_BRANCH="main",
    )
    assert result.returncode == 0, result.stderr
    assert "buildx imagetools inspect registry.example/ci:xpu-base-" in calls
    assert ("buildx build" in calls) == (base_exists == "0")
    assert "buildx bake" not in calls
    assert "\nbuild " not in calls
    assert "push registry.example/ci:commit-xpu" not in calls


@pytest.mark.parametrize(
    ("pull_request", "branch"),
    [("42", "feature"), ("false", "feature"), ("42", "main"), ("", "main")],
)
def test_base_miss_builds_locally_for_untrusted_builds(
    build_context: Path, pull_request: str, branch: str
) -> None:
    result, calls = run_build(
        build_context, BUILDKITE_PULL_REQUEST=pull_request, BUILDKITE_BRANCH=branch
    )
    assert result.returncode == 0, result.stderr
    assert "vllm-openai.contexts.vllm-base=" not in calls
    assert "buildx build" not in calls
    assert calls.count("buildx imagetools inspect registry.example/ci:xpu-base-") == 1
    assert (
        "build --file docker/Dockerfile.xpu --build-arg max_jobs=16 "
        "--build-arg buildkite_commit=commit --tag registry.example/ci:commit-xpu "
        "--progress plain ."
    ) in calls
    assert "buildx bake" not in calls
    assert "push registry.example/ci:commit-xpu" in calls


def test_local_build_failure_stops_push(build_context: Path) -> None:
    result, calls = run_build(build_context, FAIL_FINAL_BUILD="1")
    assert result.returncode != 0
    assert "final build failed" in result.stderr
    assert "push registry.example/ci:commit-xpu" not in calls


def test_pushed_base_uses_its_digest_not_a_later_tag_lookup(
    build_context: Path,
) -> None:
    result, calls = run_build(
        build_context,
        BUILDKITE_PULL_REQUEST="false",
        BUILDKITE_BRANCH="main",
    )
    assert result.returncode == 0, result.stderr
    assert "buildx build --file docker/Dockerfile.xpu --target vllm-base" in calls
    assert "--tag registry.example/ci:xpu-base-" in calls
    assert "--push --metadata-file" in calls
    assert f"@{DIGEST}" in calls
    assert f"@{OTHER_DIGEST}" not in calls


def test_base_build_failure_stops_final_build(build_context: Path) -> None:
    result, calls = run_build(
        build_context,
        BUILDKITE_PULL_REQUEST="false",
        BUILDKITE_BRANCH="main",
        FAIL_BASE_BUILD="1",
    )
    assert result.returncode != 0
    assert "base build failed" in result.stderr
    assert "buildx bake" not in calls
    assert "\nbuild " not in calls


def test_missing_pushed_digest_fails_without_building_final_image(
    build_context: Path,
) -> None:
    result, calls = run_build(
        build_context,
        BUILDKITE_PULL_REQUEST="false",
        BUILDKITE_BRANCH="main",
        PUSH_NO_DIGEST="1",
    )
    assert result.returncode != 0
    assert "Cannot resolve the pushed XPU base image digest" in result.stderr
    assert "buildx bake" not in calls
    assert "\nbuild " not in calls


def test_registry_failure_is_not_a_cache_miss(build_context: Path) -> None:
    result, calls = run_build(build_context, FAIL_BASE_ATTEMPTS="3")
    assert result.returncode != 0
    assert "503 Service Unavailable" in result.stderr
    assert calls.count("buildx imagetools inspect registry.example/ci:xpu-base-") == 3
    assert calls.count("sleep 5\n") == 2
    assert "buildx bake" not in calls
    assert "\nbuild " not in calls


def test_transient_registry_failure_is_retried(build_context: Path) -> None:
    result, calls = run_build(build_context, BASE_EXISTS="1", FAIL_BASE_ATTEMPTS="2")
    assert result.returncode == 0, result.stderr
    assert calls.count("buildx imagetools inspect registry.example/ci:xpu-base-") == 3
    assert calls.count("sleep 5\n") == 2
    assert f"@{DIGEST}" in calls
    assert "buildx bake" in calls


def test_missing_base_stage_fails_before_registry_lookup(build_context: Path) -> None:
    dockerfile = build_context / "docker/Dockerfile.xpu"
    dockerfile.write_text(dockerfile.read_text().replace("AS vllm-base", "AS other"))
    result, calls = run_build(build_context, BASE_EXISTS="1")
    assert result.returncode != 0
    assert "Missing vllm-base stage" in result.stderr
    assert "xpu-base-" not in calls


def test_base_key_ignores_application_changes(build_context: Path) -> None:
    result, calls = run_build(build_context, BASE_EXISTS="1")
    assert result.returncode == 0, result.stderr
    first = re.search(r"xpu-base-[0-9a-f]+", calls).group()
    (build_context / "calls").unlink()
    with (build_context / "docker/Dockerfile.xpu").open("a") as input_file:
        input_file.write("\n# changed\n")
    result, calls = run_build(build_context, BASE_EXISTS="1")
    assert result.returncode == 0, result.stderr
    assert re.search(r"xpu-base-[0-9a-f]+", calls).group() == first


def test_refresh_changes_base_key(build_context: Path) -> None:
    result, calls = run_build(build_context, BASE_EXISTS="1")
    assert result.returncode == 0, result.stderr
    first = re.search(r"xpu-base-[0-9a-f]+", calls).group()
    (build_context / "calls").unlink()
    dockerfile = build_context / "docker/Dockerfile.xpu"
    original = dockerfile.read_text()
    updated, count = re.subn(
        r"(?m)^ARG REFRESH=\d{4}-\d{2}-\d{2}$", "ARG REFRESH=2099-12-31", original
    )
    assert count == 1 and updated != original
    dockerfile.write_text(updated)
    result, calls = run_build(build_context, BASE_EXISTS="1")
    assert result.returncode == 0, result.stderr
    assert re.search(r"xpu-base-[0-9a-f]+", calls).group() != first


def test_global_arg_changes_base_key(build_context: Path) -> None:
    dockerfile = build_context / "docker/Dockerfile.xpu"
    original = dockerfile.read_text()
    content = original.replace(
        "FROM ubuntu:24.04 AS vllm-base", "FROM ${BASE_IMAGE} AS vllm-base"
    )
    assert content != original
    dockerfile.write_text("ARG BASE_IMAGE=ubuntu:24.04\n" + content)
    result, calls = run_build(build_context, BASE_EXISTS="1")
    assert result.returncode == 0, result.stderr
    first = re.search(r"xpu-base-[0-9a-f]+", calls).group()
    (build_context / "calls").unlink()
    dockerfile.write_text(
        dockerfile.read_text().replace(
            "BASE_IMAGE=ubuntu:24.04", "BASE_IMAGE=ubuntu:22.04"
        )
    )
    result, calls = run_build(build_context, BASE_EXISTS="1")
    assert result.returncode == 0, result.stderr
    assert re.search(r"xpu-base-[0-9a-f]+", calls).group() != first


def test_base_key_covers_context_sources() -> None:
    dockerfile = (ROOT / "docker/Dockerfile.xpu").read_text()
    base = re.search(
        r"(?ms)^FROM [^\n]+ AS vllm-base[ \t]*$.*?(?=^FROM |\Z)", dockerfile
    ).group()
    base = base.replace("\\\n", " ")
    assert not re.search(r"^\s*(COPY|ADD)\s", base, re.MULTILINE)
    sources = set()
    for options in re.findall(r"--mount=(\S+)", base):
        opts = dict(option.partition("=")[::2] for option in options.split(","))
        if opts.get("type") == "bind" and "from" not in opts:
            sources.add(opts.get("source", opts.get("src", ".")))
    assert not sources, "Base context inputs must be included in the base key"
