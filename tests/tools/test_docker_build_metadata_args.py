# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / ".buildkite" / "scripts" / "docker-build-metadata-args.sh"
CACHE_HELPER = REPO_ROOT / ".buildkite" / "scripts" / "resolve-image-cache.py"


@pytest.fixture
def cache_history(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    git_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Cache test",
        "GIT_AUTHOR_EMAIL": "cache@example.com",
        "GIT_COMMITTER_NAME": "Cache test",
        "GIT_COMMITTER_EMAIL": "cache@example.com",
    }
    for index in range(7):
        subprocess.run(
            ["git", "commit", "-q", "--allow-empty", "-m", str(index)],
            cwd=repo,
            env=git_env,
            check=True,
        )
    commits = subprocess.check_output(
        ["git", "rev-list", "HEAD"], cwd=repo, text=True
    ).splitlines()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    aws = fake_bin / "aws"
    aws.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, sys\n"
        "pathlib.Path(os.environ['AWS_ARGS']).write_text(' '.join(sys.argv[1:]))\n"
        "print(os.environ['AWS_RESULT'])\n"
        "sys.exit(int(os.environ.get('AWS_EXIT', '0')))\n"
    )
    aws.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "BUILDKITE_BRANCH": "main",
        "AWS_ARGS": str(tmp_path / "aws-args"),
        "AWS_RESULT": "[]",
    }
    return repo, commits, env


def test_cache_lookup_uses_nearest_published_ancestors(cache_history):
    """An unpublished parent must not hide completed ancestors in a merge burst."""
    repo, commits, env = cache_history
    env["AWS_RESULT"] = json.dumps(list(reversed(commits[2:])))
    result = subprocess.check_output(
        [sys.executable, str(CACHE_HELPER)], cwd=repo, env=env, text=True
    )
    assert [line.rsplit(":", 1)[1] for line in result.splitlines()] == commits[2:5]
    requested = Path(env["AWS_ARGS"]).read_text()
    assert commits[0] not in requested and commits[1] not in requested
    assert all(commit in requested for commit in commits[2:])


@pytest.mark.parametrize(
    "response,exit_code", [("[]", "0"), ("invalid", "0"), ("", "1")]
)
def test_unavailable_ancestor_cache_does_not_fail_build(
    cache_history, response, exit_code
):
    repo, _, env = cache_history
    env.update(AWS_RESULT=response, AWS_EXIT=exit_code)
    result = subprocess.run(
        [sys.executable, str(CACHE_HELPER)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout == ""


def test_pr_build_does_not_query_main_ancestor_cache(cache_history):
    repo, _, env = cache_history
    env["BUILDKITE_BRANCH"] = "contributor:feature"
    result = subprocess.check_output(
        [sys.executable, str(CACHE_HELPER)], cwd=repo, env=env, text=True
    )
    assert result == ""
    assert not Path(env["AWS_ARGS"]).exists()


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
