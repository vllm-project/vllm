# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / ".buildkite/scripts/build-rocm-base-wheels.sh"
CACHE_TOOL = "cache-rocm-base-wheels.sh"
PARENT_DIGEST = f"sha256:{'1' * 64}"
CACHED_DIGEST = f"sha256:{'2' * 64}"
BUILT_DIGEST = f"sha256:{'3' * 64}"
ECR_REPOSITORY = "public.ecr.aws/q9t5s3a7/vllm-release-repo"

# Run the release shell without a registry, object store, or container runtime.
FAKE_TOOL = r"""
import json
import os
import sys
from pathlib import Path

tool = Path(sys.argv[0]).name
args = sys.argv[1:]
env = os.environ
with open(env["CALL_LOG"], "a") as log:
    log.write(json.dumps({"tool": tool, "args": args, "env": {
        key: value for key, value in env.items()
        if key.startswith(("ROCM_BASE_", "SCCACHE_"))
        or key in ("PYTORCH_ROCM_ARCH", "DOCKER_BUILDKIT")
    }}) + "\n")
if env.get("FAIL_COMMAND") == f"{tool}:{args[0]}":
    sys.exit(17)
if tool == "cache-rocm-base-wheels.sh":
    if args == ["parent"]:
        print("rocm/base:test@sha256:" + "1" * 64)
    elif args == ["key"]:
        print("test-cache-key")
    elif args == ["check"]:
        print("miss" if env.get("ROCM_BASE_WHEEL_CACHE_FORCE") == "1"
              else env.get("CACHE_STATUS", "miss"))
    elif args == ["download"]:
        Path(env["ROCM_BASE_IMAGE_DIGEST_FILE"]).write_text(
            env.get("CACHED_DIGEST", "sha256:" + "2" * 64))
        Path("artifacts/rocm-base-wheels/cached.whl").write_text("cached")
elif tool == "aws":
    print("fake-login-password")
elif tool == "docker":
    if args[0] == "login":
        assert sys.stdin.read().strip() == "fake-login-password"
    elif args[:2] == ["manifest", "inspect"]:
        sys.exit(int(env.get("MISSING_IMAGE", "0")))
    elif args[:2] == ["buildx", "build"] and "--metadata-file" in args:
        Path(args[args.index("--metadata-file") + 1]).write_text(json.dumps({
            "containerimage.digest": env.get("BUILD_DIGEST", "sha256:" + "3" * 64)
        }))
    elif args[0] == "create":
        print("wheel-container")
    elif args[0] == "cp":
        Path(args[-1], "built.whl").write_text("built")
"""


@pytest.fixture
def release_build(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    scripts = tmp_path / ".buildkite/scripts"
    for directory in (fake_bin, scripts, tmp_path / "docker", tmp_path / "temp"):
        directory.mkdir(parents=True)
    for tool in [scripts / CACHE_TOOL] + [
        fake_bin / name for name in ("aws", "docker", "buildkite-agent")
    ]:
        tool.write_text(f"#!{sys.executable}\n{FAKE_TOOL}")
        tool.chmod(0o755)
    (tmp_path / "docker/Dockerfile.rocm_base").write_text(
        'ARG PYTORCH_ROCM_ARCH="gfx90a;gfx942"\n'
        "ARG SCCACHE_VERSION=v0.10.0\n"
        f"ARG SCCACHE_DOWNLOAD_SHA256={'4' * 64}\n"
    )
    wheels = tmp_path / "artifacts/rocm-base-wheels"
    wheels.mkdir(parents=True)
    (wheels / "stale.whl").touch()

    def run(**overrides: str) -> tuple[subprocess.CompletedProcess[str], list[dict]]:
        result = subprocess.run(
            ["bash", str(HELPER)],
            cwd=tmp_path,
            env={
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "TMPDIR": str(tmp_path / "temp"),
                "CALL_LOG": str(tmp_path / "calls.jsonl"),
                "BUILDKITE_BUILD_NUMBER": "42",
                **overrides,
            },
            capture_output=True,
            text=True,
        )
        log = tmp_path / "calls.jsonl"
        calls = (
            [json.loads(line) for line in log.read_text().splitlines()]
            if log.exists()
            else []
        )
        assert not list((tmp_path / "temp").iterdir()), result.stderr
        return result, calls

    return run


def commands(calls: list[dict], tool: str, *prefix: str) -> list[dict]:
    return [
        call
        for call in calls
        if call["tool"] == tool and call["args"][: len(prefix)] == list(prefix)
    ]


def build_args(call: dict) -> dict[str, str]:
    args = call["args"]
    return dict(
        args[i + 1].split("=", 1)
        for i, arg in enumerate(args[:-1])
        if arg == "--build-arg"
    )


def test_full_cache_hit_publishes_paired_digest_without_building(release_build):
    result, calls = release_build(CACHE_STATUS="hit")

    assert result.returncode == 0, result.stderr
    assert commands(calls, "aws")[0]["args"] == [
        "ecr-public",
        "get-login-password",
        "--region",
        "us-east-1",
    ]
    assert commands(calls, "docker", "login")[0]["args"] == [
        "login",
        "--username",
        "AWS",
        "--password-stdin",
        "public.ecr.aws/q9t5s3a7",
    ]
    assert not commands(calls, "docker", "buildx", "build")
    assert not commands(calls, CACHE_TOOL, "upload")
    assert commands(calls, "docker", "manifest", "inspect")[0]["args"][-1] == (
        f"{ECR_REPOSITORY}@{CACHED_DIGEST}"
    )
    assert commands(calls, "buildkite-agent")[0]["args"] == [
        "meta-data",
        "set",
        "rocm-base-image-tag",
        f"{ECR_REPOSITORY}@{CACHED_DIGEST}",
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"CACHE_STATUS": "hit", "ROCM_BASE_WHEEL_CACHE_FORCE": "1"},
        {"CACHE_STATUS": "hit", "FAIL_COMMAND": f"{CACHE_TOOL}:download"},
        {"CACHE_STATUS": "hit", "MISSING_IMAGE": "1"},
    ],
)
def test_rebuild_pairs_image_and_wheels_with_identical_inputs(
    release_build, tmp_path: Path, overrides: dict[str, str]
):
    result, calls = release_build(**overrides)

    assert result.returncode == 0, result.stderr
    image, wheels = commands(calls, "docker", "buildx", "build")
    expected = {
        "BASE_IMAGE": f"rocm/base:test@{PARENT_DIGEST}",
        "USE_SCCACHE": "1",
        "PYTORCH_ROCM_ARCH": "gfx90a;gfx942",
        "SCCACHE_VERSION": "v0.10.0",
        "SCCACHE_DOWNLOAD_SHA256": "4" * 64,
        "SCCACHE_ENDPOINT": "",
        "SCCACHE_BUCKET_NAME": "vllm-build-sccache",
        "SCCACHE_REGION_NAME": "us-west-2",
        "SCCACHE_S3_NO_CREDENTIALS": "0",
    }
    assert build_args(image) == build_args(wheels) == expected
    for build, target in ((image, "final"), (wheels, "debs_wheel_release")):
        args = build["args"]
        assert args[args.index("--target") + 1] == target
        assert args[args.index("--platform") + 1] == "linux/amd64"
        assert args[args.index("--file") + 1] == "docker/Dockerfile.rocm_base"
        assert build["env"]["DOCKER_BUILDKIT"] == "1"
    assert "--push" in image["args"] and "--load" in wheels["args"]
    assert ("--no-cache" in image["args"]) == (
        overrides.get("ROCM_BASE_WHEEL_CACHE_FORCE") == "1"
    )
    assert "--no-cache" not in wheels["args"]
    assert (
        commands(calls, CACHE_TOOL, "upload")[0]["env"]["ROCM_BASE_IMAGE_DIGEST"]
        == BUILT_DIGEST
    )
    assert commands(calls, "buildkite-agent")[0]["args"][-1] == (
        f"{ECR_REPOSITORY}@{BUILT_DIGEST}"
    )
    assert [p.name for p in (tmp_path / "artifacts/rocm-base-wheels").iterdir()] == [
        "built.whl"
    ]


@pytest.mark.parametrize(
    "failure", ["docker:buildx", "docker:cp", f"{CACHE_TOOL}:upload"]
)
def test_failure_prevents_publishing_and_cleans_resources(release_build, failure: str):
    result, calls = release_build(FAIL_COMMAND=failure)

    assert result.returncode == 17, result.stderr
    assert not commands(calls, "buildkite-agent")
    if failure == "docker:cp":
        assert commands(calls, "docker", "rm")[0]["args"] == [
            "rm",
            "-f",
            "wheel-container",
        ]
        assert not commands(calls, CACHE_TOOL, "upload")


def test_invalid_build_digest_stops_before_wheel_export(release_build):
    result, calls = release_build(BUILD_DIGEST="mutable-tag")

    assert result.returncode == 1
    assert "invalid ROCm base image digest" in result.stderr
    assert len(commands(calls, "docker", "buildx", "build")) == 1
    assert not commands(calls, CACHE_TOOL, "upload")
    assert not commands(calls, "buildkite-agent")


def test_unexpected_cached_image_ref_is_not_published(release_build):
    result, calls = release_build(CACHE_STATUS="hit", CACHED_DIGEST="mutable-tag")

    assert result.returncode == 1
    assert "refusing unexpected ROCm base image ref" in result.stderr
    assert not commands(calls, "buildkite-agent")


def test_overrides_remain_literal_and_release_defaults_stay_pinned(
    release_build, tmp_path: Path
):
    arch = "gfx942;$(touch injected); gfx90a"
    result, calls = release_build(
        ROCM_BASE_PYTORCH_ROCM_ARCH=arch,
        PYTORCH_ROCM_ARCH="ignored",
        SCCACHE_ENDPOINT="http://localhost:9000",
        SCCACHE_VERSION="v1.2.3",
        SCCACHE_DOWNLOAD_SHA256="5" * 64,
        SCCACHE_DOWNLOAD_URL="https://user:secret@example.test/download",
        SCCACHE_BUCKET_NAME="ignored",
        SCCACHE_REGION_NAME="ignored",
        SCCACHE_S3_NO_CREDENTIALS="1",
        ROCM_BASE_PARENT_DIGEST="ignored",
    )

    assert result.returncode == 0, result.stderr
    for build in commands(calls, "docker", "buildx", "build"):
        args = build_args(build)
        assert args["PYTORCH_ROCM_ARCH"] == arch
        assert args["SCCACHE_ENDPOINT"] == "http://localhost:9000"
        assert args["SCCACHE_VERSION"] == "v1.2.3"
        assert args["SCCACHE_DOWNLOAD_SHA256"] == "5" * 64
        assert args["SCCACHE_BUCKET_NAME"] == "vllm-build-sccache"
        assert args["SCCACHE_REGION_NAME"] == "us-west-2"
        assert args["SCCACHE_S3_NO_CREDENTIALS"] == "0"
        assert "SCCACHE_DOWNLOAD_URL" not in args
    parent = commands(calls, CACHE_TOOL, "parent")
    assert len(parent) == 1
    assert "ROCM_BASE_PARENT_DIGEST" not in parent[0]["env"]
    key_env = commands(calls, CACHE_TOOL, "key")[0]["env"]
    assert key_env["ROCM_BASE_PARENT_DIGEST"] == PARENT_DIGEST
    assert key_env["PYTORCH_ROCM_ARCH"] == arch
    assert "SCCACHE_DOWNLOAD_URL" not in key_env
    assert not (tmp_path / "injected").exists()
    assert "secret" not in result.stdout + result.stderr


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://user:secret@example.test",
        "https://example.test?secret=token",
        "https://example.test#secret",
        "https://example.test secret",
        "https://example.test\nsecret",
        "https://example.test\rsecret",
        "https://example.test\tsecret",
    ],
)
def test_private_endpoint_is_rejected_before_external_calls(release_build, endpoint):
    result, calls = release_build(SCCACHE_ENDPOINT=endpoint)

    assert result.returncode == 1
    assert "SCCACHE_ENDPOINT must not contain" in result.stderr
    assert "secret" not in result.stdout + result.stderr
    assert not calls


def test_invalid_force_value_stops_before_external_calls(release_build):
    result, calls = release_build(ROCM_BASE_WHEEL_CACHE_FORCE="yes")

    assert result.returncode == 1
    assert "ROCM_BASE_WHEEL_CACHE_FORCE must be 0 or 1" in result.stderr
    assert not calls
