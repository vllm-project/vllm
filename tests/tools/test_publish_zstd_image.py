# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUBLISH = ROOT / ".buildkite/scripts/publish-release-images.sh"
HELPER = ROOT / ".buildkite/scripts/publish-zstd-image.sh"
DIGEST = "sha256:" + "a" * 64
SOURCE = f"localhost:5000/vllm@{DIGEST}"
DESTINATION = "vllm/vllm-openai:v1.2.3-x86_64-zstd"


def fake_environment(
    root: Path, metadata: str = "false"
) -> tuple[dict[str, str], Path]:
    bin_dir = root / "bin"
    bin_dir.mkdir()
    log = root / "docker.log"
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$DOCKER_LOG"\n'
        'if [ "$1 $2 $3" = "buildx imagetools inspect" ]; then\n'
        f"  printf '\"{DIGEST}\"\\n'\n"
        'elif [ "$1 $2" = "buildx create" ]; then\n'
        '  printf "zstd-test-builder\\n"\n'
        'elif [ "$1 $2" = "buildx build" ]; then\n'
        '  cat > "$DOCKERFILE_LOG"\n'
        '  exit "${BUILD_EXIT:-0}"\n'
        'elif [ "$1 $2 $3" = "buildx rm --force" ]; then\n'
        '  exit "${RM_EXIT:-0}"\n'
        "fi\n"
    )
    docker.chmod(0o755)
    agent = bin_dir / "buildkite-agent"
    agent.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        f'  *release-version*) printf "1.2.3\\n" ;;\n'
        f'  *publish-zstd-image*) printf "{metadata}\\n"; exit "${{META_EXIT:-0}}" ;;\n'
        "esac\n"
    )
    agent.chmod(0o755)
    aws = bin_dir / "aws"
    aws.write_text("#!/bin/sh\nprintf password\\n\n")
    aws.chmod(0o755)
    env = os.environ.copy()
    env.update(
        PATH=f"{bin_dir}:{env['PATH']}",
        DOCKER_LOG=str(log),
        DOCKERFILE_LOG=str(root / "Dockerfile"),
        BUILDKITE_COMMIT="commit123",
    )
    return env, log


def run(
    script: Path, *args: str, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script), *args], env=env, cwd=ROOT, text=True, capture_output=True
    )


def test_publish_zstd_image_behavior() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        env, log = fake_environment(root)
        assert run(PUBLISH, "cuda-13-0", env=env).returncode == 0
        assert "buildx" not in log.read_text()

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        env, log = fake_environment(root, "true")
        assert run(PUBLISH, "cuda-12-9", env=env).returncode == 0
        assert "buildx" not in log.read_text()
        assert run(PUBLISH, "cuda-13-0", env=env).returncode == 0
        calls = log.read_text()
        assert "buildx imagetools inspect" in calls
        assert f"type=image,name={DESTINATION},push=true,compression=zstd" in calls
        assert "--load" not in calls
        assert f"push {DESTINATION}" not in calls
        assert (
            f"FROM public.ecr.aws/q9t5s3a7/vllm-release-repo@{DIGEST}"
            in (root / "Dockerfile").read_text()
        )

    invalid = (
        (),
        ("repo:tag", DESTINATION),
        (SOURCE, "vllm/vllm-openai:v1.2.3"),
        (SOURCE, "vllm/repo-zstd"),
        (SOURCE, DESTINATION + ",other"),
        (SOURCE + "\n", DESTINATION),
        (SOURCE, DESTINATION, "extra"),
    )
    for args in invalid:
        with tempfile.TemporaryDirectory() as directory:
            env, log = fake_environment(Path(directory))
            assert run(HELPER, *args, env=env).returncode != 0
            assert not log.exists()

    outcomes = (("0", "0", 0), ("17", "0", 17), ("0", "9", 1), ("17", "9", 17))
    for build_exit, rm_exit, expected in outcomes:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env, log = fake_environment(root)
            env["BUILD_EXIT"] = build_exit
            env["RM_EXIT"] = rm_exit
            result = run(HELPER, SOURCE, DESTINATION, env=env)
            assert result.returncode == expected
            cleanup_calls = log.read_text().splitlines()
            assert sum("buildx create" in call for call in cleanup_calls) == 1
            assert [call for call in cleanup_calls if "buildx rm" in call] == [
                "buildx rm --force zstd-test-builder"
            ]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        env, log = fake_environment(root, "true")
        env["META_EXIT"] = "8"
        assert run(PUBLISH, "cuda-13-0", env=env).returncode == 8
        assert "buildx imagetools" not in log.read_text()


if __name__ == "__main__":
    test_publish_zstd_image_behavior()
