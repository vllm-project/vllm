# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import io
import json
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import regex as re

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("hashed", [False, True])
def test_use_existing_torch_removes_complete_requirements(tmp_path, prefix, hashed):
    """Removing torch must not attach its archive hashes to another package."""
    suffix = " \\\n    --hash=sha256:" + "a" * 64 if hashed else ""
    entries = {
        name: f"{name}==1.0.0{suffix}\n"
        for name in ("torch", "torchvision", "torchmetrics", "other")
    }
    requirements = tmp_path / "requirements" / "test.txt"
    requirements.parent.mkdir()
    requirements.write_text("".join(entries.values()))

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/use_existing_torch.py"),
            *(["--prefix"] if prefix else []),
        ],
        cwd=tmp_path,
        check=True,
    )

    expected = (entries["torchmetrics"] if prefix else "") + entries["other"]
    assert requirements.read_text() == expected


@pytest.mark.parametrize(
    "reference",
    [
        "actions/checkout@v7",
        "actions/checkout@main",
        "actions/checkout@1234567",
        "owner/repo/.github/workflows/test.yml@v1",
        "docker://alpine:latest",
    ],
)
def test_action_pin_check_rejects_mutable_references(tmp_path, reference):
    workflow = tmp_path / "workflow.yml"
    workflow.write_text(json.dumps({"jobs": {"test": {"uses": reference}}}))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/pre_commit/check_github_actions_pins.py"),
            str(workflow),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert reference in result.stdout


def test_action_pin_check_accepts_immutable_and_local_actions(tmp_path):
    workflow = tmp_path / "workflow.yml"
    workflow.write_text(
        json.dumps(
            {
                "jobs": {
                    "reuse": {
                        "uses": "owner/repo/.github/workflows/test.yml@" + "a" * 40
                    },
                    "test": {
                        "steps": [
                            {"uses": "actions/checkout@" + "b" * 40},
                            {"uses": "docker://alpine@sha256:" + "c" * 64},
                            {"uses": "./.github/actions/local"},
                            {"run": "echo uses: actions/checkout@main"},
                        ]
                    },
                }
            }
        )
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/pre_commit/check_github_actions_pins.py"),
            str(workflow),
        ],
        check=True,
    )


@pytest.mark.parametrize("dockerfile", ["Dockerfile.rocm", "Dockerfile.rocm_gfx1250"])
@pytest.mark.parametrize("hash_mode", ["valid", "invalid"])
def test_rocm_source_rebuild_verifies_requirement_hashes(
    tmp_path, dockerfile, hash_mode
):
    """Run the Docker extraction/install command against a local source archive."""
    archive = tmp_path / "fastsafetensors-0.0.0.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, content in {
            "setup.py": (
                "from setuptools import setup\n"
                "setup(name='fastsafetensors', version='0.0.0', "
                "py_modules=['fastsafetensors'])\n"
            ),
            "fastsafetensors.py": "SOURCE_REBUILT = True\n",
        }.items():
            data = content.encode()
            info = tarfile.TarInfo(f"fastsafetensors-0.0.0/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if hash_mode == "invalid":
        digest = "0" * 64
    requirement = "fastsafetensors==0.0.0"
    requirement += " \\\n    --hash=sha256:" + "a" * 64
    requirement += " \\\n    --hash=sha256:" + digest
    requirements = tmp_path / "rocm-test-reqs.txt"
    requirements.write_text(f"other==1.0\n{requirement}\n# via test\nafter==2.0\n")
    target = tmp_path / "installed"
    docker = (ROOT / "docker" / dockerfile).read_text()
    match = re.search(
        r"RUN (.*?)(?:\n\n|$)",
        docker.split("# Rebuild fastsafetensors from source", 1)[1],
        re.DOTALL,
    )
    assert match is not None
    command = re.sub(r"--mount=\S+\s*\\\n\s*", "", match.group(1))
    command = command.replace("/r/test.txt", str(requirements))
    command = command.replace("/tmp/", str(tmp_path) + "/")
    command = command.replace(
        "python3 -m pip install",
        f"{shlex.quote(sys.executable)} -m pip --isolated install "
        "--no-index --no-build-isolation "
        f"--find-links {shlex.quote(str(tmp_path))} "
        f"--target {shlex.quote(str(target))}",
    )
    result = subprocess.run(["sh", "-c", command], capture_output=True, text=True)
    if hash_mode == "invalid":
        assert result.returncode != 0
        assert "DO NOT MATCH THE HASHES" in result.stderr
        assert not (target / "fastsafetensors.py").exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert (target / "fastsafetensors.py").read_text() == "SOURCE_REBUILT = True\n"
