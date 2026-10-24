# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parents[2]
TRITON_WHEELS = [
    f"triton-3.8.0+cpu-{tag}-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    for tag in ("cp311-cp311", "cp312-abi3")
]


@pytest.mark.parametrize("version", ["a" * 40, "0.32.0", "0.32.0rc1"])
def test_cpu_dependency_remains_in_commit_and_release_indices(tmp_path, version):
    """Release filtering must not discard the CPU wheel's dependency."""
    commit_dir = tmp_path / ("a" * 40)
    index_dir = tmp_path / version
    vllm_wheel = "vllm-0.32.0+cpu-cp38-abi3-manylinux_2_34_x86_64.whl"
    xpu_wheel = "triton-3.8.0+xpu-py3-none-any.whl"
    objects = tmp_path / "objects.json"
    objects.write_text(
        json.dumps(
            {
                "Contents": [
                    {"Key": f"{commit_dir.name}/{name}"}
                    for name in [vllm_wheel, xpu_wheel, *TRITON_WHEELS]
                ]
            }
        )
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / ".buildkite/scripts/generate-nightly-index.py"),
            "--version",
            version,
            "--wheel-dir",
            commit_dir.name,
            "--current-objects",
            str(objects),
            "--output-dir",
            str(index_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    links = unquote((index_dir / "cpu/triton/index.html").read_text())
    metadata = json.loads((index_dir / "cpu/triton/metadata.json").read_text())
    assert {entry["filename"] for entry in metadata} == set(TRITON_WHEELS)
    for wheel in TRITON_WHEELS:
        assert f'/{wheel}"' in links
    assert xpu_wheel not in links
    assert (index_dir / "xpu/triton/index.html").exists()


@pytest.mark.parametrize("failure", [None, "checksum", "http", "missing-pin"])
def test_stage_all_python_wheels_only_after_verifying_downloads(tmp_path, failure):
    """A failure on the second wheel must prevent every S3 upload."""
    payload = b"fixture wheel bytes"
    digest = hashlib.sha256(payload).hexdigest()
    requirements = tmp_path / "requirements"
    requirements.mkdir()
    urls = []
    for i, wheel in enumerate(TRITON_WHEELS):
        checksum = "0" * 64 if i == 1 and failure == "checksum" else digest
        fragment = "" if i == 1 and failure == "missing-pin" else f"#sha256={checksum}"
        urls.append(f"triton @ https://github.com/fixture/{wheel}{fragment}")
    (requirements / "cpu.txt").write_text("\n".join(urls))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl = bin_dir / "curl"
    curl.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\nfrom pathlib import Path\n"
        "if os.environ['FAILURE'] == 'http' and 'cp312' in sys.argv[-1]:\n"
        "    sys.exit(22)\n"
        f"Path(sys.argv[sys.argv.index('--output') + 1]).write_bytes({payload!r})\n"
    )
    aws = bin_dir / "aws"
    aws.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\nfrom pathlib import Path\n"
        "assert sys.argv[1:4] == ['s3', 'cp', '--recursive']\n"
        "Path('uploads.json').write_text(json.dumps({\n"
        "    'files': sorted(p.name for p in Path(sys.argv[4]).iterdir()),\n"
        "    'target': sys.argv[5]}))\n"
    )
    curl.chmod(0o755)
    aws.chmod(0o755)
    result = subprocess.run(
        ["bash", str(ROOT / ".buildkite/scripts/stage-cpu-triton-wheels.sh")],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "BUILDKITE_COMMIT": "a" * 40,
            "FAILURE": failure or "",
        },
        capture_output=True,
        text=True,
        timeout=15,
    )
    uploads = tmp_path / "uploads.json"
    if failure:
        assert result.returncode != 0
        assert not uploads.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert json.loads(uploads.read_text()) == {
            "files": sorted(TRITON_WHEELS),
            "target": f"s3://vllm-wheels/{'a' * 40}/",
        }


@pytest.mark.parametrize(
    "version,arch,needs_triton",
    [
        ("0.32.0+cpu", "x86_64", True),
        ("0.32.1.dev1+gabcdef.cpu", "x86_64", True),
        ("0.32.0+cpu", "aarch64", False),
        ("0.32.0", "x86_64", False),
    ],
)
def test_uploader_stages_dependency_before_x86_cpu_wheel(
    tmp_path, version, arch, needs_triton
):
    """Release and nightly builds must both publish the dependency first."""
    import zipfile

    dist = tmp_path / "artifacts/dist"
    dist.mkdir(parents=True)
    wheel = dist / f"vllm-{version}-cp38-abi3-manylinux_2_34_{arch}.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("vllm.dist-info/METADATA", f"Version: {version}\n")
    scripts = tmp_path / ".buildkite/scripts"
    scripts.mkdir(parents=True)
    (scripts / "lib").symlink_to(ROOT / ".buildkite/scripts/lib")
    # The stage command's downloads and checksum validation are exercised above.
    (scripts / "stage-cpu-triton-wheels.sh").write_text("echo dependency >> calls\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Exercise the uploader's real control flow; avoid launching auditwheel Docker.
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/bin/bash\n"
        'case "$1" in\n'
        "  run) echo fixture-container ;;\n"
        '  exec) [[ "${*: -1}" == *.whl ]] && echo "${*: -1}"; exit 0 ;;\n'
        "esac\n"
    )
    aws = bin_dir / "aws"
    aws.write_text("#!/bin/sh\necho vllm >> calls\n")
    unzip = bin_dir / "unzip"
    unzip.write_text(
        f"#!{sys.executable}\n"
        "import sys, zipfile\n"
        "with zipfile.ZipFile(sys.argv[2]) as archive:\n"
        "    path = next(n for n in archive.namelist() if n.endswith('/METADATA'))\n"
        "    sys.stdout.write(archive.read(path).decode())\n"
    )
    docker.chmod(0o755)
    aws.chmod(0o755)
    unzip.chmod(0o755)
    result = subprocess.run(
        ["bash", str(ROOT / ".buildkite/scripts/upload-nightly-wheels.sh")],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "BUILDKITE_COMMIT": "a" * 40,
            "VLLM_WHEEL_PLATFORM": "linux",
        },
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    expected = ["dependency", "vllm"] if needs_triton else ["vllm"]
    assert (tmp_path / "calls").read_text().splitlines() == expected
