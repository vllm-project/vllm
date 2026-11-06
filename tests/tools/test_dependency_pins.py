# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import hashlib
import importlib.util
import io
import json
import shlex
import subprocess
import sys
import tarfile
import types
from pathlib import Path

import pytest
import regex as re
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]


def test_docs_mock_versions_are_valid_with_hashed_requirements():
    """Exercise the docs version loader without importing the CLI generators."""
    path = ROOT / "docs/mkdocs/gen_files/generate_argparse.py"
    tree = ast.parse(path.read_text())
    loader = next(
        node
        for node in tree.body
        if isinstance(node, ast.With)
        and any(
            isinstance(child, ast.Name) and child.id == "VERSIONS"
            for child in ast.walk(node)
        )
    )
    namespace: dict = {"ROOT_DIR": ROOT}
    exec(
        compile(ast.Module(body=[loader], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    versions = namespace["VERSIONS"]
    lock = (ROOT / "requirements/test/cuda.txt").read_text()
    assert len(versions) == len(re.findall(r"^\S+==", lock, re.MULTILINE))
    for version in versions.values():
        Version(version)


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
    command = command.replace("/tmp/", str(tmp_path) + "/")
    command = command.replace("/r/test.txt", str(requirements))
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


def _load_hybrid_test_deps_check():
    path = ROOT / ".buildkite/scripts/check-hybrid-test-deps.py"
    spec = importlib.util.spec_from_file_location("check_hybrid_test_deps", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hybrid_test_deps_pins_agree_and_build_tools_are_hashed():
    """The CI-side check must pin the commits the image build fetches."""
    check = _load_hybrid_test_deps_check()
    build_script = (ROOT / "tools/build_hybrid_test_wheels.sh").read_text()
    assert f"mamba_commit={check.EXPECTED_COMMITS['mamba_ssm']}" in build_script
    assert (
        f"causal_conv1d_commit={check.EXPECTED_COMMITS['causal_conv1d']}"
        in build_script
    )
    lock = (ROOT / "requirements/build/hybrid-test-deps.txt").read_text()
    requirement_blocks = [
        block for block in re.split(r"\n(?=\S)", lock) if not block.startswith("#")
    ]
    assert len(requirement_blocks) == 4
    for block in requirement_blocks:
        assert re.match(r"\S+==\S+ \\\n", block), block
        assert "--hash=sha256:" in block, block


@pytest.mark.parametrize(
    "installed_versions, commits_text, expected_problems",
    [
        ({"mamba_ssm": "2.3.0", "causal_conv1d": "1.6.0"}, "MATCHING", []),
        (
            {"mamba_ssm": None, "causal_conv1d": "1.6.0"},
            "MATCHING",
            ["mamba_ssm: installed None, expected 2.3.0"],
        ),
        (
            {"mamba_ssm": "2.3.0", "causal_conv1d": "1.5.0"},
            "MATCHING",
            ["causal_conv1d: installed 1.5.0, expected 1.6.0"],
        ),
        (
            {"mamba_ssm": "2.3.0", "causal_conv1d": "1.6.0"},
            None,
            ["/opt/hybrid-test-deps-commits.txt is missing"],
        ),
        (
            {"mamba_ssm": "2.3.0", "causal_conv1d": "1.6.0"},
            "mamba_ssm deadbeef\n"
            "causal_conv1d da6dbaa9fd5a919967f14d3fd031da1288ad5025\n",
            ["built from"],
        ),
    ],
)
def test_hybrid_test_deps_check_reports_mismatches(
    installed_versions, commits_text, expected_problems
):
    check = _load_hybrid_test_deps_check()
    if commits_text == "MATCHING":
        commits_text = "".join(
            f"{package} {commit}\n"
            for package, commit in check.EXPECTED_COMMITS.items()
        )
    problems = check.find_problems(installed_versions, commits_text)
    assert len(problems) == len(expected_problems)
    for problem, expected in zip(problems, expected_problems):
        assert problem.startswith(expected)


def test_hybrid_test_deps_check_exits_with_every_problem(tmp_path, monkeypatch):
    check = _load_hybrid_test_deps_check()

    def fake_version(package):
        if package == "mamba_ssm":
            raise check.PackageNotFoundError(package)
        return "1.6.0"

    monkeypatch.setattr(check, "version", fake_version)
    monkeypatch.setattr(check, "COMMITS_FILE", tmp_path / "missing.txt")
    with pytest.raises(SystemExit) as exit_info:
        check.main()
    assert str(exit_info.value) == (
        "The test image's hybrid test dependencies do not match the pins in "
        "tools/build_hybrid_test_wheels.sh (arm64 and Rubin images skip them, and "
        "a failed torch-nightly build warns in the image build log and leaves "
        "them out):\n"
        "mamba_ssm: installed None, expected 2.3.0\n"
        f"{tmp_path / 'missing.txt'} is missing"
    )


def test_hybrid_test_deps_check_prints_verified_pins(tmp_path, monkeypatch, capsys):
    check = _load_hybrid_test_deps_check()
    commits_file = tmp_path / "commits.txt"
    commits_file.write_text(
        "".join(
            f"{package} {commit}\n"
            for package, commit in check.EXPECTED_COMMITS.items()
        )
    )
    monkeypatch.setattr(check, "COMMITS_FILE", commits_file)
    monkeypatch.setattr(check, "version", check.EXPECTED_VERSIONS.__getitem__)
    for package, package_version in check.EXPECTED_VERSIONS.items():
        fake_module = types.ModuleType(package)
        fake_module.__version__ = package_version  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, package, fake_module)
    check.main()
    assert capsys.readouterr().out == (
        "Verified hybrid test dependencies: "
        "mamba_ssm==2.3.0@f1493ff6e9335160eb134eb67e59f8e4d9adefd6 "
        "causal_conv1d==1.6.0@da6dbaa9fd5a919967f14d3fd031da1288ad5025\n"
    )


def test_hybrid_test_deps_check_script_runs_the_check():
    """Running the script checks the image, whichever image this runs in."""
    result = subprocess.run(
        [sys.executable, ROOT / ".buildkite/scripts/check-hybrid-test-deps.py"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        assert result.stdout.startswith("Verified hybrid test dependencies:")
    else:
        assert result.returncode == 1
        assert "do not match the pins" in result.stderr
