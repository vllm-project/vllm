# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static consistency check between the pinned zentorch and torch versions.

zentorch (the AMD Zen CPU optimization extra) tracks the torch release that
vLLM depends on, differing only by an extra trailing "patch level" segment.
Concretely, torch ``X.Y.Z`` corresponds to zentorch ``X.Y.Z.W``.

This test parses the raw text of ``setup.py`` and ``requirements/cpu.txt`` and
string-matches the versions.
"""

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

REPO_ROOT = Path(__file__).resolve().parent.parent
SETUP_PY = REPO_ROOT / "setup.py"
# zentorch is the AMD Zen *CPU* extra, so the CPU requirement file is the
# authoritative source for the torch version it must track.
CPU_REQUIREMENTS = REPO_ROOT / "requirements" / "cpu.txt"


def _extract_zentorch_version() -> str:
    """Pull the pinned zentorch version out of setup.py's extras_require."""
    text = SETUP_PY.read_text()
    matches = re.findall(r'zentorch==([0-9][0-9A-Za-z.\-+]*)', text)
    assert matches, f"No 'zentorch==<version>' pin found in {SETUP_PY}"
    assert len(set(matches)) == 1, (
        f"Multiple conflicting zentorch pins found in {SETUP_PY}: {matches}"
    )
    return matches[0]


def _extract_torch_version() -> str:
    """Pull the pinned torch version out of requirements/cpu.txt.

    The local version label (e.g. the ``+cpu`` in ``2.13.0+cpu``) is stripped
    so only the upstream release ``X.Y.Z`` is compared.
    """
    text = CPU_REQUIREMENTS.read_text()
    # Match lines like: torch==2.13.0+cpu; platform_machine == "x86_64" ...
    # but not torchaudio / torchvision / open-clip-torch etc.
    matches = re.findall(
        r'(?m)^\s*torch==([0-9][0-9A-Za-z.\-+]*)', text
    )
    assert matches, f"No 'torch==<version>' pin found in {CPU_REQUIREMENTS}"
    # Strip local version labels (+cpu, +cu130, ...) and keep the release only.
    releases = {m.split("+", 1)[0] for m in matches}
    assert len(releases) == 1, (
        f"Multiple conflicting torch release pins found in {CPU_REQUIREMENTS}: "
        f"{sorted(releases)}"
    )
    return releases.pop()


def test_zentorch_pin_tracks_torch_release():
    zentorch_version = _extract_zentorch_version()
    torch_version = _extract_torch_version()

    zentorch_parts = zentorch_version.split(".")
    torch_parts = torch_version.split(".")

    # torch is X.Y.Z; zentorch is X.Y.Z.W (one extra trailing patch-level
    # segment). Enforce both the shape and the shared prefix.
    assert len(torch_parts) == 3, (
        f"Expected torch release of the form 'X.Y.Z', got {torch_version!r}"
    )
    assert len(zentorch_parts) == 4, (
        f"Expected zentorch pin of the form 'X.Y.Z.W', got "
        f"{zentorch_version!r}"
    )
    assert zentorch_parts[:3] == torch_parts, (
        "zentorch version is out of sync with torch. "
        f"setup.py pins zentorch=={zentorch_version} (base "
        f"{'.'.join(zentorch_parts[:3])}) but requirements/cpu.txt pins "
        f"torch=={torch_version}. When bumping torch, bump the first three "
        "components of the zentorch pin in setup.py to match."
    )
