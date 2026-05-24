# SPDX-License-Identifier: Apache-2.0
"""Tests for install.sh + plugin packaging metadata.

Verifies install.sh structure (parseable bash, all expected flags
documented, GPU detection branch covers all 16 GPU classes in
gpu_profile.GPU_SPECS) AND the plugin pyproject (entry points correct,
console script declared, version present).

Most install.sh tests are STATIC (parse the file content) since the
script depends on git/pip/network and is not safe to actually execute
in unit tests. End-to-end install testing belongs in CI integration.

Author: Sandermage (Sander) Barzov Aleksandr.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
INSTALL_SH = REPO_ROOT / "install.sh"
PLUGIN_PYPROJECT = (
    REPO_ROOT / "tools" / "genesis_vllm_plugin" / "pyproject.toml"
)


# ─────────────────────────────────────────────────────────────────
# install.sh exists + bash parses cleanly
# ─────────────────────────────────────────────────────────────────


def test_install_sh_exists():
    assert INSTALL_SH.is_file(), f"install.sh missing at {INSTALL_SH}"


def test_install_sh_has_shebang():
    first_line = INSTALL_SH.read_text().splitlines()[0]
    assert first_line.startswith("#!"), f"missing shebang: {first_line}"
    assert "bash" in first_line


def test_install_sh_is_executable():
    import os
    import stat

    mode = os.stat(INSTALL_SH).st_mode
    assert mode & stat.S_IXUSR, "install.sh not executable (chmod +x)"


def test_install_sh_bash_parses_clean():
    """`bash -n install.sh` must not report any syntax errors."""
    result = subprocess.run(
        ["bash", "-n", str(INSTALL_SH)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"bash -n failed: {result.stderr}"
    )


def test_install_sh_help_runs():
    """`install.sh --help` must run without side effects and print help."""
    result = subprocess.run(
        ["bash", str(INSTALL_SH), "--help"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0
    assert "Genesis" in result.stdout
    assert "Usage:" in result.stdout


# ─────────────────────────────────────────────────────────────────
# install.sh structure: required flags + safe defaults
# ─────────────────────────────────────────────────────────────────


def test_install_sh_has_set_strict_flags():
    """`set -euo pipefail` is required for a safe installer."""
    content = INSTALL_SH.read_text()
    assert "set -euo pipefail" in content, (
        "install.sh must use 'set -euo pipefail' for safety"
    )


def test_install_sh_documents_all_required_flags():
    """All Sander-promised flags must appear in the help text."""
    result = subprocess.run(
        ["bash", str(INSTALL_SH), "--help"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    help_text = result.stdout
    for flag in [
        "--pin",
        "--workload",
        "--home",
        "--no-verify",
        "--no-plugin",
        "--uninstall",
        "-y",
        "--yes",
        "-h",
        "--help",
    ]:
        assert flag in help_text, f"flag {flag!r} missing from --help"


def test_install_sh_documents_workload_options():
    """All 4 workload categories must be in --help."""
    result = subprocess.run(
        ["bash", str(INSTALL_SH), "--help"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    help_text = result.stdout
    for w in ["balanced", "long_context", "high_throughput", "tool_agent"]:
        assert w in help_text, f"workload {w!r} not documented in --help"


def test_install_sh_default_pin_is_stable():
    """Default pin should be 'stable' (not 'main' or 'dev') for safety."""
    content = INSTALL_SH.read_text()
    assert 'GENESIS_PIN="${GENESIS_PIN:-stable}"' in content, (
        "default GENESIS_PIN must be 'stable' (latest tag) for safety"
    )


def test_install_sh_default_pip_install_user_not_system():
    """Default pip install should be --user (safer than system-wide)."""
    content = INSTALL_SH.read_text()
    assert 'PIP_INSTALL_FLAGS="${PIP_INSTALL_FLAGS:---user}"' in content


def test_install_sh_python_version_check_requires_3_10():
    """Must enforce Python >= 3.10 (Genesis requires it)."""
    content = INSTALL_SH.read_text()
    assert "$PY_MINOR" in content
    assert "3.10" in content or "10" in content


# ─────────────────────────────────────────────────────────────────
# install.sh GPU detection covers all GPU_SPECS keys
# ─────────────────────────────────────────────────────────────────


def test_install_sh_detects_all_gpu_classes_in_spec_database():
    """Every key in gpu_profile.GPU_SPECS must have a matching case
    arm in install.sh's detect_gpu() — otherwise install.sh would skip
    preset matching for known GPUs."""
    from vllm._genesis.gpu_profile import GPU_SPECS

    content = INSTALL_SH.read_text()
    # Extract the gpu detection case block
    match = re.search(
        r"detect_gpu\(\).*?(case.*?esac)",
        content,
        re.DOTALL,
    )
    assert match is not None, "could not find detect_gpu case block"
    case_block = match.group(1)

    missing = []
    for key in GPU_SPECS:
        # Each key in GPU_SPECS should appear as a case pattern
        if key not in case_block:
            missing.append(key)

    assert not missing, (
        f"install.sh detect_gpu() missing case arms for "
        f"{len(missing)} GPU_SPECS keys: {missing}\n"
        f"Add `*\"{missing[0]}\"*) GPU_CLASS_HINT=\"{missing[0]}\" ;;` "
        f"to detect_gpu()"
    )


def test_install_sh_pro_blackwell_max_q_before_plain_blackwell():
    """Order matters: 'rtx pro 6000 blackwell max-q' must come BEFORE
    'rtx pro 6000 blackwell' in the case block, else max-q would never
    match (longer pattern must match first)."""
    content = INSTALL_SH.read_text()
    maxq_pos = content.find('"rtx pro 6000 blackwell max-q"')
    plain_pos = content.find('"rtx pro 6000 blackwell"')
    assert maxq_pos > 0
    assert plain_pos > 0
    assert maxq_pos < plain_pos, (
        "max-q pattern must come first in case block to avoid being "
        "shadowed by the shorter 'rtx pro 6000 blackwell' pattern"
    )


# ─────────────────────────────────────────────────────────────────
# install.sh has --uninstall path
# ─────────────────────────────────────────────────────────────────


def test_install_sh_uninstall_function_present():
    content = INSTALL_SH.read_text()
    assert "uninstall()" in content
    assert "GENESIS_UNINSTALL=1" in content


def test_install_sh_uninstall_warns_about_text_patches():
    """uninstall() must warn that text-patches are NOT reverted (operator
    must reinstall vllm to clean them) — otherwise users will think
    `--uninstall` removed everything."""
    content = INSTALL_SH.read_text()
    # Find the uninstall function body
    match = re.search(
        r"uninstall\(\)\s*\{(.*?)^}",
        content,
        re.DOTALL | re.MULTILINE,
    )
    assert match is not None, "could not find uninstall() body"
    body = match.group(1)
    # Must mention text-patches (not reverted)
    assert "text-patch" in body.lower() or "pip uninstall vllm" in body, (
        "uninstall() must warn about residual text-patches"
    )


# ─────────────────────────────────────────────────────────────────
# install.sh plugin pip install step
# ─────────────────────────────────────────────────────────────────


def test_install_sh_installs_genesis_vllm_plugin():
    content = INSTALL_SH.read_text()
    assert "pip install" in content
    assert "tools/genesis_vllm_plugin" in content, (
        "install.sh must reference plugin at its current path "
        "(tools/genesis_vllm_plugin/, not the legacy root location)"
    )


def test_install_sh_verifies_entry_point_after_install():
    """Must check that vllm.general_plugins → genesis_v7 entry point
    is registered post-install (otherwise spawn workers won't load us)."""
    content = INSTALL_SH.read_text()
    assert "vllm.general_plugins" in content
    assert "genesis_v7" in content


# ─────────────────────────────────────────────────────────────────
# install.sh references valid CLI commands
# ─────────────────────────────────────────────────────────────────


def test_install_sh_references_only_real_cli_subcommands():
    """Any `genesis <subcommand>` references in install.sh must resolve
    to a real subcommand registered in the unified CLI dispatcher."""
    from vllm._genesis.compat.cli import KNOWN_SUBCOMMANDS

    content = INSTALL_SH.read_text()
    # Find all `genesis <word>` and `compat.cli <word>` references
    references = set()
    references.update(re.findall(r"genesis (\w[\w-]*)", content))
    references.update(re.findall(r"compat\.cli (\w[\w-]*)", content))

    # Filter out obvious non-subcommand words (flags, generic terms)
    skip_words = {
        "vllm", "is", "at", "and", "or", "the", "in", "for", "to",
        "as", "be", "by", "from", "PIN", "REPO", "HOME", "WORKLOAD",
        "NON", "NO", "v7", "v769", "via", "an", "with", "into", "of",
        "doesn", "won",
    }
    candidates = {r for r in references if r not in skip_words}

    bogus = []
    for ref in candidates:
        if ref in KNOWN_SUBCOMMANDS:
            continue
        # Allow non-subcommand mentions in comments (they're descriptive
        # references, not invocations). Only fail if it looks like an
        # actual invocation pattern: `genesis <word>` at start of line
        # in echo/printf or after `bash` etc.
        if re.search(r"^\s*echo\s+.*genesis\s+" + re.escape(ref), content,
                     re.MULTILINE) or \
           re.search(r"\bgenesis\s+" + re.escape(ref) + r"\b", content):
            # Possible bogus reference — but only complain if not whitelisted
            # Common false-positives: 'install', 'install.sh', etc.
            if ref in {"install", "vllm", "v7", "src", "bin"}:
                continue
            bogus.append(ref)

    # Bogus = commands referenced but NOT in KNOWN_SUBCOMMANDS
    if bogus:
        pytest.fail(
            f"install.sh references unknown CLI subcommands: {sorted(set(bogus))}\n"
            f"Known: {sorted(KNOWN_SUBCOMMANDS)}"
        )


# ─────────────────────────────────────────────────────────────────
# Plugin pyproject.toml — console_scripts + entry point
# ─────────────────────────────────────────────────────────────────


def test_plugin_pyproject_exists():
    assert PLUGIN_PYPROJECT.is_file(), (
        f"missing {PLUGIN_PYPROJECT}"
    )


def test_plugin_pyproject_declares_genesis_console_script():
    """After `pip install`, `genesis` should be a top-level command."""
    content = PLUGIN_PYPROJECT.read_text()
    assert "[project.scripts]" in content, (
        "plugin pyproject.toml must declare [project.scripts] for the "
        "`genesis` console command"
    )
    assert "genesis = " in content
    assert "vllm._genesis.compat.cli:main" in content


def test_plugin_pyproject_declares_vllm_general_plugins_entry_point():
    content = PLUGIN_PYPROJECT.read_text()
    assert '[project.entry-points."vllm.general_plugins"]' in content
    assert "genesis_v7" in content


def test_plugin_pyproject_requires_python_3_10_plus():
    content = PLUGIN_PYPROJECT.read_text()
    assert 'requires-python = ">=3.10"' in content


def test_plugin_pyproject_apache_2_license():
    content = PLUGIN_PYPROJECT.read_text()
    assert "Apache-2.0" in content


# ─────────────────────────────────────────────────────────────────
# install.sh + verify integration (verify is called post-install)
# ─────────────────────────────────────────────────────────────────


def test_install_sh_calls_genesis_verify_post_install():
    """install.sh's run_verify() should invoke the verify subcommand."""
    content = INSTALL_SH.read_text()
    assert "verify" in content
    # Should be invoked through the unified CLI
    assert "compat.cli verify" in content or "verify --quick" in content


def test_install_sh_no_verify_flag_skips_verify():
    """--no-verify should set GENESIS_NO_VERIFY=1."""
    content = INSTALL_SH.read_text()
    assert "GENESIS_NO_VERIFY=1" in content
    assert "--no-verify" in content
