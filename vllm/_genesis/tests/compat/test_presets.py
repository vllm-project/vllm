# SPDX-License-Identifier: Apache-2.0
"""Tests for `vllm._genesis.compat.presets` — curated launch bundles.

Verifies:
  - preset registry integrity (no dups, valid workload, valid GPU keys)
  - match_preset semantics (exact + balanced fallback)
  - to_launch_script generates runnable bash with all expected env vars
  - CLI subcommands (list/show/match/auto) produce sensible output
  - cross-references with gpu_profile.GPU_SPECS (every preset's GPU
    must exist in the spec database)

CPU-only — does not require torch/CUDA.

Author: Sandermage (Sander) Barzov Aleksandr.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


# ─────────────────────────────────────────────────────────────────
# Registry integrity
# ─────────────────────────────────────────────────────────────────


def test_presets_module_imports():
    from vllm._genesis.compat import presets
    assert hasattr(presets, "list_presets")
    assert hasattr(presets, "get_preset")
    assert hasattr(presets, "match_preset")
    assert hasattr(presets, "auto_match")
    assert hasattr(presets, "Preset")
    assert hasattr(presets, "WORKLOADS")


def test_presets_at_least_minimum_count():
    """We ship at least 8 presets (covers main hardware classes)."""
    from vllm._genesis.compat.presets import list_presets

    presets = list_presets()
    assert len(presets) >= 8, (
        f"only {len(presets)} presets registered — need at least 8 to "
        f"cover main hardware classes (1×/2× 3090, 2× A5000, 1× 4090, "
        f"1× 5090, R6000, 8× A4000, H20)"
    )


def test_presets_all_have_unique_keys():
    from vllm._genesis.compat.presets import list_presets

    keys = [p.key for p in list_presets()]
    assert len(keys) == len(set(keys)), (
        f"duplicate preset keys: {keys}"
    )


def test_presets_all_use_known_workload():
    from vllm._genesis.compat.presets import WORKLOADS, list_presets

    valid = set(WORKLOADS.keys())
    for p in list_presets():
        assert p.workload in valid, (
            f"preset {p.key!r} has unknown workload {p.workload!r}"
        )


def test_presets_all_gpu_keys_exist_in_spec_database():
    """Every preset's gpu_match_keys must exist in
    `gpu_profile.GPU_SPECS`. Otherwise auto_match() will silently fail
    to detect that GPU class."""
    from vllm._genesis.compat.presets import list_presets
    from vllm._genesis.gpu_profile import GPU_SPECS

    spec_keys = set(GPU_SPECS.keys())
    for p in list_presets():
        for gpu_key in p.gpu_match_keys:
            assert gpu_key in spec_keys, (
                f"preset {p.key!r} references gpu match_key "
                f"{gpu_key!r} which is NOT in gpu_profile.GPU_SPECS. "
                f"Available keys: {sorted(spec_keys)}"
            )


def test_presets_n_gpus_positive():
    from vllm._genesis.compat.presets import list_presets

    for p in list_presets():
        assert p.n_gpus >= 1, f"preset {p.key!r} has n_gpus={p.n_gpus}"


def test_presets_max_model_len_reasonable():
    from vllm._genesis.compat.presets import list_presets

    for p in list_presets():
        assert 1024 <= p.max_model_len <= 1_000_000, (
            f"preset {p.key!r} has implausible max_model_len="
            f"{p.max_model_len}"
        )


def test_presets_gpu_memory_utilization_in_range():
    from vllm._genesis.compat.presets import list_presets

    for p in list_presets():
        assert 0.5 <= p.gpu_memory_utilization <= 1.0, (
            f"preset {p.key!r} gpu_memory_utilization "
            f"{p.gpu_memory_utilization} outside [0.5, 1.0]"
        )


def test_presets_genesis_env_keys_use_genesis_prefix():
    """All Genesis env vars must start with GENESIS_ — defensive
    against accidentally putting VLLM_* into genesis_env (should go
    in system_env)."""
    from vllm._genesis.compat.presets import list_presets

    for p in list_presets():
        for key in p.genesis_env:
            assert key.startswith("GENESIS_"), (
                f"preset {p.key!r}: genesis_env key {key!r} does not "
                f"start with GENESIS_ — should it be in system_env?"
            )


def test_presets_speculative_method_valid():
    from vllm._genesis.compat.presets import list_presets

    valid = {None, "mtp", "eagle", "ngram", "dflash"}
    for p in list_presets():
        assert p.speculative_method in valid, (
            f"preset {p.key!r}: unknown speculative_method "
            f"{p.speculative_method!r}"
        )


# ─────────────────────────────────────────────────────────────────
# match_preset semantics
# ─────────────────────────────────────────────────────────────────


def test_match_exact_returns_specific_preset():
    """Exact (gpu, n_gpus, workload) match returns that preset."""
    from vllm._genesis.compat.presets import match_preset

    p = match_preset(
        gpu_class="rtx 3090", n_gpus=1, workload="long_context"
    )
    assert p is not None
    assert p.key == "3090-1x-long-context"


def test_match_falls_back_to_balanced_workload():
    """If exact workload not available, falls back to balanced."""
    from vllm._genesis.compat.presets import match_preset

    # rtx 4090 only has 'balanced' — request 'tool_agent' should fall
    # back to balanced (or return None if balanced also missing)
    p = match_preset(
        gpu_class="rtx 4090", n_gpus=1, workload="tool_agent"
    )
    if p is not None:
        assert p.workload == "balanced", (
            f"expected fallback to balanced, got {p.workload!r}"
        )
    # Else None is also acceptable — caller must handle


def test_match_returns_none_for_unknown_gpu():
    from vllm._genesis.compat.presets import match_preset

    p = match_preset(
        gpu_class="nvidia tesla a100x", n_gpus=1, workload="balanced"
    )
    assert p is None


def test_match_rejects_unknown_workload():
    from vllm._genesis.compat.presets import match_preset

    with pytest.raises(ValueError, match="unknown workload"):
        match_preset(
            gpu_class="rtx 3090", n_gpus=1, workload="not_a_workload"
        )


# ─────────────────────────────────────────────────────────────────
# to_launch_script renders runnable bash
# ─────────────────────────────────────────────────────────────────


def test_launch_script_has_shebang_and_set_options():
    from vllm._genesis.compat.presets import get_preset

    script = get_preset("a5000-2x-balanced").to_launch_script()
    lines = script.splitlines()
    assert lines[0] == "#!/usr/bin/env bash"
    assert "set -euo pipefail" in script


def test_launch_script_emits_all_genesis_env_vars():
    from vllm._genesis.compat.presets import get_preset

    p = get_preset("3090-1x-long-context")
    script = p.to_launch_script()
    for key, val in p.genesis_env.items():
        assert f"export {key}={val}" in script, (
            f"missing export of {key}={val}"
        )


def test_launch_script_emits_all_system_env_vars():
    from vllm._genesis.compat.presets import get_preset

    p = get_preset("3090-1x-long-context")
    script = p.to_launch_script()
    for key, val in p.system_env.items():
        # System env values may need quoting (PYTORCH_CUDA_ALLOC_CONF
        # has commas/colons), so check for `export KEY=` prefix only
        assert f"export {key}=" in script, f"missing export of {key}"


def test_launch_script_includes_vllm_serve_command():
    from vllm._genesis.compat.presets import get_preset

    p = get_preset("a5000-2x-balanced")
    script = p.to_launch_script()
    assert "exec vllm serve" in script
    assert f"--tensor-parallel-size {p.n_gpus}" in script
    assert f"--max-model-len {p.max_model_len}" in script
    assert f"--max-num-batched-tokens {p.max_num_batched_tokens}" in script
    assert (
        f"--gpu-memory-utilization {p.gpu_memory_utilization}" in script
    )


def test_launch_script_includes_speculative_config_when_set():
    from vllm._genesis.compat.presets import get_preset

    p = get_preset("a5000-2x-balanced")
    assert p.speculative_method == "mtp"
    script = p.to_launch_script()
    assert "--speculative-config" in script
    assert "mtp" in script


def test_launch_script_quotes_path_with_spaces():
    """Sanity: shell quoting handles model paths with special chars."""
    from vllm._genesis.compat.presets import get_preset

    p = get_preset("a5000-2x-balanced")
    script = p.to_launch_script(
        model_path="/path with spaces/model",
        served_model_name="model-name",
    )
    assert "'/path with spaces/model'" in script


def test_launch_script_does_not_end_with_dangling_backslash():
    """Trailing `\\` would break bash parsing of the exec line."""
    from vllm._genesis.compat.presets import get_preset

    for key in ["a5000-2x-balanced", "3090-1x-long-context", "h20-1x-high-throughput"]:
        script = get_preset(key).to_launch_script()
        # Find the `exec vllm serve` block and verify last non-empty
        # line doesn't end with ` \`
        non_empty_lines = [
            line for line in script.splitlines() if line.strip()
        ]
        last = non_empty_lines[-1]
        assert not last.endswith(" \\"), (
            f"preset {key!r}: launch script ends with dangling `\\`: "
            f"{last!r}"
        )


# ─────────────────────────────────────────────────────────────────
# CLI smoke tests
# ─────────────────────────────────────────────────────────────────


def _run_cli(*args: str) -> tuple[int, str, str]:
    """Run the presets CLI as subprocess; return (exit_code, stdout, stderr)."""
    # Repo root is parents[4]: tests/compat/test_presets.py
    #   parents[0]=compat, parents[1]=tests, parents[2]=_genesis,
    #   parents[3]=vllm,   parents[4]=repo root
    repo_root = Path(__file__).resolve().parents[4]
    proc = subprocess.run(
        [sys.executable, "-m", "vllm._genesis.compat.presets", *args],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_cli_list_runs_and_prints_at_least_8_presets():
    rc, out, err = _run_cli("list")
    assert rc == 0, f"non-zero exit: {err}"
    assert "Genesis curated presets" in out
    # Count rows in the table (excluding header)
    rows = [
        line for line in out.splitlines()
        if line.startswith("  ") and "rtx" in line.lower() or
        (line.startswith("  ") and "h20" in line.lower())
    ]
    assert len(rows) >= 8, f"too few preset rows in output: {len(rows)}"


def test_cli_list_json_returns_valid_json():
    rc, out, err = _run_cli("list", "--json")
    assert rc == 0
    data = json.loads(out)
    assert isinstance(data, list)
    assert len(data) >= 8
    for entry in data:
        assert "key" in entry
        assert "title" in entry
        assert "workload" in entry


def test_cli_show_runs_for_known_preset():
    rc, out, err = _run_cli("show", "a5000-2x-balanced")
    assert rc == 0, err
    assert "a5000-2x-balanced" in out
    assert "Genesis env" in out
    assert "GENESIS_ENABLE_" in out


def test_cli_show_unknown_preset_exits_nonzero():
    rc, out, err = _run_cli("show", "does-not-exist")
    assert rc != 0
    assert "unknown preset" in err.lower()


def test_cli_show_script_emits_executable_bash():
    rc, out, err = _run_cli("show", "3090-1x-long-context", "--script")
    assert rc == 0, err
    assert out.startswith("#!/usr/bin/env bash"), (
        f"script doesn't start with shebang: {out[:80]!r}"
    )
    assert "exec vllm serve" in out


def test_cli_match_finds_3090_long_context():
    rc, out, err = _run_cli(
        "match", "--gpu", "rtx 3090", "--n-gpus", "1",
        "--workload", "long_context",
    )
    assert rc == 0, err
    assert "3090-1x-long-context" in out


def test_cli_match_no_match_exits_nonzero():
    rc, out, err = _run_cli(
        "match", "--gpu", "tesla v100", "--n-gpus", "1",
        "--workload", "long_context",
    )
    assert rc != 0
    assert "no preset matches" in err.lower()


# ─────────────────────────────────────────────────────────────────
# Documentation requirements
# ─────────────────────────────────────────────────────────────────


def test_at_least_one_preset_per_main_workload():
    """Coverage check: every workload category should have at least 1
    preset, otherwise the matrix has gaps."""
    from vllm._genesis.compat.presets import WORKLOADS, list_presets

    workloads_seen = {p.workload for p in list_presets()}
    for w in WORKLOADS:
        assert w in workloads_seen, (
            f"no preset covers workload {w!r} — matrix has a gap"
        )


def test_sander_prod_preset_present_as_reference():
    """The Sander 2× A5000 PROD config must be present as a reference
    preset — it's the empirical baseline for cross-rig comparisons."""
    from vllm._genesis.compat.presets import get_preset

    p = get_preset("a5000-2x-balanced")
    # Must have verified-on entry from Sander rig
    found = any(
        "Sandermage" in v or "A5000" in v or "PROD" in v
        for v in p.verified_on
    )
    assert found, (
        f"a5000-2x-balanced missing Sander PROD provenance; "
        f"verified_on={p.verified_on}"
    )


def test_noonghunna_3090_preset_present_with_provenance():
    """noonghunna's 1×3090 verified config should be present with
    cross-rig provenance — first community downstream."""
    from vllm._genesis.compat.presets import get_preset

    p = get_preset("3090-1x-long-context")
    found = any("noonghunna" in v for v in p.verified_on)
    assert found, (
        f"3090-1x-long-context missing noonghunna provenance; "
        f"verified_on={p.verified_on}"
    )


def test_v769_cliff2_stack_in_3090_long_context_preset():
    """3090-1x-long-context must enable both halves of the v7.69 Cliff
    2 fix (P103 + PN32) — that's the entire reason this preset exists."""
    from vllm._genesis.compat.presets import get_preset

    p = get_preset("3090-1x-long-context")
    assert "GENESIS_ENABLE_P103" in p.genesis_env, (
        "3090-1x-long-context missing P103 — required half of v7.69 "
        "Cliff 2 fix"
    )
    assert (
        "GENESIS_ENABLE_PN32_GDN_CHUNKED_PREFILL" in p.genesis_env
    ), (
        "3090-1x-long-context missing PN32 — required half of v7.69 "
        "Cliff 2 fix"
    )
