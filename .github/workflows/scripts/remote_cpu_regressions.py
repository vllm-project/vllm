# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Select reviewed CPU regressions by the features present in a stacked PR."""

import argparse
import subprocess
import sys
from pathlib import Path


def regression_nodes() -> list[str]:
    source = Path("vllm/v1/spec_decode/remote")
    tests = "tests/v1/spec_decode/remote"
    nodes = []
    groups = (
        (
            source / "token_session.py",
            "test_token_runtime.py",
            (
                "test_token_cleanup_error_preserves_failure_policy_and_terminal_state",
                "test_token_cleanup_error_finishes_cancelled_handle",
                "test_token_server_socket_close_error_releases_connection_admission",
            ),
        ),
        (
            source / "target_kv_session.py",
            "test_target_kv_wire.py",
            (
                "test_cleanup_error_preserves_failure_policy_and_terminal_state",
                "test_cleanup_error_finishes_cancelled_handle",
                "test_server_close_error_returns_connection_and_sequence_credit",
            ),
        ),
        (
            source / "runtime/recurrent_gpu.py",
            "test_token_gpu.py",
            ("test_recurrent_startup_reads_each_checkpoint_once_and_rechecks_changes",),
        ),
        (
            source / "runtime/recurrent_gpu.py",
            "test_token_runtime.py",
            (
                "test_token_recurrent_draft_validates_effective_public_ssm_dtype",
                "test_token_recurrent_draft_rejects_resolved_bfloat16_ssm_dtype",
            ),
        ),
        (
            source / "runtime/quantization.py",
            "test_mtp_contracts.py",
            ("test_qwen35_fp8_native_load_requires_projection_and_scale",),
        ),
    )
    for feature, test_file, names in groups:
        if feature.is_file():
            nodes.extend(f"{tests}/{test_file}::{name}" for name in names)
    if (source / "runtime/tp_executor.py").is_file() and (
        source / "runtime/token_cache.py"
    ).is_file():
        nodes.extend(
            f"{tests}/test_token_runtime.py::{name}"
            for name in (
                "test_public_outcome_cache_rejects_draft_tp2",
                "test_token_cli_rejects_outcome_cache_with_draft_tp_before_loading",
            )
        )
    return nodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="Print selected pytest nodes"
    )
    args = parser.parse_args()
    nodes = regression_nodes()
    if args.list:
        print("\n".join(nodes))
    elif nodes:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--noconftest",
                "-p",
                "pytest_timeout",
                "-q",
                *nodes,
            ],
            check=True,
        )
    else:
        print("This branch has no token, target-KV, recurrent, or FP8 runtime yet.")
