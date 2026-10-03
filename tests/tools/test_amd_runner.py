# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Legacy Docker jobs must return their recorder output to the agent checkout."""

import os
import subprocess
from pathlib import Path

RUNNER = (
    Path(__file__).resolve().parents[2]
    / ".buildkite/scripts/hardware_ci/run-amd-test.sh"
)


def _recorder_args(checkout: Path, command: str):
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; configure_kernrec_docker_args "$2"; '
            'printf "%s\\0" "${kernrec_docker_args[@]}"',
            "_",
            str(RUNNER),
            command,
        ],
        env={**os.environ, "BUILDKITE_BUILD_CHECKOUT_PATH": str(checkout)},
        check=True,
        capture_output=True,
    )
    return [arg.decode() for arg in result.stdout.split(b"\0") if arg]


def test_recorder_mount_and_identity_reach_the_inner_container(tmp_path):
    checkout = tmp_path / "checkout with spaces"
    checkout.mkdir()
    args = _recorder_args(checkout, "curl recorders/kernrec/ci_setup.sh && pytest")
    assert args[:4] == [
        "-v",
        f"{checkout}/.kernrec:/tmp/kernrec-checkout/.kernrec",
        "-e",
        "KERNREC_CHECKOUT_PATH=/tmp/kernrec-checkout",
    ]
    assert set(args[5::2]) == {
        "BUILDKITE_JOB_ID",
        "BUILDKITE_STEP_KEY",
        "BUILDKITE_LABEL",
        "BUILDKITE_BUILD_NUMBER",
        "BUILDKITE_COMMIT",
        "KERNREC_PYTHON",
        "VLLM_WORKER_MULTIPROC_METHOD",
        "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS",
    }
    assert (checkout / ".kernrec").stat().st_mode & 0o777 == 0o777


def test_unavailable_checkout_does_not_block_the_workload(tmp_path):
    assert _recorder_args(tmp_path / "missing", "recorders/kernrec/ci_setup.sh") == []
    assert not (tmp_path / "missing").exists()


def test_unrecorded_jobs_do_not_create_a_recorder_mount(tmp_path):
    assert _recorder_args(tmp_path, "pytest") == []
    assert not (tmp_path / ".kernrec").exists()
