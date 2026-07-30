# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gzip
import os
import subprocess
from pathlib import Path

HELPER = (
    Path(__file__).parents[2]
    / ".buildkite"
    / "scripts"
    / "rocm"
    / "buildx-history-logs.sh"
)
FAKE_DOCKER = r"""#!/usr/bin/env bash
set -euo pipefail
if [[ "$1 $2 $3" == "buildx history ls" ]]; then
    echo "builder/node/stale"
    if [[ -f "${AFTER_SNAPSHOT}" ]]; then
        echo "builder/node/new-a"
        echo "builder/node/new-b"
    fi
elif [[ "$*" == *" history logs "*"failed-export" ]]; then
    exit 41
elif [[ "$*" == *" history logs "* ]]; then
    echo "detailed log: $*"
else
    exit 42
fi
"""


def run_helper(
    tmp_path: Path, metadata: str, command: str
) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "docker").write_text(FAKE_DOCKER)
    (fake_bin / "docker").chmod(0o755)
    (fake_bin / "buildkite-agent").write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$BUILDKITE_CALLS"\n'
    )
    (fake_bin / "buildkite-agent").chmod(0o755)
    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text(metadata)
    env = os.environ | {
        "AFTER_SNAPSHOT": str(tmp_path / "after"),
        "BUILDKITE_CALLS": str(tmp_path / "buildkite-calls"),
        "BUILDX_HISTORY_LOG_ROOT": str(tmp_path / "logs"),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }
    return subprocess.run(
        [
            "bash",
            "-c",
            'set -uo pipefail; source "$1"; ' + command,
            "test",
            str(HELPER),
            str(metadata_file),
            str(tmp_path / "after"),
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def logs(tmp_path: Path) -> list[str]:
    return [
        gzip.decompress(path.read_bytes()).decode()
        for path in sorted((tmp_path / "logs").rglob("*.log.gz"))
    ]


def test_metadata_refs_are_deduplicated_and_step_path_is_safe(
    tmp_path: Path,
) -> None:
    metadata = """{
      "a": {"buildx.build.ref": "builder/node/build-a"},
      "b": {"buildx.build.ref": "builder/node/build-b"},
      "duplicate": {"buildx.build.ref": "builder/node/build-a"}
    }"""
    result = run_helper(
        tmp_path,
        metadata,
        'capture_buildx_history_logs "$2" "../metadata" 0',
    )

    artifact_call = (tmp_path / "buildkite-calls").read_text()
    assert result.returncode == 0
    assert len(logs(tmp_path)) == 2
    assert len(list((tmp_path / "logs").iterdir())) == 1
    assert "artifact upload" in artifact_call
    assert "*.log.gz" in artifact_call


def test_failed_bake_exports_new_history_but_not_stale(
    tmp_path: Path,
) -> None:
    result = run_helper(
        tmp_path,
        "",
        (
            'snapshot_buildx_history_refs "$2"; touch "$3"; '
            'capture_buildx_history_logs "$2" failed-bake 37'
        ),
    )

    assert result.returncode == 37
    output = logs(tmp_path)
    assert all("stale" not in log for log in output)
    assert any("new-a" in log for log in output)
    assert any("new-b" in log for log in output)


def test_export_failure_does_not_mask_build_status(tmp_path: Path) -> None:
    result = run_helper(
        tmp_path,
        '{"buildx.build.ref": "builder/node/failed-export"}',
        'capture_buildx_history_logs "$2" failed-export 23',
    )

    assert result.returncode == 23
    assert logs(tmp_path) == []
    assert "could not export Buildx history log" in result.stderr
