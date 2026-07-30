# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gzip
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

HELPER = (
    Path(__file__).resolve().parents[2]
    / ".buildkite"
    / "scripts"
    / "rocm"
    / "buildx-history-logs.sh"
)


def write_fake_docker(tmp_path: Path, body: str) -> Path:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    docker.chmod(0o755)
    return fake_bin


def run_helper(
    command: str,
    metadata_file: Path,
    *,
    fake_bin: Path,
    log_root: Path,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    helper_env = os.environ.copy()
    helper_env.update(
        {
            "BUILDX_HISTORY_LOG_ROOT": str(log_root),
            "PATH": f"{fake_bin}:{helper_env['PATH']}",
            **(env or {}),
        }
    )
    return subprocess.run(
        [
            "bash",
            "-c",
            'set -uo pipefail\nsource "$1"\n' + command,
            "buildx-history-test",
            str(HELPER),
            str(metadata_file),
        ],
        check=False,
        env=helper_env,
        capture_output=True,
        text=True,
    )


def compressed_logs(log_root: Path) -> list[str]:
    return [
        gzip.decompress(path.read_bytes()).decode()
        for path in sorted(log_root.rglob("*.log.gz"))
    ]


def test_nested_bake_metadata_exports_every_builder_record(
    tmp_path: Path,
) -> None:
    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text(
        """
        {
          "group": {
            "first": {"buildx.build.ref": "builder-a/node-0/build-a"},
            "nested": {
              "second": {"buildx.build.ref": "builder-a/node-1/build-b"},
              "duplicate": {"buildx.build.ref": "builder-a/node-0/build-a"}
            }
          }
        }
        """
    )
    call_log = tmp_path / "docker-calls"
    fake_bin = write_fake_docker(
        tmp_path,
        """
printf '%s\\n' "$*" >> "${FAKE_DOCKER_CALL_LOG}"
if [[ "$*" == *" history logs "* ]]; then
    printf 'detailed log: %s\\n' "$*"
    exit 0
fi
exit 42
""",
    )
    result = run_helper(
        'capture_buildx_history_logs "$2" "metadata" 0',
        metadata_file,
        fake_bin=fake_bin,
        log_root=tmp_path / "logs",
        env={"FAKE_DOCKER_CALL_LOG": str(call_log)},
    )

    assert result.returncode == 0
    logs = compressed_logs(tmp_path / "logs")
    assert len(logs) == 2
    expected_a = "--builder builder-a history logs --progress=plain build-a"
    expected_b = "--builder builder-a history logs --progress=plain build-b"
    assert any(expected_a in log for log in logs)
    assert any(expected_b in log for log in logs)
    assert len(call_log.read_text().splitlines()) == 2


def test_failed_bake_exports_all_new_records_but_not_stale_history(
    tmp_path: Path,
) -> None:
    metadata_file = tmp_path / "metadata.json"
    metadata_file.touch()
    state_file = tmp_path / "history-list-count"
    fake_bin = write_fake_docker(
        tmp_path,
        """
if [[ "$1 $2 $3" == "buildx history ls" ]]; then
    count=0
    [[ ! -f "${FAKE_HISTORY_STATE}" ]] || read -r count < "${FAKE_HISTORY_STATE}"
    count=$((count + 1))
    printf '%s\\n' "${count}" > "${FAKE_HISTORY_STATE}"
    printf '%s\\n' "builder-b/node-0/stale"
    if ((count > 1)); then
        printf '%s\\n' "builder-b/node-0/new-a" "builder-b/node-1/new-b"
    fi
    exit 0
fi
if [[ "$*" == *" history logs "* ]]; then
    printf 'detailed log: %s\\n' "$*"
    exit 0
fi
exit 42
""",
    )
    result = run_helper(
        (
            'snapshot_buildx_history_refs "$2" || exit $?\n'
            'capture_buildx_history_logs "$2" "failed-bake" 37'
        ),
        metadata_file,
        fake_bin=fake_bin,
        log_root=tmp_path / "logs",
        env={"FAKE_HISTORY_STATE": str(state_file)},
    )

    assert result.returncode == 37
    logs = compressed_logs(tmp_path / "logs")
    assert len(logs) == 2
    assert all("stale" not in log for log in logs)
    expected_a = "--builder builder-b history logs --progress=plain new-a"
    expected_b = "--builder builder-b history logs --progress=plain new-b"
    assert any(expected_a in log for log in logs)
    assert any(expected_b in log for log in logs)


def test_log_export_failure_preserves_the_build_status(tmp_path: Path) -> None:
    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text('{"buildx.build.ref": "builder-c/node-0/failed-export"}')
    fake_bin = write_fake_docker(
        tmp_path,
        """
if [[ "$*" == *" history logs "* ]]; then
    echo "history export failed" >&2
    exit 41
fi
exit 42
""",
    )
    result = run_helper(
        'capture_buildx_history_logs "$2" "failed-export" 23',
        metadata_file,
        fake_bin=fake_bin,
        log_root=tmp_path / "logs",
    )

    assert result.returncode == 23
    assert not compressed_logs(tmp_path / "logs")
    assert "could not export Buildx history log" in result.stderr
