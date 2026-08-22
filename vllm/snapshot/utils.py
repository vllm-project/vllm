# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import socket

RESTORED_FLAG_PATH = "/root/.grusflag"
RETRY_INTERVAL = 1.0
RETRY_LOG_FREQUENCY = 60


def get_local_ip() -> str:
    """Probe the current local IP without using a cached environment value."""
    targets = (
        (socket.AF_INET, ("8.8.8.8", 80)),
        (socket.AF_INET6, ("2001:4860:4860::8888", 80)),
    )
    for family, target in targets:
        try:
            with socket.socket(family, socket.SOCK_DGRAM) as sock:
                sock.connect(target)
                return sock.getsockname()[0]
        except OSError:
            continue
    raise RuntimeError("Failed to detect the current local IP address")


def is_restore() -> bool:
    return os.path.exists(RESTORED_FLAG_PATH)


def load_snapshot_metadata(file_path: str, field_name: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Snapshot metadata file not found: {file_path}")

    with open(file_path, encoding="utf-8") as file:
        try:
            data = json.load(file)
        except Exception as exc:
            raise ValueError(
                f"Snapshot metadata is not valid JSON: {file_path}: {exc}"
            ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            "Snapshot metadata JSON root must be an object, not an array or "
            f"scalar: {file_path}"
        )

    field_value = data.get(field_name)
    if not isinstance(field_value, str):
        raise ValueError(
            "Snapshot metadata requires string field: "
            f"{field_name}, but got {type(field_value)}"
        )
    return field_value
