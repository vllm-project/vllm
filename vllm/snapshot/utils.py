# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os

from vllm.platforms import current_platform

RETRY_INTERVAL = 1.0
RETRY_LOG_FREQUENCY = 60


def is_restore() -> bool:
    return current_platform.is_restore()


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
