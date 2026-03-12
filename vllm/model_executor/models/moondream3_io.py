# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared detect/point I/O helpers for Moondream3."""

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch

MOONDREAM3_TASK_DETECT = "detect"
MOONDREAM3_TASK_POINT = "point"
MOONDREAM3_TASKS = (MOONDREAM3_TASK_DETECT, MOONDREAM3_TASK_POINT)

MOONDREAM3_MAX_OBJECTS_DEFAULT = 150

MOONDREAM3_RESULT_MODE_KEY = "moondream3_detect_point_mode"
MOONDREAM3_RESULT_DATA_KEY = "moondream3_detect_point_data"

_MODE_DETECT = 0
_MODE_POINT = 1


def build_moondream3_detect_point_prompt(
    task: Literal["detect", "point"],
    target: str,
) -> str:
    return (
        "<|endoftext|><image><|md_reserved_0|>"
        f"{task}<|md_reserved_1|> {target}<|md_reserved_2|>"
    )


def encode_moondream3_detect_point_output(
    task: Literal["detect", "point"],
    items: Sequence[Mapping[str, float]],
) -> dict[str, torch.Tensor]:
    if task == MOONDREAM3_TASK_DETECT:
        mode = _MODE_DETECT
        flat_values = [
            coord
            for item in items
            for coord in (
                item["x_min"],
                item["y_min"],
                item["x_max"],
                item["y_max"],
            )
        ]
    else:
        mode = _MODE_POINT
        flat_values = [coord for item in items for coord in (item["x"], item["y"])]

    return {
        MOONDREAM3_RESULT_MODE_KEY: torch.tensor([mode], dtype=torch.int32),
        MOONDREAM3_RESULT_DATA_KEY: torch.tensor(flat_values, dtype=torch.float32),
    }


def decode_moondream3_detect_point_output(
    model_extra_output: Mapping[str, Any],
) -> dict[str, list[dict[str, float]]] | None:
    mode_tensor = model_extra_output.get(MOONDREAM3_RESULT_MODE_KEY)
    data_tensor = model_extra_output.get(MOONDREAM3_RESULT_DATA_KEY)
    if not isinstance(mode_tensor, torch.Tensor) or not isinstance(
        data_tensor, torch.Tensor
    ):
        return None
    if mode_tensor.dim() != 1 or mode_tensor.numel() != 1 or data_tensor.dim() != 1:
        return None

    mode = int(mode_tensor.to("cpu", dtype=torch.int32).item())
    values = data_tensor.to("cpu", dtype=torch.float32).tolist()

    if mode == _MODE_DETECT:
        if len(values) % 4 != 0:
            return None
        return {
            "objects": [
                {
                    "x_min": values[i],
                    "y_min": values[i + 1],
                    "x_max": values[i + 2],
                    "y_max": values[i + 3],
                }
                for i in range(0, len(values), 4)
            ]
        }

    if mode == _MODE_POINT:
        if len(values) % 2 != 0:
            return None
        return {
            "points": [
                {
                    "x": values[i],
                    "y": values[i + 1],
                }
                for i in range(0, len(values), 2)
            ]
        }

    return None


def decode_moondream3_detect_point_output_json(
    model_extra_output: Mapping[str, Any],
) -> str | None:
    result = decode_moondream3_detect_point_output(model_extra_output)
    if result is None:
        return None
    return json.dumps(result)
