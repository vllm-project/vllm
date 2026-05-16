# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from openvla_check_config import (
    CASES_PATH,
    DATASET_ID,
    DATASET_SOURCE,
    EPISODE_INDEX,
    IMAGE_DIR,
    NUM_CASES,
)
from PIL import Image


def load_tasks() -> dict[int, str]:
    tasks_path = hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        filename="meta/tasks.jsonl",
    )
    tasks = {}
    with Path(tasks_path).open() as f:
        for line in f:
            item = json.loads(line)
            tasks[int(item["task_index"])] = item["task"]
    return tasks


def to_image(value: object) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path") is not None:
            return Image.open(value["path"]).convert("RGB")

    raise TypeError(f"Unsupported image value: {type(value)}")


def load_episode() -> list[dict[str, object]]:
    parquet_path = hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        filename=f"data/chunk-000/episode_{EPISODE_INDEX:06d}.parquet",
    )
    return pq.read_table(parquet_path).to_pylist()


def main() -> None:
    tasks = load_tasks()
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_episode()
    row_indices = [
        round(i * (len(rows) - 1) / (NUM_CASES - 1)) for i in range(NUM_CASES)
    ]

    cases = []
    for row_index in row_indices:
        row = rows[row_index]
        task_index = int(row["task_index"])
        task = tasks[task_index]
        image_path = IMAGE_DIR / f"case_{len(cases)}.jpg"
        to_image(row["image"]).save(image_path)

        cases.append(
            {
                "case_id": f"case_{len(cases)}",
                "dataset": DATASET_ID,
                "dataset_source": DATASET_SOURCE,
                "episode_index": int(row["episode_index"]),
                "frame_index": int(row["frame_index"]),
                "task_index": task_index,
                "instruction": task,
                "prompt": f"In: What action should the robot take to {task}?\nOut:",
                "image_path": str(image_path),
            }
        )

    payload = {
        "dataset": DATASET_ID,
        "dataset_source": DATASET_SOURCE,
        "episode_index": EPISODE_INDEX,
        "row_indices": row_indices,
        "num_cases": len(cases),
        "cases": cases,
    }
    CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CASES_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
