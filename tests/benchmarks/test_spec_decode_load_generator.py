# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import random
from pathlib import Path

from benchmarks.spec_decode.load_generator import load_prompts


def test_load_prompts_matches_deepspec_selection(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    rows = [{"turns": [f"prompt-{index}", "ignored"]} for index in range(10)]
    path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )

    expected = [row["turns"][0] for row in rows]
    random.Random(980406).shuffle(expected)

    assert load_prompts(path, 4) == expected[:4]


def test_load_prompts_accepts_legacy_prompt_rows(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text('{"prompt": "hello"}\n', encoding="utf-8")

    assert load_prompts(path, 0) == ["hello"]
