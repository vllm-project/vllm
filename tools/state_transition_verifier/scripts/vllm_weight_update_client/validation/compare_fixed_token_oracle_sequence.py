#!/usr/bin/env python3
"""Compare repeated fixed-token oracles and the pre/post update boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def extract(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    request = document["request"]
    completion = document["completion"]
    choice = completion["choices"][0]
    fields = {
        "token_ids": choice.get("token_ids"),
        "prompt_token_ids": choice.get("prompt_token_ids"),
        "logprobs": choice.get("logprobs"),
        "prompt_logprobs": choice.get("prompt_logprobs"),
        "finish_reason": choice.get("finish_reason"),
        "stop_reason": choice.get("stop_reason"),
        "request_id": request.get("request_id"),
        "response_id": completion.get("id"),
        "usage": completion.get("usage"),
        "input_token_ids": request.get("prompt"),
    }
    required = set(fields) - {"stop_reason"}
    missing = [name for name in required if fields[name] is None]
    if missing:
        raise ValueError(f"missing oracle fields in {path}: {missing}")
    return fields


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pre", type=Path)
    parser.add_argument("pre_repeat", type=Path)
    parser.add_argument("post", type=Path)
    parser.add_argument("post_repeat", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    pre = extract(args.pre)
    pre_repeat = extract(args.pre_repeat)
    post = extract(args.post)
    post_repeat = extract(args.post_repeat)
    result = {
        "status": "PASS" if pre == pre_repeat == post == post_repeat else "FAIL",
        "pre_repeat_same": pre == pre_repeat,
        "post_repeat_same": post == post_repeat,
        "pre_post_same": pre == post,
        "post_first_second_same": post == post_repeat,
        "pre": pre,
        "pre_repeat": pre_repeat,
        "post": post,
        "post_repeat": post_repeat,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
