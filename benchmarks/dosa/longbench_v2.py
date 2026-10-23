# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired greedy LongBench-v2 quality evaluation through the real vLLM API."""

import argparse
import asyncio
import csv
import json
import time
import uuid
from collections import Counter
from pathlib import Path

import aiohttp
import regex as re

MAX_NEW_TOKENS = 128
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
DATASET_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
TEMPLATE = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: "The correct answer is (insert answer here)"."""


def prompt(item, template):
    fields = {"DOC": "context", "Q": "question"}
    fields.update({f"C_{letter}": f"choice_{letter}" for letter in "ABCD"})
    for slot, field in fields.items():
        template = template.replace(f"${slot}$", item[field].strip())
    return template


def answer(text):
    # The benchmark's original answer parser; unparsed responses score zero.
    text = text.replace("*", "")
    for pattern in (
        r"The correct answer is \(([A-D])\)",
        r"The correct answer is ([A-D])",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def save(path, data):
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


async def main(args):
    with args.samples.open(newline="") as stream:
        samples = [
            {"id": row["id"], "tokens": int(row["prompt_tokens"])}
            for row in csv.DictReader(stream)
        ]
    dataset = {row["_id"]: row for row in json.loads(args.data.read_bytes())}
    template = TEMPLATE
    args.output.mkdir(parents=True, exist_ok=False)
    state = {
        "status": "RUNNING",
        "arm": args.arm,
        "samples_file": str(args.samples),
        "data": str(args.data),
        "dataset_revision": DATASET_REVISION,
        "started_at": time.time(),
        "samples": [],
        "timing_valid": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
    }
    save(args.output / "STATE.json", state)
    try:
        timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for sample in samples:
                item = dataset[sample["id"]]
                payload = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt(item, template)}],
                    "max_tokens": MAX_NEW_TOKENS,
                    "temperature": 0,
                    "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
                }
                rid = f"dosa-eval-{args.arm}-{uuid.uuid4().hex}"
                async with session.post(
                    args.base_url + "/reset_prefix_cache"
                ) as response:
                    response.raise_for_status()
                    reset = await response.json()
                if reset != {"success": True}:
                    raise RuntimeError(f"Prefix reset failed: {reset}")
                async with session.post(
                    args.base_url + "/v1/chat/completions",
                    json=payload,
                    headers={"x-request-id": rid},
                ) as response:
                    raw = await response.json()
                    receipt = {"request_id": rid, "http_status": response.status, **raw}
                    save(args.output / f"{sample['id']}.json", receipt)
                    response.raise_for_status()
                message = raw["choices"][0]["message"]
                text = message["content"] or ""
                pred = answer(text)
                result = {
                    **sample,
                    "answer": item["answer"],
                    "predicted": pred,
                    "correct": pred == item["answer"],
                    "usage": raw["usage"],
                    "finish_reason": raw["choices"][0]["finish_reason"],
                    "length_capped": raw["choices"][0]["finish_reason"] == "length",
                    "at_output_cap": raw["usage"]["completion_tokens"]
                    >= MAX_NEW_TOKENS,
                    "unparsed": pred is None,
                    "content_chars": len(text),
                    "reasoning_chars": len(message.get("reasoning") or ""),
                }
                state["samples"].append(result)
                save(args.output / "STATE.json", state)
                if raw["usage"]["prompt_tokens"] != sample["tokens"]:
                    raise AssertionError(
                        (sample["id"], sample["tokens"], raw["usage"]["prompt_tokens"])
                    )
                print(json.dumps(result), flush=True)
        state.update(
            status="COMPLETE",
            count=len(state["samples"]),
            correct=sum(sample["correct"] for sample in state["samples"]),
            unparsed=sum(sample["unparsed"] for sample in state["samples"]),
            length_capped=sum(sample["length_capped"] for sample in state["samples"]),
            at_output_cap=sum(sample["at_output_cap"] for sample in state["samples"]),
            finish_reason_counts=dict(
                Counter(sample["finish_reason"] for sample in state["samples"])
            ),
        )
    except Exception as error:
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["finished_at"] = time.time()
        save(args.output / "STATE.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path(__file__).with_name("longbench_v2_glm.csv"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", choices=("stock", "dosa"), required=True)
    asyncio.run(main(parser.parse_args()))
