# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serial FP8 model TTFT using the frozen upstream vLLM benchmark client."""

import argparse
import asyncio
import dataclasses
import json
import statistics
import time
import uuid
from pathlib import Path

import aiohttp

from vllm.benchmarks.lib.endpoint_request_func import (
    AIOHTTP_TIMEOUT,
    RequestFuncInput,
    async_request_openai_completions,
)


def save(path, value):
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


async def run(args):
    plan = json.loads(args.requests.read_bytes())
    args.output.mkdir(parents=True, exist_ok=False)
    state = {
        "status": "RUNNING",
        "started_at": time.time(),
        "arm": args.arm,
        "source": args.source,
        "request_plan": str(args.requests),
        "client": (
            "vllm.benchmarks.lib.endpoint_request_func.async_request_openai_completions"
        ),
        "protocol": (
            "original UTF8 text; fresh RID and prefix reset; "
            "1 warm + 3 formal; first streamed completion choice"
        ),
        "generation": {"max_tokens": 2, "temperature": 0, "ignore_eos": False},
        "requests": [],
    }
    save(args.output / "STATE.json", state)
    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            for item in plan:
                prompt_path = Path(item["input"])
                original = prompt_path.read_bytes()
                prompt = original.decode("utf-8")
                cell = {
                    "name": item["name"],
                    "input": str(prompt_path),
                    "bytes": len(original),
                    "status": "MEASURING",
                    "receipts": [],
                }
                state["requests"].append(cell)
                save(args.output / "STATE.json", state)
                for repeat in range(4):
                    role = "warm" if repeat == 0 else f"formal{repeat}"
                    request_id = f"dosa-{args.arm}-{item['name']}-{uuid.uuid4().hex}"
                    async with session.post(
                        args.base_url.rstrip("/") + "/reset_prefix_cache"
                    ) as response:
                        response.raise_for_status()
                        reset = await response.json()
                    if reset != {"success": True}:
                        raise RuntimeError(f"Prefix reset failed: {reset}")
                    request = RequestFuncInput(
                        prompt=prompt,
                        api_url=args.base_url.rstrip("/") + "/v1/completions",
                        prompt_len=item.get("tokens", 0),
                        output_len=2,
                        model=args.model,
                        extra_body={"temperature": 0},
                        ignore_eos=False,
                        request_id=request_id,
                    )
                    result = dataclasses.asdict(
                        await async_request_openai_completions(request, session)
                    )
                    receipt = {
                        "request_id": request_id,
                        "role": role,
                        "cache_reset": reset,
                        **result,
                    }
                    receipt_name = f"{item['name']}-{role}.json"
                    save(args.output / receipt_name, receipt)
                    cell["receipts"].append(receipt_name)
                    save(args.output / "STATE.json", state)
                    if not result["success"] or result["ttft"] <= 0:
                        raise RuntimeError(f"{item['name']} {role}: {result['error']}")
                    if "tokens" in item and result["prompt_len"] != item["tokens"]:
                        raise AssertionError(
                            (item["name"], result["prompt_len"], item["tokens"])
                        )
                    print(
                        json.dumps(
                            {
                                "name": item["name"],
                                "role": role,
                                "tokens": result["prompt_len"],
                                "ttft_s": result["ttft"],
                            }
                        ),
                        flush=True,
                    )
                measured = [
                    json.loads((args.output / name).read_bytes())
                    for name in cell["receipts"][1:]
                ]
                cell.update(
                    status="COMPLETE",
                    prompt_tokens=measured[0]["prompt_len"],
                    median_ttft_s=statistics.median(x["ttft"] for x in measured),
                )
                if len({x["prompt_len"] for x in measured}) != 1:
                    raise AssertionError("Prompt length changed between repetitions")
                save(args.output / "STATE.json", state)
        state["status"] = "COMPLETE"
    except Exception as error:
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["finished_at"] = time.time()
        save(args.output / "STATE.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", choices=("stock", "dosa"), required=True)
    parser.add_argument("--source", required=True)
    asyncio.run(run(parser.parse_args()))
