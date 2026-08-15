# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Drive real shared-prefix requests for GVR selector data capture."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer


def make_prompt_ids(model: str, length: int) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    pieces: list[str] = []
    chars = 0
    for path in sorted(Path("docs").rglob("*.md")):
        text = path.read_text(errors="ignore")
        pieces.append(text)
        chars += len(text)
        if chars >= length * 6:
            break
    token_ids = tokenizer("\n\n".join(pieces), add_special_tokens=False).input_ids
    if len(token_ids) < length:
        copies = (length + len(token_ids) - 1) // len(token_ids)
        token_ids = (token_ids * copies)[:length]
    return token_ids[:length]


async def run_wave(
    session: aiohttp.ClientSession,
    url: str,
    model_name: str,
    prompt_ids: list[int],
    batch: int,
    output_len: int,
    admission_chunk: int,
    admission_delay: float,
    single_request_n: bool,
    temperature: float,
) -> None:
    start = asyncio.Event()

    payload = {
        "model": model_name,
        "prompt": prompt_ids,
        "max_tokens": output_len,
        "min_tokens": output_len,
        "temperature": temperature,
        "ignore_eos": True,
        "seed": 0,
    }
    if single_request_n:
        payload["n"] = batch
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}

    async def run_one() -> None:
        await start.wait()
        for attempt in range(3):
            try:
                async with session.post(url, data=body, headers=headers) as response:
                    if response.status != 200:
                        raise RuntimeError(await response.text())
                    await response.read()
                return
            except aiohttp.ClientOSError:
                if attempt == 2:
                    raise
                await asyncio.sleep(0.25 * (attempt + 1))

    begin = time.perf_counter()
    if single_request_n:
        tasks = [asyncio.create_task(run_one())]
        start.set()
    elif admission_chunk:
        tasks = []
        start.set()
        for offset in range(0, batch, admission_chunk):
            tasks.extend(
                asyncio.create_task(run_one())
                for _ in range(offset, min(offset + admission_chunk, batch))
            )
            await asyncio.sleep(admission_delay)
    else:
        tasks = [asyncio.create_task(run_one()) for _ in range(batch)]
        await asyncio.sleep(0)
        start.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    errors = [result for result in results if isinstance(result, BaseException)]
    if errors:
        raise RuntimeError(f"{len(errors)} requests failed; first error: {errors[0]}")
    elapsed = time.perf_counter() - begin
    print(
        f"length={len(prompt_ids)} batch={batch} elapsed={elapsed:.3f}s",
        flush=True,
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default="GLM-5.2")
    parser.add_argument("--url", default="http://127.0.0.1:8005/v1/completions")
    parser.add_argument("--lengths", default="200000,50000,10000")
    parser.add_argument("--batches", default="1,8,32")
    parser.add_argument("--output-len", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--admission-chunk", type=int, default=0)
    parser.add_argument("--admission-delay", type=float, default=0.1)
    parser.add_argument(
        "--single-request-n",
        action="store_true",
        help="Create the batch as n child sequences in one completion request.",
    )
    args = parser.parse_args()

    lengths = [int(value) for value in args.lengths.split(",")]
    batches = [int(value) for value in args.batches.split(",")]
    all_ids = make_prompt_ids(args.model, max(lengths))
    timeout = aiohttp.ClientTimeout(total=7200)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for length in lengths:
            prompt_ids = all_ids[:length]
            for batch in batches:
                await run_wave(
                    session,
                    args.url,
                    args.served_model_name,
                    prompt_ids,
                    batch,
                    args.output_len,
                    args.admission_chunk,
                    args.admission_delay,
                    args.single_request_n,
                    args.temperature,
                )


if __name__ == "__main__":
    asyncio.run(main())
