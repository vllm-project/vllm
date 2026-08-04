# SPDX-License-Identifier: Apache-2.0
# Standard
import argparse
import asyncio
import copy
import json
import logging
import os
import time
import traceback

# Third Party
from tqdm import tqdm
import aiohttp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


async def make_request(session, payload: tuple, args):
    endpoint = args.endpoint
    output_file = args.output_file
    obj_name, payload = payload
    headers = {
        "Authorization": f"Bearer {os.getenv('API_KEY', '')}",
        "Content-Type": "application/json",
    }
    try:
        async with session.post(
            endpoint, json=payload, headers=headers, ssl=False
        ) as response:
            json_data = await response.json()
            content = (
                json_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            logger.info(content)
            if json_data.get("id") is not None:
                with open(output_file, "a") as f:
                    f.write(f"{json_data.get('id')}:\n{content}\n")
                if json_data.get("choices", [{}])[0].get("finish_reason", "") != "stop":
                    with open(output_file.replace(".txt", "_length.txt"), "a") as f:
                        f.write(f"{json_data.get('id')}:\n{content}\n")
            return True, json_data
    except Exception:
        logger.error(f"Request failed (Object name: {obj_name})")
        traceback.print_exc()
        return False, None


async def process_requests(payloads: dict, args):
    semaphore = asyncio.Semaphore(args.max_concurrency)

    async def sem_task(session, payload: tuple, args):
        async with semaphore:
            return await make_request(session, payload, args)

    timeout = aiohttp.ClientTimeout(total=6000)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [sem_task(session, payload, args) for payload in payloads.items()]

        generated_token = []
        for coro in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Requests Progress"
        ):
            success, content = await coro
            if success:
                generated_token.append(
                    content.get("usage", {}).get("completion_tokens", 0)
                )
            else:
                logger.error(f"Error: {content}")
        logger.info(f"Total generated tokens: {sum(generated_token)}")


def request_concurrency(requests: dict, args):
    same_req = args.same_req
    rid = args.rid
    counts = args.counts
    model = args.model
    payloads = dict()
    if same_req:
        for index, request in requests.items():
            if index == rid:
                request.update({"model": model})
                request.update({"stream": False})
                request.update({"temperature": 0})
                request.update({"request_id": index})
                payloads[index] = request
                for i in range(counts):
                    payloads[i] = copy.deepcopy(payloads[index])
                    payloads[i]["request_id"] = str(i)
                break

    else:
        for index, request in requests.items():
            request.update({"model": model})
            request.update({"stream": False})
            request.update({"temperature": 0})
            request["request_id"] = index
            payloads[index] = request

    start = time.time()
    asyncio.run(process_requests(payloads, args))
    end = time.time()
    logger.info(f"Total time: {end - start:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Concurrent request tester")

    parser.add_argument(
        "--model", type=str, default="Qwen3/Qwen3-32B", help="Model name"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        required=True,
        help="API endpoint, e.g. http://host:port/v1/chat/completions",
    )
    parser.add_argument(
        "--request-number",
        type=int,
        default=500,
        help="Number of requests to read from dataset",
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=128, help="Max concurrency"
    )
    parser.add_argument(
        "--dataset_file", type=str, default="data.json", help="Dataset json file"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="result_output.txt",
        help="Output result file prefix",
    )
    parser.add_argument(
        "--same-req",
        action="store_true",
        help="Send the same rid request multiple times",
    )
    parser.add_argument(
        "--rid", type=str, default="", help="The rid to repeat if same_req=True"
    )
    parser.add_argument(
        "--counts", type=int, default=100, help="How many times to send the same rid"
    )

    args = parser.parse_args()

    # --- load dataset ---
    with open(args.dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = dict(list(data.items())[: args.request_number])

    print(f"Loaded {len(data)} requests.")

    # --- call your function ---
    request_concurrency(data, args)


if __name__ == "__main__":
    main()
