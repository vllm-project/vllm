# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import base64
import time
import uuid
from openai import AsyncOpenAI


async def async_query_openai(query, model_path, port):
    aclient = AsyncOpenAI(
        base_url=f"http://localhost:{str(port)}/v1", 
        api_key="EMPTY",
        timeout = 100000,
    )
    completion = await aclient.chat.completions.create(
        model=model_path,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": query,
            },
        ],
        temperature=0.0,
        top_p=0.1,
        max_tokens=512,
    )
    return completion.choices[0].message.content


async def async_process_queries(queries, model_path, port):
    results = await asyncio.gather(
        *(async_query_openai(query, model_path, port) for query in queries)
    )
    return results


async def main(args):
    # single query
    image_path = args.image_path
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read())
    encoded_image_text = encoded_image.decode("utf-8")
    image_base64 = f"data:image;base64,{encoded_image_text}"
    query = [
        {
            "type": "image_url",
            "image_url": {"url": image_base64},
        },
        {"type": "text", "text": "What is shown in the image.?"},
    ]
    bs = args.batch_size
    queries = [query for i in range(bs)]

    start_time = time.time()
    results = await async_process_queries(
        queries, 
        args.model_path, 
        args.port
    )
    end_time = time.time()
    for result in results:
        print(result)
        print("-" * 50)
    print(f"Total time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="test")
    parser.add_argument(
        "--model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct"
    )
    parser.add_argument("--image_path", type=str, default="./demo.jpeg")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--port", type=int, default=1)
    args, _ = parser.parse_known_args()

    asyncio.run(main(args))