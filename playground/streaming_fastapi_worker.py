import argparse
import asyncio
import time
from typing import Union
import json

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn


app = FastAPI()


async def text_streamer(args):
    context = args["prompt"]
    words = context.split(" ")
    for word in words:
        await asyncio.sleep(1)
        print("word:", word)
        ret = {
            "text": word + " ",
            "error": 0,
        }
        yield (json.dumps(ret) + "\0").encode("utf-8")


@app.post("/")
async def read_root(request: Request):
    args = await request.json()
    return StreamingResponse(text_streamer(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=10002)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
