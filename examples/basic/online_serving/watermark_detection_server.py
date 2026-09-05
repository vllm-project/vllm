# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal reference server for watermark detection."""

import argparse

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from vllm.tokenizers import TokenizerLike, cached_get_tokenizer
from vllm.v1.watermarking import GumbelWatermarkDetector

app = FastAPI()
tokenizer: TokenizerLike | None = None
detector: GumbelWatermarkDetector | None = None


class DetectionRequest(BaseModel):
    text: str


class DetectionResponse(BaseModel):
    score: float
    p_value: float
    num_scored_tokens: int
    is_watermarked: bool


@app.post("/detect")
def detect(request: DetectionRequest) -> DetectionResponse:
    assert tokenizer is not None
    assert detector is not None
    token_ids = tokenizer.encode(request.text, add_special_tokens=False)
    result = detector.detect(token_ids)
    return DetectionResponse(
        score=result.score,
        p_value=result.p_value,
        num_scored_tokens=result.num_scored_tokens,
        is_watermarked=result.is_watermarked,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--key", required=True, type=int)
    parser.add_argument("--prf", choices=("philox", "hmac_sha256"), default="philox")
    parser.add_argument("--context-width", type=int, default=4)
    parser.add_argument("--p-value-threshold", type=float, default=0.01)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    global tokenizer, detector
    tokenizer = cached_get_tokenizer(args.tokenizer)
    detector = GumbelWatermarkDetector(
        key=args.key,
        context_width=args.context_width,
        p_value_threshold=args.p_value_threshold,
        prf=args.prf,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main(parse_args())
