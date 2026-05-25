# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Kimi-K2.5 front-end preprocessing example.

This script demonstrates the SGLang-style "front-end + vLLM gRPC back-end"
pattern for Kimi-K2.5:

  1. A stand-alone Python preprocessor
     (`vllm.model_executor.models.kimi_k25_frontend.KimiK25Preprocessor`)
     turns a raw prompt string + a list of vision chunks into a fully-
     rendered `MultiModalInput` dict.
  2. The dict is handed to `LLMEngine.add_request(...)` directly, bypassing
     the renderer's `_process_multimodal` path and the
     `MULTIMODAL_REGISTRY`-driven multi-modal processor altogether. The
     back-end engine just runs inference.

Compared to `LLM.generate(prompts=[{"prompt": ..., "multi_modal_data": ...}])`,
this front-end path:

  - Has zero dependency on `BaseProcessingInfo` / `BaseDummyInputsBuilder`
    / `BaseMultiModalProcessor`. Algorithm engineers can change Kimi-K2.5
    vision preprocessing by editing **only** `kimi_k25_frontend.py`.
  - Lets the user run preprocessing on a different process / host than
    the inference engine. Just pickle the returned dict and ship it.

Run with (assuming a checkpoint locally available or pre-pulled):

    python examples/generate/multimodal/kimi_k25_frontend_offline.py \\
        --model moonshotai/Kimi-K2.5 \\
        --tensor-parallel-size 4
"""

from __future__ import annotations

import argparse

from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.model_executor.models.kimi_k25_frontend import KimiK25Preprocessor
from vllm.utils.argparse_utils import FlexibleArgumentParser

# Matches `KimiK25ForConditionalGeneration.get_placeholder_str("image", ...)`.
IMAGE_PLACEHOLDER = "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"


def build_prompt(question: str) -> str:
    return (
        f"<|im_user|>user{IMAGE_PLACEHOLDER}{question}<|im_end|>"
        "<|im_assistant|>assistant<|im_middle|>"
    )


def parse_args() -> argparse.Namespace:
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="/mnt/data3/kimi/models/Kimi-K2.5")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--question",
        type=str,
        default="Describe this image in detail.",
        help="Question to ask about the image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Step 1: front-end preprocessing. No vLLM engine yet. -------------
    # The preprocessor only needs the model id; it pulls the tokenizer and
    # `preprocessor_config.json` itself. In a real SGLang-style deployment
    # you would create this on the front-end host.
    preprocessor = KimiK25Preprocessor.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True
    )

    image = ImageAsset("cherry_blossom").pil_image.convert("RGB")
    prompt = build_prompt(args.question)

    engine_input = preprocessor.preprocess(
        prompt=prompt,
        vision_chunks=[{"type": "image", "image": image}],
    )

    vc_ranges = [
        (r.offset, r.length) for r in engine_input["mm_placeholders"]["vision_chunk"]
    ]
    print(
        f"[frontend] prompt_token_ids len = "
        f"{len(engine_input['prompt_token_ids'])}, "
        f"placeholder ranges = {vc_ranges}, "
        f"mm_hashes = {engine_input['mm_hashes']}"
    )

    # ---- Step 2: ship the dict to the vLLM back-end. ----------------------
    # `engine_input` already has the shape that
    # `BaseRenderer._process_multimodal` would have produced, so the engine
    # consumes it directly via `add_request`. This is the "vLLM gRPC" entry
    # point: in a real deployment, `engine_input` could just as easily be
    # serialized and sent over the wire.
    engine_args = EngineArgs(
        model=args.model,
        revision=args.revision,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        limit_mm_per_prompt={"vision_chunk": 1},
    )
    llm = LLM.from_engine_args(engine_args)

    sampling_params = SamplingParams(temperature=0.2, max_tokens=args.max_tokens)
    llm.llm_engine.add_request("0", engine_input, sampling_params)

    while llm.llm_engine.has_unfinished_requests():
        for output in llm.llm_engine.step():
            if output.finished:
                print("=" * 50)
                print(f"[backend] generated text:\n{output.outputs[0].text}")
                print("=" * 50)


if __name__ == "__main__":
    main()
