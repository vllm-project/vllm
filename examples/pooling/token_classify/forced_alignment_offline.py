# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from Qwen3-ForcedAligner inference:
# https://github.com/QwenLM/Qwen3-ASR

"""
Offline forced alignment example using Qwen3-ForcedAligner-0.6B.

Forced alignment takes audio and reference text as input and produces
word-level timestamps. The model predicts a time bin at each <timestamp>
token position; multiplying by ``timestamp_segment_time`` gives milliseconds.

Usage::

    python forced_alignment_offline.py \
        --model Qwen/Qwen3-ForcedAligner-0.6B
"""

from argparse import Namespace

import numpy as np

from vllm import LLM, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


def parse_args():
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    parser.set_defaults(
        model="Qwen/Qwen3-ForcedAligner-0.6B",
        runner="pooling",
        enforce_eager=True,
        hf_overrides={"architectures": ["Qwen3ASRForcedAlignerForTokenClassification"]},
    )
    return parser.parse_args()


def build_prompt(words: list[str]) -> str:
    """Build the forced alignment prompt from a word list.

    Format: <|audio_start|><|audio_pad|><|audio_end|>
            word1<timestamp><timestamp>word2<timestamp><timestamp>...
    """
    body = "<timestamp><timestamp>".join(words) + "<timestamp><timestamp>"
    return f"<|audio_start|><|audio_pad|><|audio_end|>{body}"


def main(args: Namespace):
    llm = LLM(**vars(args))

    config = llm.llm_engine.vllm_config.model_config.hf_config
    timestamp_token_id = config.timestamp_token_id
    timestamp_segment_time = config.timestamp_segment_time

    # Example: align these words against a 5-second audio clip
    words = ["Hello", "world"]
    prompt = build_prompt(words)

    # Use a 5-second silent audio as placeholder (replace with real audio)
    sample_rate = 16000
    audio = np.zeros(sample_rate * 5, dtype=np.float32)

    outputs = llm.encode(
        [{"prompt": prompt, "multi_modal_data": {"audio": audio}}],
        pooling_task="token_classify",
    )

    for output in outputs:
        logits = output.outputs.data  # [num_tokens, classify_num]
        predictions = logits.argmax(dim=-1)
        token_ids = output.prompt_token_ids

        # Extract timestamps at <timestamp> positions
        ts_predictions = [
            pred.item() * timestamp_segment_time
            for tid, pred in zip(token_ids, predictions)
            if tid == timestamp_token_id
        ]

        # Pair up start/end times per word
        for i, word in enumerate(words):
            start_ms = ts_predictions[i * 2]
            end_ms = ts_predictions[i * 2 + 1]
            print(f"{word:15s} {start_ms / 1000:.3f}s - {end_ms / 1000:.3f}s")


if __name__ == "__main__":
    args = parse_args()
    main(args)
