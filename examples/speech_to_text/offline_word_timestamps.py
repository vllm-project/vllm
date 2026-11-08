# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline Nemotron Transcribe word timestamps for clips up to 30 seconds.

Example:
    python examples/speech_to_text/offline_word_timestamps.py clip.wav --mtp

"""

import argparse
import json

import soundfile as sf

from vllm import LLM, SamplingParams
from vllm.entrypoints.speech_to_text.offline import transcribe_with_word_timestamps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="Path to a mono audio file, at most 30 seconds")
    parser.add_argument(
        "--model", default="nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
    )
    parser.add_argument("--mtp", action="store_true", help="Draft two tokens with MTP")
    args = parser.parse_args()
    audio, sample_rate = sf.read(args.audio, dtype="float32")
    if audio.ndim != 1 or not 0 < len(audio) / sample_rate <= 30:
        parser.error("Provide one nonempty mono clip lasting at most 30 seconds")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        hf_overrides={"enable_ctc_timestamps": True},
        enforce_eager=True,
        enable_prefix_caching=False,
        max_model_len=4096,
        max_num_seqs=1,
        limit_mm_per_prompt={"audio": {"count": 1, "length": 480000}},
        speculative_config=(
            {"method": "mtp", "num_speculative_tokens": 2} if args.mtp else None
        ),
    )
    prompt = llm.get_tokenizer().apply_chat_template(
        [
            {
                "role": "user",
                "content": (
                    "Audio transcription. Language: as spoken. "
                    "Wording: verbatim, including fillers and repetitions. "
                    "Formatting: punctuation and capitalization. "
                    "Output: transcript only. <|audio|>"
                ),
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    result = transcribe_with_word_timestamps(
        llm,
        {"prompt": prompt, "multi_modal_data": {"audio": (audio, sample_rate)}},
        SamplingParams(temperature=0, max_tokens=1024),
    )[0]
    print(
        json.dumps(
            {
                "text": result.output.outputs[0].text,
                "words": [word.model_dump() for word in result.words],
                "error": result.error,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
