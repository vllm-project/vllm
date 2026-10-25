# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline Ming ASR, including language hints, hotwords and dialogue context."""

import argparse
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.config.speech_to_text import SpeechToTextParams
from vllm.model_executor.models.ming_v3_asr import (
    MingV3ASRForConditionalGeneration as MingASR,
)
from vllm.multimodal.audio import AudioResampler
from vllm.multimodal.media.audio import load_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="inclusionAI/Ming-3.0-asr-tiny")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--language", choices=("zh", "en"))
    parser.add_argument("--prompt", default="")
    parser.add_argument("--hotwords")
    chunks = parser.add_mutually_exclusive_group()
    chunks.add_argument("--audio-chunk-frames", type=int)
    chunks.add_argument("--context-chunks", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=8192,
        max_num_seqs=4,
        enforce_eager=True,
        logits_processors=[
            (
                "vllm.model_executor.models.ming_v3_asr_logits_processor:"
                "NoRepeatNGramLogitsProcessor"
            )
        ],
    )
    audio, rate = load_audio(args.audio, sr=None, mono=True)
    audio = AudioResampler(16000).resample(audio, orig_sr=rate)
    processor_options = {"context_chunks": args.context_chunks}
    if args.audio_chunk_frames is not None:
        processor_options["audio_chunk_frames"] = args.audio_chunk_frames
    prompt = MingASR.get_generation_prompt(
        SpeechToTextParams(
            audio=audio,
            stt_config=MingASR.get_speech_to_text_config(
                llm.model_config, "transcribe"
            ),
            model_config=llm.model_config,
            language=args.language,
            request_prompt=args.prompt,
            hotwords=args.hotwords,
            extra_args=processor_options,
        )
    )
    outputs = llm.generate(
        prompt,
        SamplingParams(
            temperature=0,
            max_tokens=args.max_new_tokens,
            extra_args={"no_repeat_ngram_size": 10},
        ),
    )
    print(MingASR.post_process_output(outputs[0].outputs[0].text))


if __name__ == "__main__":
    main()
