# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline speaker diarization with NVIDIA Nemotron-3-Diarization.

Each request is one whole recording. The model returns the logits of up to 8
speakers for every 10 ms frame, `[num_frames, 8]`, and the Transformers processor
turns them into speech segments.

Usage::

    python nemotron3_diarization_offline.py --audio meeting.wav
"""

from argparse import Namespace

from transformers import AutoProcessor

from vllm import LLM, EngineArgs
from vllm.assets.audio import AudioAsset
from vllm.multimodal.media.audio import load_audio
from vllm.utils.argparse_utils import FlexibleArgumentParser


def parse_args():
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to the recording to diarize. Defaults to a vLLM audio asset.",
    )
    parser.set_defaults(
        model="nvidia/Nemotron-3-Diarization",
        runner="pooling",
        skip_tokenizer_init=True,
        dtype="float32",
    )
    return parser.parse_args()


def main(args: Namespace):
    audio_path = args.__dict__.pop("audio")
    if audio_path is None:
        audio, sample_rate = AudioAsset("winning_call").audio_and_sample_rate
    else:
        audio, sample_rate = load_audio(audio_path, sr=None)

    llm = LLM(**vars(args))
    processor = AutoProcessor.from_pretrained(args.model)

    # The model reads the audio only; the prompt is a placeholder token.
    outputs = llm.encode(
        [
            {
                "prompt_token_ids": [0],
                "multi_modal_data": {"audio": (audio, sample_rate)},
            }
        ],
        pooling_task="plugin",
    )

    for output in outputs:
        logits = output.outputs.data  # [num_frames, num_speakers]
        for segment in processor.extract_speaker_dict(logits[None])[0]:
            print(
                f"speaker {segment['Speaker']}: "
                f"{segment['Start']:.2f}s - {segment['End']:.2f}s"
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
