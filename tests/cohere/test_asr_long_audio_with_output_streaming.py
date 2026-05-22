# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import asyncio
import json
import os
from pathlib import Path

import aiohttp
from evaluate import load
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from tests.models.registry import HF_EXAMPLE_MODELS
from tests.utils import RemoteOpenAIServer
from vllm.tokenizers import get_tokenizer

"""
This checks concurrent long audio (bw 23-45mins) transcription with output
streaming.
time pytest -sv \
    tests/cohere/test_asr_long_audio_with_output_streaming.py::\
test_asr_long_audio_with_output_streaming
"""

# Not all model tokenizers expose a normalize() helper, so use the Whisper
# English normalizer as a stable normalization path for WER.
normalizer_model_info = HF_EXAMPLE_MODELS.find_hf_info("openai/whisper-large-v3")
normalizer_tokenizer = get_tokenizer(
    "openai/whisper-large-v3",
    tokenizer_mode=normalizer_model_info.tokenizer_mode,
    trust_remote_code=normalizer_model_info.trust_remote_code,
)
normalizer = EnglishTextNormalizer(normalizer_tokenizer.english_spelling_normalizer)

ASR_MODEL_NAME = "CohereLabs/cohere-transcribe-03-2026"
ASR_MODEL_DIR_NAME = "cohere-transcribe-03-2026"
MAX_EXPECTED_WER = 0.5
LONG_AUDIO_TEST_CONCURRENCY = 3
MAX_AUDIO_CLIP_FILESIZE_MB = "50"
LONG_AUDIO_DATASET_DIR_NAME = "longform-audio-transcription"


def _get_server_model() -> str:
    engines_dir = os.environ.get("ENGINES_DIR", "/root/engines")
    local_model_dir = Path(engines_dir) / ASR_MODEL_DIR_NAME
    if local_model_dir.is_dir():
        return str(local_model_dir)
    return ASR_MODEL_NAME


def _get_long_audio_repo_dir() -> str:
    dataset_dir_override = os.environ.get("ASR_LONG_AUDIO_DATASET_DIR")
    if dataset_dir_override and Path(dataset_dir_override).is_dir():
        return dataset_dir_override

    data_dir = os.environ.get("DATA_DIR", "/root/data")
    local_dataset_dir = Path(data_dir) / LONG_AUDIO_DATASET_DIR_NAME
    if local_dataset_dir.is_dir():
        return str(local_dataset_dir)

    print(
        "Long-audio dataset not found locally. Run "
        '"DATA_DIR=/root/data ENGINES_DIR=/root/engines bash '
        'tests/cohere/scripts/download_checkpoints.sh asr" '
        "before running this script."
    )
    raise SystemExit(1)


def async_process(line):
    if type(line) is bytes:
        line = line.decode("utf-8")

    if not line.startswith("data: "):
        return None

    # End of stream
    if line == "data: [DONE]":
        print("\n[Stream finished]")
        return -1

    payload = line[len("data: ") :]

    data = json.loads(payload)

    choice = data["choices"][0]
    delta = choice["delta"]["content"]

    if choice.get("finish_reason") and choice["finish_reason"] != "stop":
        print(f"\n[Stream finished reason: {choice['finish_reason']}] != stop")
        return -1

    return delta


def sync_process(line):
    payload = line.decode("utf-8")
    data = json.loads(payload)

    text = data["text"]
    return text


async def stream_long_audio(
    session: aiohttp.ClientSession,
    data: tuple[str, str],
    model: str,
    streaming: bool,
    openai_api_base: str,
    api_key: str,
    concurrency: int,
    user_id: int,
):
    api_url = f"{openai_api_base}/audio/transcriptions"

    headers = {
        "User-Agent": "Transcription-Client",
        "Authorization": f"Bearer {api_key}",
    }

    audio_path, reference = data

    with open(audio_path, "rb") as f:
        # Prepare multipart form-data
        form = aiohttp.FormData()
        form.add_field("model", model)
        form.add_field("stream", f"{streaming}".lower())
        form.add_field("language", "en")
        form.add_field("response_format", "json")

        # Attach the file
        form.add_field(
            "file", f, filename=os.path.basename(audio_path), content_type="audio/mpeg"
        )

        # Output file for concurrent mode
        audio_name = os.path.basename(audio_path)
        output_file = f"{audio_name}_transcription_{user_id}.txt"
        buffer = ""

        if concurrency == 1:
            print(f"\n[User {user_id}] transcription result:", end=" ")
        else:
            print(
                f"\n[User {user_id}] transcription result will "
                f"be saved to {output_file}"
            )

        async with session.post(api_url, headers=headers, data=form) as response:
            if response.status != 200:
                text = await response.text()
                raise AssertionError(
                    f"[User {user_id}] Server returned {response.status}: {text}"
                )

            # Where to write (console or file)
            hypothesis = ""

            with open(output_file, "w") as f_out:
                async for chunk in response.content.iter_chunked(4096):
                    if not chunk:
                        continue

                    text = chunk.decode("utf-8", errors="replace")
                    buffer += text

                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        chunk_text = (
                            async_process(line) if streaming else sync_process(line)
                        )

                        if chunk_text == -1:
                            break

                        if chunk_text is None:
                            continue

                        if concurrency == 1:
                            print(chunk_text, end="", flush=True)
                        else:
                            f_out.write(chunk_text)
                            f_out.flush()

                        hypothesis += chunk_text

    # Evaluation
    wer_metric = load("wer")
    reference = normalizer(reference)
    hypothesis = normalizer(hypothesis)
    wer = 100 * wer_metric.compute(references=[reference], predictions=[hypothesis])
    print(f"\n[User {user_id}] WER: {wer:.4f}")
    assert wer < MAX_EXPECTED_WER, (
        f"[User {user_id}] Expected WER < {MAX_EXPECTED_WER}, got {wer:.4f}"
    )
    return wer


def get_long_audio_data():
    repo_dir = _get_long_audio_repo_dir()
    audio_dir = os.path.join(repo_dir, "audio")
    transcripts_dir = os.path.join(repo_dir, "transcripts")

    mapping = {
        "fireside_chat_w_mistral_ceo_arthur_mensch.mp3": (
            "nondiarized_fireside_chat_w_mistral_ceo_arthur_mensch_transcript.txt",
        ),
        "gpu_mode_karpathy.mp3": ("gpumode_irl_2024_karpathy_transcript.txt",),
        "state_of_gpt_BRK216HFS.mp3": ("state_of_gpt_BRK216HFS_transcript.txt",),
    }

    audio_paths = [
        os.path.join(audio_dir, fn)
        for fn in os.listdir(audio_dir)
        if fn.endswith(".mp3")
    ]

    transcript_paths = [
        os.path.join(transcripts_dir, fn)
        for fn in os.listdir(transcripts_dir)
        if fn.endswith(".txt")
    ]

    data = []
    transcript_names = [os.path.basename(tp) for tp in transcript_paths]

    for audio_path in audio_paths:
        audio_name = os.path.basename(audio_path)
        transcript_entry = mapping.get(audio_name)
        assert transcript_entry is not None, (
            f"No mapping found for audio file {audio_name}"
        )

        transcript_name = transcript_entry[0]
        assert transcript_name in transcript_names, (
            f"No mapping found for audio file {audio_name}"
        )

        with open(os.path.join(transcripts_dir, transcript_name)) as f:
            reference = f.read().strip()
        data.append((audio_path, reference))

    return data


async def main(args, openai_api_base: str = "http://localhost:8000/v1"):
    # data: [(audio_path, reference), ...]
    data = [(args.audio_path, "")] if args.audio_path else get_long_audio_data()

    # necessary to avoid timeout for long audio streaming
    timeout = aiohttp.ClientTimeout(
        total=None,  # let streaming run forever
        sock_read=600,  # 10-minute chunk wait
        sock_connect=60,  # 1-minute connect timeout
    )

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            asyncio.create_task(
                stream_long_audio(
                    session,
                    data[user_id % len(data)],
                    args.model,
                    args.streaming,
                    openai_api_base,
                    args.api_key,
                    args.concurrency,
                    user_id,
                )
            )
            for user_id in range(args.concurrency)
        ]

        return await asyncio.gather(*tasks)


def test_asr_long_audio_with_output_streaming():
    model_info = HF_EXAMPLE_MODELS.find_hf_info(ASR_MODEL_NAME)
    server_model = _get_server_model()
    server_args = [
        f"--served-model-name={ASR_MODEL_NAME}",
    ]
    if model_info.trust_remote_code:
        server_args.append("--trust-remote-code")

    args = argparse.Namespace(
        model=ASR_MODEL_NAME,
        concurrency=LONG_AUDIO_TEST_CONCURRENCY,
        api_key=RemoteOpenAIServer.DUMMY_API_KEY,
        audio_path=None,
        streaming=True,
    )

    with RemoteOpenAIServer(
        server_model,
        server_args,
        env_dict={"VLLM_MAX_AUDIO_CLIP_FILESIZE_MB": MAX_AUDIO_CLIP_FILESIZE_MB},
    ) as remote_server:
        results = asyncio.run(main(args, openai_api_base=remote_server.url_for("v1")))

    assert len(results) == LONG_AUDIO_TEST_CONCURRENCY
    assert all(result is not None for result in results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OpenAI Transcription Client using vLLM API Server"
    )
    parser.add_argument("--model", type=str, default="openai/whisper-large-v3-turbo")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--audio_path", type=str, default=None)
    parser.add_argument(
        "--streaming", action="store_true", help="Whether to use streaming mode"
    )
    args = parser.parse_args()

    asyncio.run(main(args))
