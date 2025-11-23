import asyncio
import argparse
import os
import json
import aiohttp
from evaluate import load
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

def async_process(line):
    
    if type(line) == bytes:
        line = line.decode("utf-8")

    if not line.startswith("data: "):
        return None
    
    # End of stream
    if line == "data: [DONE]":
        print("\n[Stream finished]")
        return -1

    payload = line[len("data: "):]

    data = json.loads(payload)

    choice = data["choices"][0]
    delta = choice["delta"]["content"]

    # import pdb; pdb.set_trace()
    
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
        tokenizer,
        openai_api_base: str,
        api_key: str,
        concurrency: int,
        user_id: int
    ):

    api_url = f"{openai_api_base}/audio/transcriptions"

    headers = {
        "User-Agent": "Transcription-Client",
        "Authorization": f"Bearer {api_key}",
    }

    # Prepare multipart form-data
    form = aiohttp.FormData()
    form.add_field("model", model)
    form.add_field("stream", f"{args.streaming}".lower())
    form.add_field("language", "en")
    form.add_field("response_format", "json")

    audio_path, reference = data

    # Attach the file
    form.add_field(
        "file",
        open(audio_path, "rb"),
        filename=os.path.basename(audio_path),
        content_type="audio/mpeg"
    )

    # Output file for concurrent mode
    audio_name = os.path.basename(audio_path)
    output_file = f"{audio_name}_transcription_{user_id}.txt"
    buffer = ""

    print(f"\n[User {user_id}] transcription result:", end=" ")
    async with session.post(api_url, headers=headers, data=form) as response:

        if response.status != 200:
            text = await response.text()
            print(f"\nError from server: {response.status}\n{text}")
            return

        # Where to write (console or file)
        f_out = None

        if concurrency > 1:
            f_out = open(output_file, "w")

        # async for raw_line in response.content:
        #     line = raw_line.strip()
            
        #     if not line:
        #         continue

        hypothesis = ""

        async for chunk in response.content.iter_chunked(4096):
            if not chunk:
                continue

            # line = chunk.decode("utf-8", errors="ignore")
            text = chunk.decode("utf-8", errors="ignore")
            buffer += text

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if args.streaming:
                    data = async_process(line)
                else:
                    data = sync_process(line)

                if data == -1:
                    break

                if data is None:
                    continue

                if concurrency == 1:
                    print(data, end="", flush=True)
                else:
                    f_out.write(data)
                    f_out.flush()

                hypothesis += data


        if f_out:
            f_out.close()

    # Evaluation
    if reference:
        wer_metric = load("wer")
        reference = tokenizer.normalize(reference)
        hypothesis = tokenizer.normalize(hypothesis)
        wer = 100 * wer_metric.compute(references=[reference], predictions=[hypothesis])
        print(f"\n[User {user_id}] WER: {wer:.4f}")
    else:
        print(f"\n[User {user_id}] No reference provided, skipping WER computation.")


def get_long_audio_data(model):
    repo_dir = snapshot_download("amgadhasan/longform-audio-transcription", repo_type="dataset")
    audio_dir = os.path.join(repo_dir, "audio")
    transcripts_dir = os.path.join(repo_dir, "transcripts")

    mapping = {
        "fireside_chat_w_mistral_ceo_arthur_mensch.mp3": "nondiarized_fireside_chat_w_mistral_ceo_arthur_mensch_transcript.txt",
        "gpu_mode_karpathy.mp3": "gpumode_irl_2024_karpathy_transcript.txt",
        "state_of_gpt_BRK216HFS.mp3": "state_of_gpt_BRK216HFS_transcript.txt"
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
        transcript_name = mapping.get(audio_name, None)
        assert transcript_name is not None and transcript_name in transcript_names, \
            f"No mapping found for audio file {audio_name}"

        reference = open(
            os.path.join(transcripts_dir, transcript_name),
            "r"
        ).read().strip()
        data.append((audio_path, reference))

    return data
        


async def main(args):

    openai_api_base = "http://localhost:8000/v1"

    if args.audio_path:
        data = [args.audio_path, ""]
    else:
        data = get_long_audio_data(args.model)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # necessary to avoid timeout for long audio streaming
    timeout = aiohttp.ClientTimeout(
        total=None,        # let streaming run forever
        sock_read=600,     # 10-minute chunk wait
        sock_connect=60    # 1-minute connect timeout
    )

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            asyncio.create_task(
                stream_long_audio(
                    session,
                    data[user_id % len(data)],
                    # data[-1],
                    args.model,
                    tokenizer,
                    openai_api_base,
                    args.api_key,
                    args.concurrency,
                    user_id
                )
            )
            for user_id in range(args.concurrency)
        ]

        await asyncio.gather(*tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI Transcription Client using vLLM API Server")
    parser.add_argument("--model", type=str, default="openai/whisper-large-v3-turbo")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--audio_path", type=str, default=None)
    parser.add_argument("--streaming", action="store_true", help="Whether to use streaming mode")
    args = parser.parse_args()

    asyncio.run(main(args))