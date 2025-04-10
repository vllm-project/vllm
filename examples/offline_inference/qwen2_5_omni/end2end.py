#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import copy
import json
import os
import queue
import signal
import tempfile
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Union
from urllib.request import urlopen

import librosa
import numpy as np
import psutil
import requests
import resampy
import soundfile as sf
import torch
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from vllm.engine.arg_utils import nullable_str
from vllm.engine.async_llm_engine import AsyncEngineArgs
from vllm.engine.omni_llm_engine import OmniLLMEngine
from vllm.inputs import TextPrompt, TokensPrompt
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.multimodal.processing_omni import fetch_image, fetch_video
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

logger = init_logger('vllm.omni')

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--thinker-model', type=str, default=None)
parser.add_argument('--talker-model', type=str, default=None)
parser.add_argument('--code2wav-model', type=str, default=None)
parser.add_argument('--tokenize', action='store_true')
parser.add_argument('--legacy-omni-video', action='store_true')
parser.add_argument("--thinker-only", action="store_true")
parser.add_argument("--text-only", action="store_true")
parser.add_argument("--do-wave", action="store_true")
parser.add_argument('--max-num-seqs', type=int, default=64)
parser.add_argument('--block-size', type=int, default=16)
parser.add_argument('--enforce-eager', action='store_true')
parser.add_argument('--thinker-enforce-eager', action='store_true')
parser.add_argument('--talker-enforce-eager', action='store_true')
parser.add_argument('--enable-prefix-caching', action='store_true')
parser.add_argument('--thinker-quantization',
                    type=nullable_str,
                    choices=QUANTIZATION_METHODS)
parser.add_argument('--talker-quantization',
                    type=nullable_str,
                    choices=QUANTIZATION_METHODS)
parser.add_argument('--enable-torch-compile', action='store_true')
parser.add_argument('--enable-torch-compile-first-chunk', action='store_true')
parser.add_argument("--odeint-method",
                    type=str,
                    default="rk4",
                    choices=["euler", "rk4"])
parser.add_argument("--odeint-method-relaxed", action="store_true")
parser.add_argument("--batched-chunk", type=int, default=None)
parser.add_argument("--code2wav-frequency",
                    type=str,
                    default='50hz',
                    choices=['50hz'])
parser.add_argument('--voice-type', type=nullable_str, default='m02')
parser.add_argument('--warmup-voice-type', type=nullable_str, default='m02')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument("--max-tokens", type=int, default=2048)
parser.add_argument("--num-prompts", type=int, default=1)
parser.add_argument('--sample-rate', type=int, default=24000)
parser.add_argument('--use-torchvision', action='store_true')
parser.add_argument('--prompt',
                    choices=[
                        'text', 'audio', 'audio-long', 'audio-long-chunks',
                        'audio-long-expand-chunks', 'image', 'video',
                        'video-frames', 'audio-in-video', 'audio-in-video-v2',
                        "audio-multi-round", "badcase-vl", "badcase-text",
                        "badcase-image-early-stop", "badcase-two-audios",
                        "badcase-two-videos", "badcase-multi-round",
                        "badcase-voice-type", "badcase-voice-type-v2",
                        "badcase-audio-tower-1", "badcase-audio-only"
                    ],
                    default='text')

parser.add_argument('--thinker-devices', type=json.loads, default="[0]")
parser.add_argument('--talker-devices', type=json.loads, default="[0]")
parser.add_argument('--code2wav-devices', type=json.loads, default="[0]")
parser.add_argument('--code2wav-dynamic-batch',
                    action='store_true',
                    help='Enable code2wav dynamic batch')
parser.add_argument('--thinker-gpu-memory-utilization',
                    type=float,
                    default=0.4)
parser.add_argument('--talker-gpu-memory-utilization', type=float, default=0.4)

parser.add_argument('-o',
                    '--output-dir',
                    type=str,
                    default='.',
                    help="Audio output directory")

args = parser.parse_args()


def resample_wav_to_16khz(input_filepath):
    data, original_sample_rate = sf.read(input_filepath)
    # Only use the first channel
    if len(data.shape) > 1:
        data = data[:, 0]
    # resample to 16kHz
    data_resampled = resampy.resample(data,
                                      sr_orig=original_sample_rate,
                                      sr_new=16000)
    return data_resampled


def fetch_and_read_video(video_url: str, fps=2):
    import torchvision.io

    def read_video_with_torchvision(video_file_name: str):
        video, audio, info = torchvision.io.read_video(
            video_file_name,
            start_pts=0.0,
            end_pts=None,
            pts_unit="sec",
            output_format="TCHW",
        )

        total_frames, video_fps = video.size(0), info["video_fps"]
        total_duration = round(total_frames / video_fps, 3)
        nframes = int(total_frames / video_fps * fps)

        frame_timestamps = total_duration * torch.arange(1,
                                                         nframes + 1) / nframes
        grid_timestamps = frame_timestamps[::2]
        second_per_grid = grid_timestamps[1] - grid_timestamps[0]

        idx = torch.linspace(0, video.size(0) - 1, nframes).round().long()
        video_height, video_width = video.shape[2:]
        video = video[idx]

        if args.legacy_omni_video:
            return [video, total_duration, nframes, second_per_grid.item()]
        else:
            return video

    def read_video_with_transformers(video_file_name: Union[str, List[str]]):
        video, total_duration, nframes, second_per_grid = fetch_video(
            {'video': video_file_name})
        if total_duration is None and nframes is None:
            nframes = len(video)
            total_duration = 0.5 * nframes
            second_per_grid = 1.0
        if args.legacy_omni_video:
            return [video, total_duration, nframes, second_per_grid]
        else:
            return video

    def read_video(video_file_name: str):
        if args.use_torchvision:
            return read_video_with_torchvision(video_file_name)
        else:
            return read_video_with_transformers(video_file_name)

    if isinstance(video_url, str) and video_url.startswith("http"):
        with tempfile.NamedTemporaryFile(delete=True) as temp_video_file:
            resp = requests.get(video_url)
            assert resp.status_code == requests.codes.ok, f"Failed to fetch video from {video_url}, status_code:{resp.status_code}, resp:{resp}"

            temp_video_file.write(urlopen(video_url).read())
            temp_video_file_path = temp_video_file.name
            video_file_name = temp_video_file_path
            return read_video(video_file_name)
    else:
        video_file_name = video_url
        return read_video(video_file_name)


def make_inputs_qwen2_omni(
    messages: List[Dict[str, Union[str, List[Dict[str, str]]]]],
    use_audio_in_video: Optional[bool] = False,
    tokenize: bool = args.tokenize,
) -> Union[TokensPrompt, TextPrompt]:
    processor = AutoProcessor.from_pretrained(args.thinker_model)
    tokenizer = AutoTokenizer.from_pretrained(args.thinker_model)

    try:
        config = AutoConfig.from_pretrained(args.thinker_model)
        if 'Qwen2_5OmniModel' in config.architectures:
            args.legacy_omni_video = False
        else:
            args.legacy_omni_video = True
    except:
        args.legacy_omni_video = True

    audios, images, videos = [], [], []
    for message in messages:
        if not isinstance(message['content'], list):
            message['content'] = [{
                'type': 'text',
                'text': message['content'],
            }]
        index, num_contents = 0, len(message['content'])
        while index < num_contents:
            ele = message['content'][index]
            if 'type' not in ele:
                if 'text' in ele:
                    ele['type'] = 'text'
                elif 'audio' in ele:
                    ele['type'] = 'audio'
                elif 'audio_url' in ele:
                    ele['type'] = 'audio_url'
                elif 'image' in ele:
                    ele['type'] = 'image'
                elif 'image_url' in ele:
                    ele['type'] = 'image_url'
                elif 'video' in ele:
                    ele['type'] = 'video'
                elif 'video_url' in ele:
                    ele['type'] = 'video_url'
                else:
                    raise ValueError(f'Unknown ele: {ele}')

            if ele['type'] == 'audio' or ele['type'] == 'audio_url':
                if 'audio_url' in ele:
                    audio_key = 'audio_url'
                    with tempfile.NamedTemporaryFile(
                            delete=True) as temp_audio_file:
                        temp_audio_file.write(urlopen(ele[audio_key]).read())
                        temp_audio_file_path = temp_audio_file.name
                        audios.append(
                            resample_wav_to_16khz(temp_audio_file_path))
                        ele['audio'] = temp_audio_file_path
                elif 'audio' in ele:
                    audio_key = 'audio'
                    audios.append(resample_wav_to_16khz(ele[audio_key]))
                else:
                    raise ValueError(f'Unknown ele {ele}')
            elif use_audio_in_video and (ele['type'] == 'video'
                                         or ele['type'] == 'video_url'):
                # use video as audio as well
                if 'video_url' in ele:
                    audio_key = 'video_url'
                    with tempfile.NamedTemporaryFile(
                            delete=True) as temp_video_file:
                        temp_video_file.write(urlopen(ele[audio_key]).read())
                        temp_video_file_path = temp_video_file.name
                        ele[audio_key] = temp_video_file_path
                        audios.append(
                            librosa.load(temp_video_file_path, sr=16000)[0])
                        videos.append(
                            fetch_and_read_video(temp_video_file_path))
                        ele['video'] = temp_video_file_path
                elif 'video' in ele:
                    audio_key = 'video'
                    audios.append(librosa.load(ele[audio_key], sr=16000)[0])
                    videos.append(fetch_and_read_video(audio_key))
                else:
                    raise ValueError("Unknown ele {}".format(ele))
                # insert a audio after the video
                message['content'].insert(index + 1, {
                    "type": "audio",
                    "audio": ele[audio_key],
                })
                # no need to load the added audio again
                index += 1
            elif ele['type'] == 'video' or ele['type'] == 'video_url':
                if 'video_url' in ele:
                    video_key = 'video_url'
                    with tempfile.NamedTemporaryFile(
                            delete=True) as temp_video_file:
                        temp_video_file.write(urlopen(ele['video_url']).read())
                        temp_video_file_path = temp_video_file.name
                        videos.append(fetch_and_read_video(temp_video_file))
                        ele['video'] = temp_video_file_path
                else:
                    video_key = 'video'
                    videos.append(fetch_and_read_video(ele[video_key]))
            elif ele['type'] == 'image' or ele['type'] == 'image_url':
                images.append(fetch_image(ele))

            # move to the next content
            index += 1

    prompt = processor.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=True,
        add_vision_id=True,
    )

    audios = audios if len(audios) > 0 else None
    images = images if len(images) > 0 else None
    videos = videos if len(videos) > 0 else None

    logger.info(f'{prompt}, '
                f'audios = {len(audios) if audios else None}, '
                f'images = {len(images) if images else None}, '
                f'videos = {len(videos) if videos else None}')

    multi_modal_data = {}
    if audios:
        multi_modal_data["audio"] = audios
    if images:
        multi_modal_data["image"] = images
    if videos:
        multi_modal_data["video"] = videos

    # pass through the use_audio_in_video to llm engine
    multi_modal_data["use_audio_in_video"] = use_audio_in_video

    if isinstance(prompt, list) and isinstance(prompt[0], (list, str)):
        prompt = prompt[0]

    if tokenize:
        return TokensPrompt(
            prompt_token_ids=prompt,
            multi_modal_data=multi_modal_data,
        )
    else:
        return TextPrompt(
            prompt=prompt,
            multi_modal_data=multi_modal_data,
        )


def get_system_prompt():
    if args.text_only:
        return {
            'role': 'system',
            'content': [{
                'type': 'text',
                'text': 'You are a helpful assistant.'
            }]
        }
    else:
        return {
            'role':
            'system',
            'content': [{
                'type':
                'text',
                'text':
                'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'
            }]
        }


def make_text_prompt():
    messages = [
        get_system_prompt(),
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Who are you?"
                },
            ]
        },
    ]

    prompt = make_inputs_qwen2_omni(messages, )
    return prompt


def make_audio_in_video_v2_prompt():
    messages = [
        {
            'role':
            'system',
            'content': [{
                'type':
                'text',
                'text':
                'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'
            }]
        },
        {
            "role":
            "user",
            "content": [
                {
                    "type":
                    "video_url",
                    "video_url":
                    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw_small.mp4"
                },
            ]
        },
    ]
    prompt = make_inputs_qwen2_omni(
        messages,
        use_audio_in_video=True,
    )
    return prompt


def init_omni_engine():
    thinker_engine_args = AsyncEngineArgs(
        model=args.thinker_model,
        trust_remote_code=True,
        gpu_memory_utilization=args.thinker_gpu_memory_utilization,
        tensor_parallel_size=len(args.thinker_devices),
        enforce_eager=args.enforce_eager or args.thinker_enforce_eager,
        distributed_executor_backend="mp",
        limit_mm_per_prompt={
            'audio': 32,
            'image': 960,
            'video': 32
        },
        max_model_len=32768,
        max_num_seqs=args.max_num_seqs,
        block_size=args.block_size,
        quantization=args.thinker_quantization,
        enable_prefix_caching=args.enable_prefix_caching,
    )
    talker_engine_args = AsyncEngineArgs(
        model=args.talker_model,
        trust_remote_code=True,
        gpu_memory_utilization=args.talker_gpu_memory_utilization,
        tensor_parallel_size=1,
        enforce_eager=args.enforce_eager or args.talker_enforce_eager,
        distributed_executor_backend="mp",
        limit_mm_per_prompt={
            'audio': 32,
            'image': 960,
            'video': 32
        },
        max_model_len=32768,
        max_num_seqs=args.max_num_seqs,
        block_size=args.block_size,
        quantization=args.talker_quantization,
        enable_prefix_caching=args.enable_prefix_caching,
    )

    if args.thinker_only:
        return OmniLLMEngine(
            thinker_engine_args,
            thinker_visible_devices=args.thinker_devices,
        )
    elif not args.do_wave:
        return OmniLLMEngine(
            thinker_engine_args,
            talker_engine_args,
            thinker_visible_devices=args.thinker_devices,
            talker_visible_devices=args.talker_devices,
        )
    else:
        return OmniLLMEngine(
            thinker_engine_args,
            talker_engine_args,
            args.code2wav_model,
            code2wav_enable_torch_compile=args.enable_torch_compile,
            code2wav_enable_torch_compile_first_chunk=args.
            enable_torch_compile_first_chunk,
            code2wav_odeint_method=args.odeint_method,
            code2wav_odeint_method_relaxed=args.odeint_method_relaxed,
            code2wav_batched_chunk=args.batched_chunk,
            code2wav_frequency=args.code2wav_frequency,
            thinker_visible_devices=args.thinker_devices,
            talker_visible_devices=args.talker_devices,
            code2wav_visible_devices=args.code2wav_devices,
            code2wav_dynamic_batch=args.code2wav_dynamic_batch,
        )


def make_omni_prompt() -> Union[TokensPrompt, List[TokensPrompt]]:
    if args.prompt == 'text':
        prompt = make_text_prompt()
    elif args.prompt == 'audio-in-video-v2':
        prompt = make_audio_in_video_v2_prompt()
    else:
        raise ValueError(f'Unsupported prompt type: {args.prompt}')
    return prompt


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")


def parse_response(
    i: int,
    output_queue: queue.Queue[Union[RequestOutput, np.ndarray]],
):
    last_output: RequestOutput = None
    waveforms: List[np.ndarray] = []

    start_time = time.perf_counter()
    thinker_finish_time = start_time
    end_to_end_finish_time = start_time

    waveform_times = [start_time]

    while True:
        output = output_queue.get()
        if output is None:
            end_to_end_finish_time = time.perf_counter()
            break

        if isinstance(output, RequestOutput):
            if output.outputs[0].text:
                print(
                    f'[R-{i}][{now()}] Input: [{len(output.prompt_token_ids)}], '
                    f'Output: [{len(output.outputs[0].token_ids)}] {output.outputs[0].text}'
                )
            else:
                print(
                    f'[R-{i}][{now()}] Input: [{len(output.prompt_token_ids)}], '
                    f'Output: [{len(output.outputs[0].token_ids)}] {output.outputs[0].token_ids}'
                )
            if output.finished:
                thinker_finish_time = time.perf_counter()
            last_output = output
        elif isinstance(output, tuple) and isinstance(output[0], np.ndarray):
            output, output_tokens = output
            print(
                f'[R-{i}][{now()}] Waveform: [{len(waveforms)+1}] {output.dtype=}, {output.shape=}, {output_tokens=}'
            )
            waveforms.append(output)
            waveform_times.append(time.perf_counter())
        else:
            raise ValueError(f'[R-{i}] Unknown output type: {output}')

    print(
        f'[R-{i}] Thinker finished in {thinker_finish_time - start_time} seconds'
        f', end-to-end finished in {end_to_end_finish_time - start_time} seconds'
        f', {len(last_output.outputs[0].token_ids)} tokens'
        f', {len(waveforms)} waveforms')
    waveform_times = [
        waveform_times[i] - waveform_times[i - 1]
        for i in range(1, len(waveform_times))
    ]
    print(f'[R-{i}] Waveform times: {waveform_times} seconds')

    os.makedirs(args.output_dir, exist_ok=True)
    for j, waveform in enumerate(waveforms):
        tmp_wav_path = os.path.join(args.output_dir,
                                    f"waveform-{i}-chunk{j}.wav")
        sf.write(tmp_wav_path, waveform, samplerate=args.sample_rate)
        print(f"[R-{i}] Generated: {tmp_wav_path}")

    # Write the complete wave file
    print(
        f'[R-{i}] Writting waveforms to waveform.wav: {len(waveforms)} waveforms'
    )
    if len(waveforms) > 0:
        tmp_wav_path = os.path.join(args.output_dir, f"waveform-{i}.wav")
        sf.write(tmp_wav_path,
                 np.concatenate(waveforms),
                 samplerate=args.sample_rate)
        print(f"[R-{i}] Generated: {tmp_wav_path}")


def run_omni_engine(
    prompt: Union[TokensPrompt, List[TokensPrompt]],
    omni: OmniLLMEngine,
    num_prompts: int = 1,
    is_warmup: bool = False,
):
    sampling_params = SamplingParams(
        temperature=0.0,
        top_k=-1,
        top_p=1.0,
        repetition_penalty=1.1,
        max_tokens=args.max_tokens,
        detokenize=True,
        seed=args.seed,
    )
    talker_sampling_params = SamplingParams(
        temperature=0.9,
        top_k=40,
        top_p=0.8,
        repetition_penalty=1.05,
        max_tokens=2048,
        detokenize=False,
        seed=args.seed,
    )

    if not isinstance(prompt, list):
        prompts = [copy.deepcopy(prompt) for _ in range(num_prompts)]
    else:
        prompts = prompt

    # add request
    output_queues = []
    for i, prompt in enumerate(prompts):
        request_id = str(uuid.uuid4())
        logger.info(f'[R-{i}][{now()}] Adding request {request_id}')
        output_queue = omni.add_request(
            request_id,
            prompt,
            sampling_params,
            **{
                "talker_params": talker_sampling_params,
            } if not args.thinker_only else {},
            voice_type=args.warmup_voice_type
            if is_warmup else args.voice_type,
        )
        logger.info(f'[R-{i}][{now()}] Added request {request_id}')
        output_queues.append(output_queue)

    parse_threads = []
    for i, output_queue in enumerate(output_queues):
        t = threading.Thread(target=parse_response, args=(i, output_queue))
        t.start()
        parse_threads.append(t)
    for t in parse_threads:
        t.join()


def parse_voice_type(voice_type) -> str:
    if not voice_type:
        return voice_type

    voice_types = {
        "晨煦": "m02",
        "ethan": "m02",
        "千雪": "f030",
        "chelsie": "f030",
    }
    voice_type = voice_type.lower()
    return voice_types.get(voice_type, voice_type)


def main():
    if args.thinker_model is None or not os.path.exists(args.thinker_model):
        if os.path.exists(f'{args.model}/thinker'):
            args.thinker_model = f'{args.model}/thinker'
        else:
            args.thinker_model = f"{args.model}"
    if args.talker_model is None or not os.path.exists(args.talker_model):
        if os.path.exists(f'{args.model}/talker'):
            args.talker_model = f'{args.model}/talker'
        else:
            args.talker_model = f"{args.model}"
    if args.code2wav_model is None or not os.path.exists(args.code2wav_model):
        if os.path.exists(f'{args.model}/code2wav'):
            args.code2wav_model = f'{args.model}/code2wav'
        else:
            args.code2wav_model = f"{args.model}"

    prompt = make_omni_prompt()

    # init engine
    omni = init_omni_engine()
    args.voice_type = parse_voice_type(args.voice_type)
    args.warmup_voice_type = parse_voice_type(args.warmup_voice_type)
    logger.info("voice_type {} warmup_voice_type {}".format(
        args.voice_type, args.warmup_voice_type))

    # warmup
    run_omni_engine(prompt, omni, num_prompts=args.num_prompts, is_warmup=True)

    try:
        run_omni_engine(prompt, omni, args.num_prompts, is_warmup=False)
    except Exception as e:
        logger.exception('Error {e} in run_omni_engine')

    omni.shutdown()

    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        os.kill(child.pid, signal.SIGTERM)


if __name__ == '__main__':
    main()
