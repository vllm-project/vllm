#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
import soundfile as sf
import torch

from vllm.engine.arg_utils import nullable_str
from vllm.logger import init_logger
from vllm.model_executor.models.qwen2_code2wav_dit import Qwen2Code2wav

logger = init_logger('vllm.omni')

parser = argparse.ArgumentParser()
parser.add_argument('--code2wav-model',
                    type=str,
                    default=os.path.expanduser("~/models/omni-v4/code2wav"))
parser.add_argument('--input-json',
                    type=str,
                    default=os.path.expanduser("~/vllm/generated-codes.json"))
parser.add_argument('--voice-type', type=nullable_str, default='default')
parser.add_argument("--batched-chunk", type=int, default=None)
parser.add_argument("--frequency", type=str, default='50hz', choices=['50hz'])
parser.add_argument('--sample-rate', type=int, default=24000)
parser.add_argument('--warmup', type=int, default=1)
parser.add_argument('--concurrency', type=int, default=1)
parser.add_argument('--enable-torch-compile', action='store_true')
parser.add_argument('--enable-torch-compile-first-chunk', action='store_true')
parser.add_argument("--odeint-method",
                    type=str,
                    default="rk4",
                    choices=["euler", "rk4"])
parser.add_argument('--multi-waveforms', action='store_true')

args = parser.parse_args()


def process_code(
    code: List[int],
    code2wav: Qwen2Code2wav,
    code2wav_cond: torch.Tensor,
    code2wav_ref_mel: torch.Tensor,
    code2wav_y_all: torch.Tensor,
    code2wav_steps: int,
    device: torch.device,
) -> List[np.ndarray]:
    # start the code2wav thread
    code = torch.tensor(code, dtype=torch.long, device=device).reshape(1, -1)
    progress, prev_generated, waveforms = 0, None, []
    for i in range(code.size(1)):
        finished = i == code.size(1) - 1
        chunk_code_length = i * (2 if args.frequency == "50hz" else
                                 4) - code2wav.future_cache_size
        if (chunk_code_length > 0
                and chunk_code_length % code2wav.chunk_size == 0) or finished:
            start_chunk_time = time.perf_counter()

            if progress == 0 and finished:
                process_chunk = code2wav.process_little_chunk
            else:
                process_chunk = code2wav.process_chunk

            prev_generated, audio = process_chunk(
                code2wav_cond,
                code2wav_ref_mel,
                codec_all=code,
                y_all=code2wav_y_all,
                i=progress,
                steps=code2wav_steps,
                prev_generated=prev_generated,
                finished=finished,
            )
            progress += 1
            waveforms.append(audio)

            end_chunk_time = time.perf_counter()
            print(
                f'Chunk {progress} took {end_chunk_time - start_chunk_time} seconds'
            )
    return [waveform.detach().cpu().numpy() for waveform in waveforms]


def main():
    # code2wav model
    model_path = args.code2wav_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # code2wav model
    dit_model_path = glob.glob(os.path.join(model_path, 'dit',
                                            'model_*.pt'))[0]
    bigvgan_model_path = glob.glob(os.path.join(model_path, 'bigvgan',
                                                'g_*'))[0]

    def parse_key(fname, key):
        if fname == key:
            return 'default'
        return fname.split('_')[0]

    code2wav_conds = {
        parse_key(os.path.basename(f), 'spk_emb.npy'):
        torch.tensor(np.load(f)).to(device)
        for f in sorted(
            glob.glob(os.path.join(model_path, 'inputs', '*spk_emb.npy')) +
            glob.glob(
                os.path.join(model_path, 'inputs_sft4spks', '*spk_emb.npy')))
    }
    code2wav_ref_mels = {
        parse_key(os.path.basename(f), 'ref_mel.npy'):
        torch.tensor(np.load(f)).to(device)
        for f in sorted(
            glob.glob(os.path.join(model_path, 'inputs', '*ref_mel.npy')) +
            glob.glob(
                os.path.join(model_path, 'inputs_sft4spks', '*ref_mel.npy')))
    }

    if 'default' not in code2wav_conds:
        code2wav_conds['default'] = list(code2wav_conds.values())[0]
    if 'default' not in code2wav_ref_mels:
        code2wav_ref_mels['default'] = list(code2wav_ref_mels.values())[0]

    code2wav_cond = code2wav_conds[args.voice_type]
    code2wav_ref_mel = code2wav_ref_mels[args.voice_type]

    if args.batched_chunk is None:
        if args.frequency == "50hz":
            args.batched_chunk = 2
        else:
            args.batched_chunk = 1
    args.frequency = args.frequency

    code2wav_steps: int = 10
    code2wav_bs_mel: int = 24 if args.frequency == "50hz" else 32
    code2wav = Qwen2Code2wav(dit_ckpt=dit_model_path,
                             bigvgan_ckpt=bigvgan_model_path,
                             steps=code2wav_steps,
                             bs_mel=code2wav_bs_mel,
                             odeint_method=args.odeint_method,
                             batched_chunk=args.batched_chunk,
                             frequency=args.frequency,
                             device=device)

    if args.enable_torch_compile:
        code2wav.enable_torch_compile(args.enable_torch_compile_first_chunk)

    # read the inputs
    with open(args.input_json) as f:
        code = json.load(f)

    code2wav_y_all = torch.randn(args.concurrency,
                                 1,
                                 32768,
                                 80,
                                 device=device,
                                 dtype=code2wav_ref_mel.dtype)

    start_time = time.perf_counter()

    # warmup
    for _ in range(args.warmup):
        process_code(
            code,
            code2wav,
            code2wav_cond,
            code2wav_ref_mel,
            code2wav_y_all[0],
            code2wav_steps,
            device,
        )

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = []
        for i in range(args.concurrency):
            futures.append(
                executor.submit(
                    process_code,
                    code,
                    code2wav,
                    code2wav_cond,
                    code2wav_ref_mel,
                    code2wav_y_all[i],
                    code2wav_steps,
                    device,
                ))

        waveforms = []
        for future in futures:
            waveforms.append(future.result())
        waveforms = waveforms[0]

    end_time = time.perf_counter()
    print(f"Code2wav for {args.concurrency} times "
          f"took {end_time - start_time} seconds "
          f"for {len(code)} tokens, {len(waveforms)} waveforms")

    print('Writting waveforms to output.wav')
    if args.multi_waveforms:
        for i, waveform in enumerate(waveforms):
            sf.write(f'output_{i}.wav', waveform, samplerate=args.sample_rate)
    else:
        sf.write('output.wav',
                 np.concatenate(waveforms),
                 samplerate=args.sample_rate)

    end_write_time = time.perf_counter()
    print(f'Writing waveforms took {end_write_time - end_time} seconds')


if __name__ == '__main__':
    main()
