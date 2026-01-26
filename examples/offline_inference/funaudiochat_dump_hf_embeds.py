# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Dump FunAudioChat audio embeddings from transformers (LLaMA-Factory impl).

This helper is intended for debugging parity between vLLM and transformers by
comparing the projected audio embeddings inserted into `<|AUDIO|>` tokens.
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np


def _read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        sr = int(wf.getframerate())
        channels = int(wf.getnchannels())
        sampwidth = int(wf.getsampwidth())
        nframes = int(wf.getnframes())
        raw = wf.readframes(nframes)

    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sampwidth={sampwidth} bytes for wav: {path}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio, sr


def _build_prompt(model_dir: str, question: str, use_chat_template: bool) -> str:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    content = f"<|audio_bos|><|AUDIO|><|audio_eos|>\n{question}"

    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": content}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    return content


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--llamafactory-src", required=True)
    p.add_argument("--question", default="请转写这段音频。")
    p.add_argument("--use-chat-template", action="store_true")
    p.add_argument("--omit-sampling-rate", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    args = p.parse_args()

    args.model = str(Path(args.model))
    args.audio = str(Path(args.audio))
    args.out = str(Path(args.out))

    sys.path.insert(0, args.llamafactory_src)
    from llamafactory.model.funaudiochat import register_funaudiochat

    register_funaudiochat()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoProcessor

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    audio, sr = _read_wav_mono(args.audio)
    prompt = _build_prompt(args.model, args.question, args.use_chat_template)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=False)
    if args.omit_sampling_rate:
        inputs = processor(text=[prompt], audio=[audio], return_tensors="pt")
    else:
        inputs = processor(
            text=[prompt],
            audio=[audio],
            sampling_rate=int(sr),
            return_tensors="pt",
        )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=False,
    ).to(device)
    model.eval()

    speech_ids = inputs["speech_ids"].to(device)
    speech_attention_mask = inputs["speech_attention_mask"].to(device)
    input_features = inputs["input_features"].to(device)
    feature_attention_mask = inputs["feature_attention_mask"].to(device)
    feature_exist_mask = inputs.get("feature_exist_mask", None)
    if feature_exist_mask is None:
        feature_exist_mask = torch.ones((speech_ids.shape[0],), dtype=torch.bool)
    feature_exist_mask = feature_exist_mask.to(device)

    group_size = int(getattr(model.audio_tower, "group_size", 5))
    pad_id = int(getattr(model, "audio_pad_token_id", 0))
    target_len = ((speech_ids.shape[-1] + group_size - 1) // group_size) * group_size
    if target_len > speech_ids.shape[-1]:
        pad_len = target_len - speech_ids.shape[-1]
        speech_ids = torch.nn.functional.pad(speech_ids, (0, pad_len), value=pad_id)

    with torch.no_grad():
        cont, cont_out_lens = model.get_audio_features(
            input_features,
            feature_attention_mask=feature_attention_mask,
            speech_maxlen=int(speech_ids.shape[-1]),
        )
        disc, *_ = model.audio_tower(
            speech_ids,
            continuous_audio_features=cont,
            continuous_audio_output_lengths=cont_out_lens,
            feature_exist_mask=feature_exist_mask,
        )
        _, out_lens = model.audio_tower._get_feat_extract_output_lengths(speech_attention_mask.sum(-1))
        l0 = int(out_lens[0].item())
        arr = disc[0, :l0].detach().float().cpu().numpy()
        cont_arr = cont.detach().float().cpu().numpy()

    np.save(args.out, arr)
    np.save(args.out.replace(".npy", "_cont.npy"), cont_arr)
    print(f"saved {arr.shape} -> {args.out}")


if __name__ == "__main__":
    main()
