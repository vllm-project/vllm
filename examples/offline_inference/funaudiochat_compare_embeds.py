# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Compare FunAudioChat audio embeddings between:
  - transformers (LLaMA-Factory FunAudioChat implementation)
  - vLLM's FunAudioChat port (encoder/projector only)

This is a debugging helper to narrow down parity issues.
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


def _pad_to_multiple(
    ids, mask, *, multiple: int, pad_id: int
):  # torch.Tensor -> torch.Tensor, torch.Tensor
    import torch

    target_len = ((ids.shape[-1] + multiple - 1) // multiple) * multiple
    if target_len <= ids.shape[-1]:
        return ids, mask
    pad_len = target_len - ids.shape[-1]
    ids = torch.nn.functional.pad(ids, (0, pad_len), value=pad_id)
    mask = torch.nn.functional.pad(mask, (0, pad_len), value=0)
    return ids, mask


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--llamafactory-src", required=True)
    p.add_argument("--question", default="请转写这段音频。")
    p.add_argument("--use-chat-template", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    args = p.parse_args()

    args.model = str(Path(args.model))
    args.audio = str(Path(args.audio))

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
    audio_pad_id = int(getattr(model, "audio_pad_token_id", 0))
    speech_ids, speech_attention_mask = _pad_to_multiple(
        speech_ids, speech_attention_mask, multiple=group_size, pad_id=audio_pad_id
    )

    # === HF (transformers) continuous tower ===
    with torch.no_grad():
        hf_cont, hf_cont_out_lens = model.get_audio_features(
            input_features,
            feature_attention_mask=feature_attention_mask,
            speech_maxlen=int(speech_ids.shape[-1]),
        )

    # === vLLM towers (load weights from HF modules) ===
    from vllm.model_executor.models.funaudiochat import (
        FunAudioChatAudioEncoder,
        FunAudioChatDiscreteEncoder,
    )

    v_cont = FunAudioChatAudioEncoder(model.config.audio_config).to(
        device=device, dtype=dtype
    )
    v_disc = FunAudioChatDiscreteEncoder(model.config.audio_config).to(
        device=device, dtype=dtype
    )

    v_cont.load_state_dict(model.continuous_audio_tower.state_dict(), strict=True)
    v_disc.load_state_dict(model.audio_tower.state_dict(), strict=True)
    v_cont.eval()
    v_disc.eval()

    # vLLM continuous tower forward (match HF get_audio_features flattening)
    with torch.no_grad():
        if feature_attention_mask.shape[1] != input_features.shape[-1]:
            min_len = min(
                int(feature_attention_mask.shape[1]), int(input_features.shape[-1])
            )
            feature_attention_mask = feature_attention_mask[:, :min_len]
            input_features = input_features[:, :, :min_len]

        feature_lens = feature_attention_mask.sum(-1)
        flat_features = input_features.permute(0, 2, 1)[
            feature_attention_mask.bool()
        ].permute(1, 0)
        aftercnn_lens, v_cont_out_lens = v_cont._get_feat_extract_output_lengths(
            feature_lens
        )
        v_cont_out = v_cont(
            flat_features,
            feature_lens=feature_lens,
            aftercnn_lens=aftercnn_lens,
            speech_maxlen=int(speech_ids.shape[-1]),
        ).last_hidden_state

    # Compare continuous tower outputs
    cont_max = (hf_cont - v_cont_out).abs().max().item()
    cont_mse = ((hf_cont - v_cont_out) ** 2).mean().item()
    print(
        f"continuous: shape={tuple(hf_cont.shape)} "
        f"max_abs_diff={cont_max:.6g} mse={cont_mse:.6g}"
    )

    # Compare discrete tower outputs using the SAME continuous features (HF)
    with torch.no_grad():
        hf_disc_out, *_ = model.audio_tower(
            speech_ids,
            continuous_audio_features=hf_cont,
            continuous_audio_output_lengths=hf_cont_out_lens,
            feature_exist_mask=feature_exist_mask,
        )
        v_disc_out = v_disc(
            speech_ids,
            continuous_audio_features=hf_cont,
            continuous_audio_output_lengths=hf_cont_out_lens,
            feature_exist_mask=feature_exist_mask,
        )

        get_out_lens = model.audio_tower._get_feat_extract_output_lengths
        _, out_lens = get_out_lens(speech_attention_mask.sum(-1))
        lengths = [int(x) for x in out_lens.tolist()]

    for i, length in enumerate(lengths):
        a = hf_disc_out[i, :length]
        b = v_disc_out[i, :length]
        max_diff = (a - b).abs().max().item()
        mse = ((a - b) ** 2).mean().item()
        print(
            f"discrete[{i}]: shape={tuple(a.shape)} "
            f"max_abs_diff={max_diff:.6g} mse={mse:.6g}"
        )


if __name__ == "__main__":
    main()
