# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end inference helper for FunAudioChat (audio -> text).

Supports running via:
  - vLLM (backend=vllm)
  - transformers + LLaMA-Factory registration (backend=transformers)

Example (vLLM):
  python examples/offline_inference/funaudiochat.py \\
    --backend vllm \\
    --model /path/to/funaudiochat_model \\
    --audio /path/to/sample.wav

Example (transformers; requires LLaMA-Factory code available):
  conda run -n llamafactory python examples/offline_inference/funaudiochat.py \\
    --backend transformers \\
    --llamafactory-src /path/to/LLaMA-Factory/src \\
    --model /path/to/funaudiochat_model \\
    --audio /path/to/sample.wav
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


def _build_prompt(
    model_dir: str,
    question: str,
    use_chat_template: bool,
    *,
    include_audio: bool,
    trust_remote_code: bool,
) -> str:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=trust_remote_code
    )
    if include_audio:
        content = f"<|audio_bos|><|AUDIO|><|audio_eos|>\n{question}"
    else:
        content = question

    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": content}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    return content


def _run_vllm(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    prompt = _build_prompt(
        args.model,
        args.question,
        args.use_chat_template,
        include_audio=not args.text_only,
        trust_remote_code=args.trust_remote_code,
    )

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"audio": 1} if not args.text_only else {},
        enforce_eager=args.enforce_eager,
        trust_remote_code=args.trust_remote_code,
    )

    req: dict[str, object] = {"prompt": prompt}
    if not args.text_only:
        audio, sr = _read_wav_mono(args.audio)
        # Passing sampling rate enables vLLM to auto-resample audio when needed
        # (and avoids silently assuming 16kHz).
        req["multi_modal_data"] = {"audio": (audio, float(sr))}

    outputs = llm.generate(
        [req],
        sampling_params=SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
        ),
    )

    req_out = outputs[0]
    comp = req_out.outputs[0]

    if args.debug_mm:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=args.trust_remote_code
        )
        if req_out.prompt_token_ids is not None:
            print("=== vLLM prompt (decoded) ===")
            print(tokenizer.decode(req_out.prompt_token_ids, skip_special_tokens=False))
            print(f"prompt_len={len(req_out.prompt_token_ids)}")
        print("=== vLLM output (decoded) ===")
        print(tokenizer.decode(comp.token_ids, skip_special_tokens=False))
        print(f"output_len={len(comp.token_ids)}")
        print(f"mm_placeholders={req_out.multi_modal_placeholders}")

    print(comp.text)


def _run_transformers(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoProcessor, AutoTokenizer

    if args.llamafactory_src:
        sys.path.insert(0, args.llamafactory_src)
        from llamafactory.model.funaudiochat import register_funaudiochat

        register_funaudiochat()

    device = torch.device(args.device)
    prompt = _build_prompt(
        args.model,
        args.question,
        args.use_chat_template,
        include_audio=not args.text_only,
        trust_remote_code=args.trust_remote_code,
    )

    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float16,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()

    if args.text_only:
        inputs = processor(text=[prompt], return_tensors="pt")
    else:
        audio, sr = _read_wav_mono(args.audio)
        inputs = processor(
            text=[prompt],
            audio=[audio],
            sampling_rate=int(sr),
            return_tensors="pt",
        )
    inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=int(args.max_tokens),
        )

    token_ids = out_ids
    if hasattr(token_ids, "sequences"):
        token_ids = token_ids.sequences
    if isinstance(token_ids, (torch.Tensor, np.ndarray)):
        token_ids = token_ids.tolist()

    # Normalize arbitrary nesting to a single token id list.
    while (
        isinstance(token_ids, (list, tuple))
        and token_ids
        and not isinstance(token_ids[0], (int, np.integer))
    ):
        token_ids = token_ids[0]
    if isinstance(token_ids, (torch.Tensor, np.ndarray)):
        token_ids = token_ids.tolist()

    def _flatten_token_ids(x) -> list[int]:
        if isinstance(x, (int, np.integer)):
            return [int(x)]
        if isinstance(x, torch.Tensor):
            return _flatten_token_ids(x.tolist())
        if isinstance(x, np.ndarray):
            return _flatten_token_ids(x.tolist())
        if isinstance(x, (list, tuple)):
            out: list[int] = []
            for y in x:
                out.extend(_flatten_token_ids(y))
            return out
        raise TypeError(f"Unsupported token id container: {type(x)}")

    try:
        text = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
        )
    except TypeError:
        text = tokenizer.decode(
            _flatten_token_ids(token_ids), skip_special_tokens=False
        )
    print(text)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["vllm", "transformers"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--question", default="请转写这段音频。")
    p.add_argument("--use-chat-template", action="store_true")
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--debug-mm", action="store_true")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--llamafactory-src", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    args = p.parse_args()

    args.model = str(Path(args.model))
    args.audio = str(Path(args.audio))

    if args.backend == "vllm":
        _run_vllm(args)
    else:
        _run_transformers(args)


if __name__ == "__main__":
    main()
