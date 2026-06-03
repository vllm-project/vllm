#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

smoke_threads="${VLLM_P550_THREADS:-4}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cpu}"
export VLLM_RVV_VLEN="${VLLM_RVV_VLEN:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$smoke_threads}"
export VLLM_CPU_OMP_THREADS_BIND="${VLLM_CPU_OMP_THREADS_BIND:-nobind}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-fork}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

model="${VLLM_P550_REAL_MODEL:-HuggingFaceTB/SmolLM2-135M-Instruct}"
prompt="${VLLM_P550_REAL_PROMPT:-What is 1+2? Answer with only the number.}"
expect="${VLLM_P550_EXPECT_SUBSTRING:-3}"
max_tokens="${VLLM_P550_REAL_MAX_TOKENS:-16}"
max_model_len="${VLLM_P550_REAL_MAX_MODEL_LEN:-256}"
kv_cache_bytes="${VLLM_P550_KV_CACHE_BYTES:-536870912}"

smoke_py="$(mktemp)"
trap 'rm -f "$smoke_py"' EXIT

cat >"$smoke_py" <<'PY'
from __future__ import annotations

import os
import platform
import sys

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.platforms import current_platform
from vllm.v1.attention.backends.cpu_attn import _get_attn_isa


def main() -> None:
    model, prompt, expect, max_tokens, max_model_len, kv_cache_bytes = sys.argv[1:7]
    max_tokens_i = int(max_tokens)
    max_model_len_i = int(max_model_len)
    kv_cache_bytes_i = int(kv_cache_bytes)

    print("machine:", platform.machine())
    print("VLLM_TARGET_DEVICE:", os.environ.get("VLLM_TARGET_DEVICE"))
    print("VLLM_RVV_VLEN:", os.environ.get("VLLM_RVV_VLEN", "<unset>"))
    print("kv_cache_memory_bytes:", kv_cache_bytes_i)
    print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
    print("model:", model)
    print("prompt:", prompt)

    import torch

    print("torch:", torch.__version__)
    print("vllm platform:", current_platform)
    print("cpu architecture:", current_platform.get_cpu_architecture())
    print("attention isa:", _get_attn_isa(torch.float32, 128, 64, "auto"))

    tokenizer = AutoTokenizer.from_pretrained(model)
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        rendered_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        rendered_prompt = f"User: {prompt}\nAssistant:"

    llm = LLM(
        model=model,
        tokenizer=model,
        dtype="float",
        max_model_len=max_model_len_i,
        max_num_seqs=1,
        enforce_eager=True,
        disable_log_stats=True,
        kv_cache_memory_bytes=kv_cache_bytes_i,
    )
    outputs = llm.generate(
        [rendered_prompt],
        SamplingParams(max_tokens=max_tokens_i, temperature=0.0),
        use_tqdm=False,
    )
    generated = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    print("generated:", generated)

    if expect and expect not in generated:
        raise SystemExit(
            f"expected substring {expect!r} was not found in generated text"
        )


if __name__ == "__main__":
    main()
PY

python "$smoke_py" \
    "$model" "$prompt" "$expect" "$max_tokens" "$max_model_len" "$kv_cache_bytes"
