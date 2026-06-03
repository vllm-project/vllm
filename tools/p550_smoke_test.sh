#!/usr/bin/env bash
set -euo pipefail

cores="$(nproc 2>/dev/null || printf '1')"
if [ "$cores" -gt 1 ]; then
    smoke_threads="$((cores - 1))"
else
    smoke_threads=1
fi

export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cpu}"
export VLLM_CPU_KVCACHE_SPACE="${VLLM_CPU_KVCACHE_SPACE:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$smoke_threads}"
export VLLM_CPU_OMP_THREADS_BIND="${VLLM_CPU_OMP_THREADS_BIND:-nobind}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-fork}"

local_model="${PWD}/.p550_models/tiny-llama"
if [ -n "${VLLM_P550_MODEL:-}" ]; then
    model="${VLLM_P550_MODEL}"
elif [ -d "$local_model" ]; then
    model="$local_model"
else
    model="hmellor/tiny-random-LlamaForCausalLM"
fi

smoke_py="$(mktemp)"
trap 'rm -f "$smoke_py"' EXIT

cat >"$smoke_py" <<'PY'
import os
import platform
import sys


def main() -> None:
    model = sys.argv[1]

    print("machine:", platform.machine())
    print("VLLM_TARGET_DEVICE:", os.environ.get("VLLM_TARGET_DEVICE"))
    print("VLLM_RVV_VLEN:", os.environ.get("VLLM_RVV_VLEN", "<unset>"))
    print("VLLM_CPU_KVCACHE_SPACE:", os.environ.get("VLLM_CPU_KVCACHE_SPACE"))
    print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
    print("VLLM_CPU_OMP_THREADS_BIND:", os.environ.get("VLLM_CPU_OMP_THREADS_BIND"))

    try:
        import torch

        print("torch:", torch.__version__)
    except Exception as exc:
        raise SystemExit(f"torch import failed: {type(exc).__name__}: {exc}") from exc

    try:
        from vllm.platforms import current_platform
        from vllm.v1.attention.backends.cpu_attn import _get_attn_isa

        print("vllm platform:", current_platform)
        print("cpu architecture:", current_platform.get_cpu_architecture())
        print("attention isa:", _get_attn_isa(torch.float32, 128, 64, "auto"))
    except Exception as exc:
        print("attention isa probe failed:", type(exc).__name__, exc)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        dtype="float",
        max_model_len=64,
        max_num_seqs=1,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(max_tokens=4, temperature=0.0)
    outputs = llm.generate(["Hello from P550"], sampling_params)

    for output in outputs:
        print("prompt:", output.prompt)
        for item in output.outputs:
            print("generated:", item.text)


if __name__ == "__main__":
    main()
PY

python "$smoke_py" "$model"
