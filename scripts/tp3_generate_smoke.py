# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from vllm import LLM, SamplingParams


def main() -> None:
    model = Path(
        "/home/s0l/.cache/huggingface/hub/"
        "models--mconcat--Qwopus3.6-27B-v2-NVFP4/"
        "snapshots/497ecd00bfddd3d049c46547946dce1817bc93e5"
    )
    llm = LLM(
        model=str(model),
        tensor_parallel_size=3,
        max_model_len=1024,
        max_num_seqs=1,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        trust_remote_code=True,
        enable_prefix_caching=False,
        language_model_only=True,
        linear_backend="cutlass",
        attention_config={"backend": "TRITON_ATTN"},
    )
    outputs = llm.generate(
        ["Say hello in one short sentence."],
        SamplingParams(max_tokens=8, temperature=0.0),
    )
    print("GENERATED:", outputs[0].outputs[0].text, flush=True)


if __name__ == "__main__":
    main()
