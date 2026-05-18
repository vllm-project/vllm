# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hybrid co-location with partial rollout — selective offload example.

This example demonstrates the `offload_tags` parameter on `LLM.sleep`,
which enables tag-wise selective offload. The driving use case is an RL
training loop in which a trainer and an inference engine share the same
GPUs ("hybrid co-location"):

  1. Inference produces a partial rollout (some tokens generated; the
     prompt + generated prefix lives in the KV cache).
  2. The trainer needs the GPU, so the inference engine must release
     its weights' GPU memory — but it would be wasteful to also discard
     the KV cache, since the rollout is only partial.
  3. After the training step, the engine reloads its weights and
     resumes generation. Because the KV cache is intact, the prefill
     stage for the in-flight prompt does NOT need to be repeated.

`offload_tags=["weights"]` expresses exactly this: offload weights to
CPU while keeping the KV cache mapped on GPU. Conversely,
`offload_tags=["kv_cache"]` keeps the weights resident and frees the
KV cache (useful when switching to a different mode of usage), and
`offload_tags=["weights", "kv_cache"]` is equivalent (in memory effect)
to the legacy `level=1` sleep.

Run:
    python examples/features/pause_resume/hybrid_partial_rollout.py
"""

from __future__ import annotations

import torch

from vllm import LLM, SamplingParams


def _gib(n: int) -> str:
    return f"{n / (1024**3):.2f} GiB"


def main() -> None:
    llm = LLM(
        model="hmellor/tiny-random-LlamaForCausalLM",
        enable_sleep_mode=True,
        enforce_eager=True,
    )

    prompt = "The capital of France is"
    sampling = SamplingParams(temperature=0, max_tokens=16)

    # 1) Run a generation. After this, the KV cache pool contains live
    #    blocks that we'd like to preserve across a hybrid hand-off.
    reference = llm.generate(prompt, sampling)
    print("reference output:", reference[0].outputs[0].text)

    free_pre, total = torch.cuda.mem_get_info()
    used_pre = total - free_pre
    print(f"before sleep: {_gib(used_pre)} GPU memory in use")

    # 2) Tag-wise sleep: offload weights only. The KV cache stays
    #    mapped on the GPU at the same virtual addresses. In a real
    #    partial-rollout scenario you would also pass `mode="keep"`
    #    to preserve any in-flight requests across the sleep window;
    #    we use the default `mode="abort"` here because there is no
    #    in-flight generation between the two `generate()` calls.
    llm.sleep(offload_tags=["weights"])
    free_post, _ = torch.cuda.mem_get_info()
    used_post = total - free_post
    print(
        f"after  sleep(offload_tags=['weights']): "
        f"{_gib(used_post)} GPU memory in use "
        f"(freed {_gib(used_pre - used_post)})"
    )

    # 3) Wake just the weights tag — KV cache was never asleep, so we do
    #    not pass it in. wake_up('kv_cache') would warn, since the
    #    executor's `sleeping_tags` only contains 'weights'.
    llm.wake_up(tags=["weights"])

    # 4) Generation works again, and (because KV cache survived) prefill
    #    for previously-seen prefixes is reusable.
    after = llm.generate(prompt, sampling)
    print("after-sleep output:", after[0].outputs[0].text)
    assert reference[0].outputs[0].text == after[0].outputs[0].text, (
        "Determinism broke after hybrid sleep/wake cycle."
    )

    print("ok: hybrid + partial rollout sleep/wake cycle preserved output")


if __name__ == "__main__":
    main()
