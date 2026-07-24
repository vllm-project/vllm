# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke tests for the CPU MLA backend.

Two levels of coverage are provided:

* :func:`test_concat_and_cache_mla_cpu_fallback` checks that the pure
  PyTorch fallback wired in :mod:`vllm._custom_ops` writes the latent
  KV cache in the same layout the CPU decode kernel expects. This runs
  without downloading any model weights and is safe for CI.
* :func:`test_cpu_mla_backend_smoke` exercises the full end-to-end path
  (LLM engine, model construction, prefill + decode) against a shrunk
  DeepSeek-V2-Lite with `hf_overrides`. It uses `dummy` weights so the
  outputs are not meaningful, only shape / plumbing correctness is
  asserted. Marked `cpu_model` so it is skipped on non-CPU CI.
"""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU only")
def test_concat_and_cache_mla_cpu_fallback() -> None:
    kv_lora_rank = 512
    pe_dim = 64
    block_size = 16
    num_blocks = 4
    num_tokens = 8
    dtype = torch.float32

    kv_c = torch.randn(num_tokens, kv_lora_rank, dtype=dtype)
    k_pe = torch.randn(num_tokens, pe_dim, dtype=dtype)
    kv_cache = torch.zeros(num_blocks, block_size, kv_lora_rank + pe_dim, dtype=dtype)
    slot_mapping = torch.arange(num_tokens, dtype=torch.long)
    # Mark the last slot as padding to make sure we honour negative slots.
    slot_mapping[-1] = -1
    scale = torch.tensor(1.0, dtype=torch.float32)

    ops.concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping, "auto", scale)

    flat = kv_cache.view(-1, kv_lora_rank + pe_dim)
    for i in range(num_tokens - 1):
        assert torch.equal(flat[i, :kv_lora_rank], kv_c[i])
        assert torch.equal(flat[i, kv_lora_rank:], k_pe[i])
    # The padding slot must remain untouched.
    assert torch.all(flat[num_tokens - 1] == 0)


@pytest.mark.cpu_model
@pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU only")
def test_cpu_mla_backend_smoke() -> None:
    # Shrink the architecture aggressively so the test is CI-friendly.
    from vllm import LLM, SamplingParams

    hf_overrides = {
        "num_hidden_layers": 2,
        "n_routed_experts": 4,
        "first_k_dense_replace": 0,
        "num_experts_per_tok": 2,
    }
    llm = LLM(
        model="deepseek-ai/DeepSeek-V2-Lite",
        trust_remote_code=True,
        load_format="dummy",
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=2,
        block_size=16,
        hf_overrides=hf_overrides,
    )

    sampling_params = SamplingParams(max_tokens=4, temperature=0.0)
    outputs = llm.generate(["Hello", "MLA on CPU"], sampling_params)
    assert len(outputs) == 2
    for output in outputs:
        # `dummy` weights do not produce meaningful text, but the number
        # of generated tokens must match what we asked for.
        assert len(output.outputs[0].token_ids) == 4
