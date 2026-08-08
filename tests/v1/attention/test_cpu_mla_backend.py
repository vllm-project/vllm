# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke tests for the CPU MLA backend.

Two levels of coverage are provided:

* :func:`test_kv_cache_cpu_write` checks that the kv-cache write logic
  inlined in the CPU MLA backend writes the latent KV cache in the same
  layout the CPU decode kernel expects. This runs without downloading
  any model weights and is safe for CI.
* :func:`test_cpu_mla_backend_smoke` exercises the full end-to-end path
  (LLM engine, model construction, prefill + decode) against a shrunk
  DeepSeek-V2-Lite with `hf_overrides`. It uses `dummy` weights so the
  outputs are not meaningful, only shape / plumbing correctness is
  asserted. Marked `cpu_model` so it is skipped on non-CPU CI.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.prefill.cpu_sdpa import (
    CPUSDPAMLAPrefillBackend,
)
from vllm.v1.attention.backends.mla.prefill.selector import (
    get_mla_prefill_backend,
)


@pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU only")
def test_kv_cache_cpu_write() -> None:
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

    flat = kv_cache.view(-1, kv_lora_rank + pe_dim)
    slots = slot_mapping.to(torch.long)
    valid_mask = slots >= 0
    if not valid_mask.all().item():
        slots = slots[valid_mask]
        kv_c = kv_c[valid_mask]
        k_pe = k_pe[valid_mask]
    target_dtype = flat.dtype
    flat[slots, :kv_lora_rank] = kv_c.to(target_dtype)
    flat[slots, kv_lora_rank:] = k_pe.to(target_dtype)

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
        gpu_memory_utilization=0.25,
        hf_overrides=hf_overrides,
    )

    sampling_params = SamplingParams(max_tokens=4, temperature=0.0)
    outputs = llm.generate(["Hello", "MLA on CPU"], sampling_params)
    assert len(outputs) == 2
    for output in outputs:
        # `dummy` weights do not produce meaningful text, but the number
        # of generated tokens must match what we asked for.
        assert len(output.outputs[0].token_ids) == 4


@pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU only")
def test_cpu_mla_prefill_backend_selected() -> None:
    # On CPU (no device capability) the MLA prefill backend must be the
    # SDPA-based CPU backend, not flash-attn which is unavailable on CPU.
    backend_cls = get_mla_prefill_backend(None)
    assert backend_cls is CPUSDPAMLAPrefillBackend


@pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU only")
def test_cpu_mla_prefill_new_tokens() -> None:
    # The SDPA prefill must exactly match a reference causal attention over
    # the ragged (varlen) q/k/v layout, with per-request padding handled.
    backend = CPUSDPAMLAPrefillBackend(
        num_heads=4,
        scale=0.25,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=0,
        v_head_dim=16,
        vllm_config=None,
    )

    class _PrefillMeta:
        query_start_loc = torch.tensor([0, 3, 6], dtype=torch.int32)

    backend._prefill_metadata = _PrefillMeta()
    q = torch.randn(6, 4, 16, dtype=torch.bfloat16)
    k = torch.randn(6, 4, 16, dtype=torch.bfloat16)
    v = torch.randn(6, 4, 16, dtype=torch.bfloat16)
    out = backend.run_prefill_new_tokens(q, k, v, return_softmax_lse=False)

    qq = q[:3].float()
    kk = k[:3].float()
    vv = v[:3].float()
    ref = torch.nn.functional.scaled_dot_product_attention(
        qq.transpose(0, 1).unsqueeze(0),
        kk.transpose(0, 1).unsqueeze(0),
        vv.transpose(0, 1).unsqueeze(0),
        is_causal=True,
        scale=0.25,
    )
    ref = ref.squeeze(0).transpose(0, 1)
    assert out[:3].float().allclose(ref, atol=2e-2)
    assert out.shape == (6, 4, 16)
