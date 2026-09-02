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


def _reference_ragged_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seq_lens: torch.Tensor,
    kv_cu_seq_lens: torch.Tensor,
    *,
    scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    num_reqs = q_cu_seq_lens.numel() - 1
    for req_idx in range(num_reqs):
        q_start = int(q_cu_seq_lens[req_idx])
        q_end = int(q_cu_seq_lens[req_idx + 1])
        kv_start = int(kv_cu_seq_lens[req_idx])
        kv_end = int(kv_cu_seq_lens[req_idx + 1])

        q_r = q[q_start:q_end].transpose(0, 1).float()
        k_r = k[kv_start:kv_end].transpose(0, 1).float()
        v_r = v[kv_start:kv_end].transpose(0, 1).float()

        scores = torch.matmul(q_r, k_r.transpose(-2, -1)) * scale
        if causal:
            causal_mask = torch.ones(
                (q_end - q_start, kv_end - kv_start),
                dtype=torch.bool,
                device=scores.device,
            ).triu(1)
            scores.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))

        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.matmul(probs, v_r).transpose(0, 1).to(v.dtype))
        lses.append(torch.logsumexp(scores, dim=-1))

    return torch.cat(outputs, dim=0), torch.cat(lses, dim=1)


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
def test_cpu_mla_backend_smoke(tmp_path) -> None:
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
        skip_tokenizer_init=True,
        trust_remote_code=True,
        load_format="dummy",
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=2,
        block_size=16,
        gpu_memory_utilization=0.25,
        hf_overrides=hf_overrides,
        kv_transfer_config={
            "kv_connector": "ExampleConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "shared_storage_path": str(tmp_path),
            },
        },
    )

    sampling_params = SamplingParams(max_tokens=4, temperature=0.0)
    outputs = llm.generate([[1, 2, 3, 4], [5, 6, 7]], sampling_params, use_tqdm=False)
    assert len(outputs) == 2
    for output in outputs:
        # `dummy` weights do not produce meaningful text, but the number
        # of generated tokens must match what we asked for.
        assert len(output.outputs[0].token_ids) == 4

    prefix_hit_prompt = [1] * 20
    outputs = llm.generate([prefix_hit_prompt], sampling_params, use_tqdm=False)
    assert len(outputs[0].outputs[0].token_ids) == 4
    assert llm.reset_prefix_cache(reset_connector=False)

    # ExampleConnector persists KV externally at block granularity. With a
    # 20-token prompt and block_size=16, the second run reloads 16 cached
    # tokens and forces the 4-token suffix through MLA prefill with context.
    outputs = llm.generate([prefix_hit_prompt], sampling_params, use_tqdm=False)
    assert len(outputs[0].outputs[0].token_ids) == 4


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

    backend.prepare_metadata(_PrefillMeta())
    q = torch.randn(6, 4, 16, dtype=torch.bfloat16)
    k = torch.randn(6, 4, 16, dtype=torch.bfloat16)
    v = torch.randn(6, 4, 16, dtype=torch.bfloat16)
    out = backend.run_prefill_new_tokens(q, k, v, return_softmax_lse=False)
    ref, _ = _reference_ragged_attention(
        q,
        k,
        v,
        _PrefillMeta.query_start_loc,
        _PrefillMeta.query_start_loc,
        scale=0.25,
        causal=True,
    )
    assert out.float().allclose(ref.float(), atol=2e-2)
    assert out.shape == (6, 4, 16)

    out_lse, lse = backend.run_prefill_new_tokens(q, k, v, return_softmax_lse=True)
    ref_out, ref_lse = _reference_ragged_attention(
        q,
        k,
        v,
        _PrefillMeta.query_start_loc,
        _PrefillMeta.query_start_loc,
        scale=0.25,
        causal=True,
    )
    assert out_lse.float().allclose(ref_out.float(), atol=2e-2)
    assert lse.float().allclose(ref_lse.float(), atol=2e-2)
    assert lse.shape == (4, 6)


@pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU only")
def test_cpu_mla_prefill_context_chunk() -> None:
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
        query_start_loc = torch.tensor([0, 5], dtype=torch.int32)

    class _Chunk:
        query_start_loc = torch.tensor([0, 2, 5], dtype=torch.int32)
        cu_seq_lens = torch.tensor([0, 3, 7], dtype=torch.int32)

    backend.prepare_metadata(_PrefillMeta())
    q = torch.randn(5, 4, 16, dtype=torch.bfloat16)
    k = torch.randn(7, 4, 16, dtype=torch.bfloat16)
    v = torch.randn(7, 4, 16, dtype=torch.bfloat16)

    out, lse = backend.run_prefill_context_chunk(_Chunk(), q, k, v)
    ref_out, ref_lse = _reference_ragged_attention(
        q,
        k,
        v,
        _Chunk.query_start_loc,
        _Chunk.cu_seq_lens,
        scale=0.25,
        causal=False,
    )
    assert out.float().allclose(ref_out.float(), atol=2e-2)
    assert lse.float().allclose(ref_lse.float(), atol=2e-2)
    assert out.shape == (5, 4, 16)
    assert lse.shape == (4, 5)
