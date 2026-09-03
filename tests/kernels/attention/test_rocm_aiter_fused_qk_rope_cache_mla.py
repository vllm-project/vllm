# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER fused_qk_rope_concat_and_cache_mla (VLLM_ROCM_USE_AITER_FUSED_QK_ROPE_CACHE_MLA)
vs the un-fused reference: DeepSeek YaRN rope -> concat_and_cache_mla -> cat + static fp8
quant. Inputs use the model's real layouts (transposed bmm output, strided split views)."""
import pytest
import torch

from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

NUM_TOKENS = [7, 64, 300]  # 300 selects AITER's general (non-per-head) variant
SEEDS = [0]
CUDA_DEVICES = ["cuda:0"]


@pytest.mark.skipif(
    not current_platform.is_rocm() or not rocm_aiter_ops.is_enabled(),
    reason="ROCm AITER only",
)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_fused_qk_rope_concat_and_cache_mla_matches_reference(
    num_tokens: int,
    seed: int,
    device: str,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)
    with set_current_vllm_config(VllmConfig()):
        _run(num_tokens, device)


def _run(B: int, dev: str) -> None:
    N, L, P, BS = 16, 512, 64, 64  # heads, kv_lora, pe, block_size (R1 @ TP8)
    NB = (B // BS + 2) * 2  # enough blocks for B distinct slots
    fp8 = current_platform.fp8_dtype()
    rope_scaling = {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 40,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "rope_type": "deepseek_yarn",
    }
    rope = get_rope(
        P,
        max_position=4096 * 40,
        is_neox_style=False,
        rope_parameters={"rope_theta": 10000, **rope_scaling},
        dtype=torch.bfloat16,
    ).to(dev)
    # the model's real layouts: ql_nope is the absorbed bmm output (batch is
    # the middle stride), q_pe a strided split view of the q_b_proj output
    ql_nope_t = torch.randn(N, B, L, device=dev, dtype=torch.bfloat16) * 0.5
    ql_nope = ql_nope_t.transpose(0, 1)
    q_full = torch.randn(B, N, 128 + P, device=dev, dtype=torch.bfloat16)
    q_pe = q_full[..., 128:]
    kv_c = torch.randn(B, L, device=dev, dtype=torch.bfloat16)
    k_pe = torch.randn(B, 1, P, device=dev, dtype=torch.bfloat16)
    positions = torch.randint(0, 3000, (B,), device=dev, dtype=torch.int64)
    slot_mapping = torch.randperm(NB * BS, device=dev)[:B].to(torch.int64)
    slot_mapping[-3:] = -1  # cudagraph-padding tokens: must be skipped
    q_scale = torch.tensor([0.37], device=dev, dtype=torch.float32)
    k_scale = torch.tensor([0.61], device=dev, dtype=torch.float32)

    # ---- reference: the un-fused sequence this kernel replaces ----
    q_pe_r, k_pe_r = rope(positions, q_pe.clone(), k_pe.clone())
    # vLLM allocates fp8 caches as uint8 storage
    cache_store = torch.zeros(NB, BS, L + P, device=dev, dtype=torch.uint8)
    ops.concat_and_cache_mla(
        kv_c, k_pe_r.squeeze(1), cache_store, slot_mapping, "fp8", k_scale
    )
    cache_ref = cache_store.view(fp8)
    fmax = torch.finfo(fp8).max
    q_ref = (
        (torch.cat([ql_nope, q_pe_r], dim=-1).float() / q_scale)
        .clamp(-fmax, fmax)
        .to(fp8)
    )

    # ---- fused ----
    cache_f = torch.zeros_like(cache_store).view(fp8)
    q_out = torch.empty(B, N, L + P, device=dev, dtype=fp8)
    cos, sin = rope.cos_sin_cache.chunk(2, dim=-1)
    rocm_aiter_ops.fused_qk_rope_concat_and_cache_mla(
        ql_nope,
        q_pe,
        kv_c,
        k_pe.squeeze(1),
        cache_f.view(-1, 1, L + P),
        q_out,
        slot_mapping,
        k_scale,
        q_scale,
        positions,
        cos.contiguous(),
        sin.contiguous(),
        is_neox=rope.is_neox_style,
    )
    torch.cuda.synchronize()

    def cmp(name: str, a: torch.Tensor, b: torch.Tensor, exact: bool = False):
        a, b = a.float(), b.float()
        if exact:
            # no rope on these dims: pure quant, must agree bit-for-bit
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)
            return
        # kernel ropes in fp32 vs bf16 reference: allow 1 fp8 ulp (e4m3:
        # 2^-3 relative) on a small fraction of elements
        rel = (a - b).abs() / b.abs().clamp(min=1e-2)
        assert (rel > 0.13).float().mean() < 0.01, f"{name}: too many mismatches"
        assert (a - b).abs().mean() < 0.02, f"{name}: mean error too large"

    valid = slot_mapping >= 0
    cmp("q_out nope", q_out[valid][..., :L], q_ref[valid][..., :L], exact=True)
    cmp("q_out pe", q_out[valid][..., L:], q_ref[valid][..., L:])
    cache_mask = torch.zeros(NB * BS, dtype=torch.bool, device=dev)
    cache_mask[slot_mapping[valid]] = True
    cf = cache_f.view(NB * BS, L + P)[cache_mask]
    cr = cache_ref.view(NB * BS, L + P)[cache_mask]
    cmp("cache nope", cf[:, :L], cr[:, :L], exact=True)
    cmp("cache pe", cf[:, L:], cr[:, L:])
    # padded (-1) slots skipped, untouched slots byte-identical
    torch.testing.assert_close(
        cache_f.view(NB * BS, -1)[~cache_mask].view(torch.uint8).float(),
        cache_ref.view(NB * BS, -1)[~cache_mask].view(torch.uint8).float(),
        rtol=0,
        atol=0,
    )
    # the kernel must not rotate q_pe in place
    assert torch.equal(q_pe, q_full[..., 128:])
