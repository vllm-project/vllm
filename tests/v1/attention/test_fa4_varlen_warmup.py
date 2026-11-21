# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.warmup.fa4_varlen_warmup import (
    FA4_VARLEN_ATTENTION_KERNEL,
    FA4VarlenCallShape,
    FA4VarlenWarmupSpec,
    get_candidate_shapes,
    get_missing_fa4_kernels,
)
from vllm.platforms import current_platform
from vllm.v1.attention.ops.vit_attn_wrappers import vit_flash_attn_wrapper

SPEC = FA4VarlenWarmupSpec(
    num_heads=16,
    num_kv_heads=16,
    head_dim=128,
    dtype=torch.bfloat16,
    max_seqlen=1024,
    max_num_tokens=4096,
    max_num_seqs=32,
)

requires_sm100 = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="The varlen FA4 warmup targets SM100-family GPUs",
)


def test_candidate_shapes_bracket_every_tile_within_bounds():
    shapes = get_candidate_shapes(SPEC)

    assert all(
        shape.max_seqlen <= SPEC.max_seqlen
        and shape.num_seqs <= SPEC.max_num_seqs
        and shape.num_tokens <= SPEC.max_num_tokens
        for shape in shapes
    )
    seqlens = {shape.max_seqlen for shape in shapes if shape.num_seqs == 1}
    for start in range(0, SPEC.max_seqlen, 128):
        assert {start + 1, start + 128} <= seqlens
    # Equal lengths and one long sequence among single-token ones.
    assert FA4VarlenCallShape(4, 1024, 4096) in shapes
    assert FA4VarlenCallShape(4, 1024, 1027) in shapes


def _dense_shapes(spec: FA4VarlenWarmupSpec) -> list[FA4VarlenCallShape]:
    shapes = []
    for max_seqlen in range(1, spec.max_seqlen + 1):
        for num_seqs in (1, 2, 3):
            num_tokens = max_seqlen + num_seqs - 1
            shapes.append(FA4VarlenCallShape(num_seqs, max_seqlen, num_tokens))
    for max_seqlen in (1, 128, 129, 256, 257, 512, 640, 1024):
        for num_seqs in range(1, spec.max_num_seqs + 1):
            low = max_seqlen + num_seqs - 1
            high = max_seqlen * num_seqs
            for num_tokens in (low, (low + high) // 2, high):
                shapes.append(FA4VarlenCallShape(num_seqs, max_seqlen, num_tokens))
    return [shape for shape in shapes if shape.num_tokens <= spec.max_num_tokens]


@requires_sm100
@torch.inference_mode()
def test_warmup_compiles_every_kernel_a_denser_sweep_reaches():
    """Guards the shape grid against FA4 selection changes across FA pins."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    for key in FA4_VARLEN_ATTENTION_KERNEL.get_warmup_keys(SPEC):
        FA4_VARLEN_ATTENTION_KERNEL.compile(key)

    device = current_platform.current_device()
    with FakeTensorMode():
        missed = {
            shape: misses
            for shape in _dense_shapes(SPEC)
            if (misses := get_missing_fa4_kernels(SPEC, shape, device))
        }
    assert not missed, f"{len(missed)} shapes still compile, e.g. {next(iter(missed))}"


@requires_sm100
@pytest.mark.parametrize("num_seqs,seqlen", [(1, 257), (4, 1024)])
@torch.inference_mode()
def test_vit_attention_reuses_warmed_kernels(monkeypatch, num_seqs, seqlen):
    """The warmup's fake call must key FA4 like the ViT wrapper's real one."""
    from vllm.vllm_flash_attn.cute import interface as fa4

    for key in FA4_VARLEN_ATTENTION_KERNEL.get_warmup_keys(SPEC):
        FA4_VARLEN_ATTENTION_KERNEL.compile(key)

    def unexpected_compile(*args, **kwargs):
        pytest.fail("ViT attention compiled a kernel the warmup missed")

    monkeypatch.setattr(fa4.cute, "compile", unexpected_compile)
    total = num_seqs * seqlen
    qkv = torch.randn(total, 3, 16, 128, device="cuda", dtype=SPEC.dtype)
    q, k, v = qkv.unbind(1)
    # RoPE materializes Q/K; V stays a view into the packed QKV, as in MoonViT.
    q, k = q.contiguous(), k.contiguous()
    vit_flash_attn_wrapper(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        batch_size=1,
        is_rocm_aiter=False,
        fa_version=4,
        scale=SPEC.head_dim**-0.5,
        cu_seqlens=torch.arange(0, total + 1, seqlen, device="cuda", dtype=torch.int32),
        max_seqlen=torch.tensor(seqlen, device="cuda"),
    )
