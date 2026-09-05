# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU tests for the rope-free NoPE zero-padded 576/656B DS-MLA envelope on
SM90 (FLASHMLA_SPARSE serving head_size=512 via fp8_ds_mla).

See openspec change fp8-ds-mla-nope-sm90:
- Cache-write contract: rope bytes [640:768] of every 656B row are bf16 zero
  and the NoPE bytes carry per-128-block fp8 quantization.
- Kernel equivalence: padded-envelope fp8 matches a bf16 NoPE-512 reference
  within fp8 block-quant noise; a garbage-rope control diverges.
- Ragged mixed-batch rows (kpool>1 valid counts incl. 0 and 1, -1 tails)
  stay finite and all-invalid rows neutralize to (0, -inf).
"""

from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention.mla_attention import (
    _NOPE_ZERO_ROPE_PAD_DIM,
    needs_nope_zero_rope_pad,
    nope_zero_rope_pad,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseImpl,
    FlashMLASparseMetadata,
)

if not current_platform.is_cuda():
    pytest.skip(
        "fp8_ds_mla NoPE SM90 GPU tests need CUDA (packed-cache ops and the "
        "FlashMLA kernels); CPU-eligible selection tests live in "
        "test_flashmla_nope_sm90_backend_selection.py",
        allow_module_level=True,
    )


def test_nope_zero_rope_pad_helper_shapes_and_zeros():
    """The shim provides the exact padded-envelope contract: zero q_pe
    [T, H, 64] appended to q and a persistent bf16-zero k_pe [T, 1, 64]."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    q = torch.randn(7, 4, 256, dtype=torch.bfloat16, device=device)
    kv_c = torch.randn(7, 512, dtype=torch.bfloat16, device=device)
    k_pe_empty = torch.empty(7, 1, 0, dtype=torch.bfloat16, device=device)

    q_padded, _, k_pe = nope_zero_rope_pad(q, kv_c, k_pe_empty)
    assert q_padded.shape == (7, 4, 256 + _NOPE_ZERO_ROPE_PAD_DIM)
    assert torch.equal(q_padded[..., :256], q)
    assert q_padded[..., 256:].abs().max().item() == 0.0
    assert k_pe.shape == (7, 1, _NOPE_ZERO_ROPE_PAD_DIM)
    assert k_pe.dtype == torch.bfloat16
    assert k_pe.abs().max().item() == 0.0

    # The k_pe buffer is persistent: a second call reuses the same storage
    # (CUDA-graph capture stability) and keeps it zero.
    k_pe2 = nope_zero_rope_pad(q, kv_c, k_pe_empty)[2]
    assert k_pe2.data_ptr() == k_pe.data_ptr()
    assert k_pe2.abs().max().item() == 0.0


def test_needs_nope_zero_rope_pad_gate():
    assert needs_nope_zero_rope_pad(0, "fp8_ds_mla")
    assert needs_nope_zero_rope_pad(0, "nvfp4_ds_mla")
    # bf16/auto NoPE and rope>0 models never activate the shim.
    assert not needs_nope_zero_rope_pad(0, "auto")
    assert not needs_nope_zero_rope_pad(0, "bfloat16")
    assert not needs_nope_zero_rope_pad(0, "fp8")
    assert not needs_nope_zero_rope_pad(0, None)
    assert not needs_nope_zero_rope_pad(64, "fp8_ds_mla")


def _block_quant_fp8(x: torch.Tensor, group_size: int = 128):
    """Per-128-element block fp8 quantization reference (fp8_ds_mla layout)."""
    rows, cols = x.shape
    assert cols % group_size == 0
    groups = cols // group_size
    xg = x.float().view(rows, groups, group_size)
    amax = xg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    finfo = torch.finfo(torch.float8_e4m3fn)
    scale = (finfo.max / amax).float()
    q = (xg * scale).clamp(min=finfo.min, max=finfo.max).to(torch.float8_e4m3fn)
    return q.view(rows, cols), scale.view(rows, groups)


def test_nope_fp8_ds_mla_cache_write_zero_rope_bytes():
    """The shim writes bf16 zeros into rope bytes [640:768] of every 656B row
    and the packed NoPE bytes carry per-128-block fp8 quantization."""
    from vllm import _custom_ops as ops

    device = torch.device("cuda")
    torch.manual_seed(0)
    num_tokens, block_size = 12, 4
    kv_lora_rank = 512
    kv_c = torch.randn(num_tokens, kv_lora_rank, dtype=torch.bfloat16, device=device)

    # Shim: zero q_pe/k_pe ahead of the concat path.
    _, _, k_pe = nope_zero_rope_pad(
        torch.empty(num_tokens, 1, 0, dtype=torch.bfloat16, device=device),
        kv_c,
        torch.empty(num_tokens, 1, 0, dtype=torch.bfloat16, device=device),
    )
    assert k_pe.shape == (num_tokens, 1, _NOPE_ZERO_ROPE_PAD_DIM)
    assert k_pe.dtype == torch.bfloat16
    assert k_pe.abs().max().item() == 0.0

    kv_cache = torch.zeros(
        (num_tokens + block_size - 1) // block_size,
        block_size,
        656,
        dtype=torch.uint8,
        device=device,
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    scale = torch.ones(1, dtype=torch.float32, device=device)

    # concat_and_cache_mla's fp8_ds_mla asserts (kv_lora_rank == 512,
    # pe_dim == 64, 656B row, 2-byte elements) must pass unchanged.
    ops.concat_and_cache_mla(
        kv_c,
        k_pe.squeeze(1),
        kv_cache,
        slot_mapping,
        kv_cache_dtype="fp8_ds_mla",
        scale=scale,
    )

    rows = kv_cache.view(-1, 656)[:num_tokens]
    # RoPE bytes [640:768] are 64 bf16 zeros.
    rope = rows[:, 640:768].view(torch.bfloat16)
    assert rope.abs().max().item() == 0.0
    # NoPE bytes [0:512] decode to the block-quantized reference.
    q_ref, scale_ref = _block_quant_fp8(kv_c.cpu())
    packed = rows[:, :512].view(torch.float8_e4m3fn).cpu()
    torch.testing.assert_close(packed.float(), q_ref.float(), rtol=0, atol=0)
    # Scale bytes [512:528]: 4 fp32 scales per token.
    scales = rows[:, 512:528].view(torch.float32).cpu()
    torch.testing.assert_close(scales, scale_ref, rtol=1e-6, atol=1e-8)


def test_nope_padded_envelope_matches_bf16_reference():
    """Zero-padded-rope fp8 sparse attention matches a bf16 NoPE-512
    reference within fp8 block-quant noise (rel-err ~2.5e-2 observed on the
    H20 probe); a garbage-rope control must diverge, proving the test has
    teeth."""
    import vllm.v1.attention.ops.flashmla as fm

    ok, reason = fm.is_flashmla_sparse_supported()
    if not ok:
        pytest.skip(reason)

    device = torch.device("cuda")
    torch.manual_seed(0)
    num_tokens = 8
    num_heads = 64
    kv_lora_rank = 512
    topk = 128
    page_block_size = 64

    kv_c = torch.randn(num_tokens, kv_lora_rank, dtype=torch.bfloat16, device=device)
    q_latent = torch.randn(
        num_tokens, num_heads, kv_lora_rank, dtype=torch.bfloat16, device=device
    )
    softmax_scale = (kv_lora_rank + _NOPE_ZERO_ROPE_PAD_DIM) ** -0.5

    # Build the padded 656B envelope cache via the shim's zero k_pe.
    k_pe = nope_zero_rope_pad(
        torch.empty(num_tokens, 1, 0, dtype=torch.bfloat16, device=device),
        kv_c,
        torch.empty(num_tokens, 1, 0, dtype=torch.bfloat16, device=device),
    )[2]
    kv_cache = torch.zeros(1, page_block_size, 1, 656, dtype=torch.uint8, device=device)
    assert num_tokens <= page_block_size
    from vllm import _custom_ops as ops

    ops.concat_and_cache_mla(
        kv_c,
        k_pe.squeeze(1),
        kv_cache,
        torch.arange(num_tokens, dtype=torch.int64, device=device),
        kv_cache_dtype="fp8_ds_mla",
        scale=torch.ones(1, dtype=torch.float32, device=device),
    )

    # Padded query: q_latent + zero q_pe -> [T, H, 576].
    q_padded = torch.cat(
        [
            q_latent,
            torch.zeros(
                num_tokens,
                num_heads,
                _NOPE_ZERO_ROPE_PAD_DIM,
                dtype=torch.bfloat16,
                device=device,
            ),
        ],
        dim=-1,
    )

    cache_seqlens = torch.full(
        (num_tokens,), num_tokens, dtype=torch.int32, device=device
    )
    tile_md, num_splits = fm.get_mla_metadata(
        cache_seqlens,
        num_heads,
        1,
        num_heads_q=num_heads,
        topk=topk,
        is_fp8_kvcache=True,
    )
    indices = torch.arange(topk, dtype=torch.int32, device=device)
    indices = indices.clamp(max=num_tokens - 1).expand(num_tokens, topk).contiguous()

    def run(cache):
        out, _ = fm.flash_mla_with_kvcache(
            q=q_padded.unsqueeze(0),
            k_cache=cache.squeeze(2).unsqueeze(-2),
            block_table=torch.zeros((num_tokens, 1), dtype=torch.int32, device=device),
            head_dim_v=512,
            cache_seqlens=cache_seqlens,
            tile_scheduler_metadata=tile_md,
            num_splits=num_splits,
            is_fp8_kvcache=True,
            indices=indices.unsqueeze(0),
            softmax_scale=softmax_scale,
        )
        return out.squeeze(0)  # (T, H, 512)

    out = run(kv_cache)

    # bf16 NoPE-512 reference: softmax(q . k) @ k over the latent dim only --
    # the zero rope part contributes exactly nothing.
    ref_scores = (
        torch.einsum("xhd,wd->xhw", q_latent.float(), kv_c.float()) * softmax_scale
    )
    ref_probs = torch.softmax(ref_scores, dim=1)
    ref_out = torch.einsum("xhw,wd->xhd", ref_probs, kv_c.float())

    rel_err = ((out.float() - ref_out).norm() / ref_out.norm().clamp(min=1e-6)).item()
    # fp8 block-quant noise floor (~2.5e-2 observed on the probe).
    assert rel_err < 0.08, f"padded-envelope fp8 rel-err {rel_err:.3e}"

    # Garbage-rope control: nonzero rope bytes must diverge from the NoPE
    # reference -- the padding contract is what keeps the fp8 path
    # NoPE-correct, not luck.
    garbage_pe = torch.randn(
        num_tokens, _NOPE_ZERO_ROPE_PAD_DIM, dtype=torch.bfloat16, device=device
    )
    kv_cache_garbage = kv_cache.clone()
    ops.concat_and_cache_mla(
        kv_c,
        garbage_pe,
        kv_cache_garbage,
        torch.arange(num_tokens, dtype=torch.int64, device=device),
        kv_cache_dtype="fp8_ds_mla",
        scale=torch.ones(1, dtype=torch.float32, device=device),
    )
    rel_err_garbage = (
        (run(kv_cache_garbage).float() - ref_out).norm()
        / ref_out.norm().clamp(min=1e-6)
    ).item()
    assert rel_err_garbage > 2 * rel_err + 1e-3, (
        f"garbage-rope control did not diverge ({rel_err_garbage:.3e} vs {rel_err:.3e})"
    )


def test_fp8_mixed_batch_ragged_valid_counts():
    """Ragged top-k rows (valid counts 0, 1, partial with -1 tails, as the
    kpool>1 indexer produces) must give finite outputs and neutralize
    all-invalid rows to (0, -inf) per the DCP merge contract."""
    import vllm.v1.attention.backends.mla.flashmla_sparse as fms

    num_tokens, num_heads, head_dim = 5, 2, 3
    device = torch.device("cuda")
    q = torch.randn(
        num_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    # Ragged rows: full, 0-valid, 1-valid, partial with -1 tail, all -1.
    local_indices = torch.tensor(
        [
            [0, 1, 2, 3],
            [-1, -1, -1, -1],
            [5, -1, -1, -1],
            [7, 8, -1, -1],
            [-1, -1, -1, -1],
        ],
        dtype=torch.int32,
        device=device,
    )

    def convert_indices(*args, **kwargs):  # noqa: ARG001
        return kwargs["output"].copy_(local_indices)

    orig = fms.triton_filter_and_convert_dcp_index
    fms.triton_filter_and_convert_dcp_index = convert_indices
    try:

        def run_kernel(**kwargs):
            out = torch.full((1, num_tokens, num_heads, 1), float("nan"), device=device)
            lse = torch.full((1, num_heads, num_tokens), float("nan"), device=device)
            for token_id in (0, 2, 3):  # rows with local candidates
                out[0, token_id] = float(token_id + 1)
                lse[0, :, token_id] = float(token_id + 1)
            return out, lse

        metadata = SimpleNamespace(
            fp8_extra_metadata=FlashMLASparseMetadata.FP8KernelMetadata(
                scheduler_metadata=object(),  # type: ignore[arg-type]
                dummy_block_table=torch.empty(1, 1, dtype=torch.int32, device=device),
                cache_lens=torch.empty(1, dtype=torch.int32, device=device),
            ),
            req_id_per_token=torch.empty(num_tokens, dtype=torch.int32, device=device),
            block_table=torch.empty(1, 1, dtype=torch.int32, device=device),
            block_size=64,
            cp_kv_cache_interleave_size=1,
            fp8_use_mixed_batch=True,
            physical_topk_indices=torch.empty_like(local_indices),
            physical_topk_valid_counts=torch.empty(
                num_tokens, dtype=torch.int32, device=device
            ),
            physical_topk_is_valid=False,
        )
        impl = SimpleNamespace(
            kv_cache_dtype="fp8_ds_mla",
            topk_indices_buffer=local_indices,
            dcp_world_size=2,
            dcp_rank=0,
            need_to_return_lse_for_decode=True,
            _fp8_flash_mla_kernel=run_kernel,
        )
        impl._forward_fp8_kv_mixed_batch = MethodType(
            FlashMLASparseImpl._forward_fp8_kv_mixed_batch, impl
        )

        out, lse = FlashMLASparseImpl.forward_mqa(
            impl, q, torch.empty(0, device=device), metadata, None
        )
    finally:
        fms.triton_filter_and_convert_dcp_index = orig

    assert out is not None and lse is not None
    assert not out.isnan().any()
    assert not lse.isnan().any()
    # All-invalid rows neutralize to the DCP merge identity (0, -inf).
    for token_id in (1, 4):
        assert torch.equal(out[token_id], torch.zeros_like(out[token_id]))
        assert torch.isneginf(lse[token_id]).all()
    # Rows with local candidates keep their kernel values.
    for token_id in (0, 2, 3):
        assert torch.equal(out[token_id], torch.full_like(out[token_id], token_id + 1))
        assert torch.equal(lse[token_id], torch.full_like(lse[token_id], token_id + 1))
