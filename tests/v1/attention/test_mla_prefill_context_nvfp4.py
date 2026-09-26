# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chunked-context MLA prefill with an nvfp4_ds_mla KV cache.

``MLACommonBaseImpl._compute_prefill_context`` gathers each chunk's context KV
into a BF16 workspace. For nvfp4_ds_mla it must use the NVFP4 upconvert gather
(the generic one rejects the dtype), and because that gather has no
``seq_starts``, a continuation chunk must still read from its start rather than
from the request's first token.

The reference runs the same method on a BF16 cache that holds the dequantized
values, through the generic gather, which applies ``seq_starts`` itself. Every
chunk must then see bit-identical Q, K and V.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBaseImpl,
    MLACommonPrefillMetadata,
    build_mla_chunked_context_metadata,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="nvfp4_ds_mla requires SM100 (Blackwell)",
)

_KV_LORA_RANK = 512
_QK_NOPE = 128
_QK_ROPE = 64
_V_HEAD_DIM = 128
_ENTRY = _KV_LORA_RANK + _QK_ROPE
_NVFP4_ENTRY = 352  # 256 e2m1 NoPE + 64 e4m3 RoPE + 32 e4m3 scale factors
_NUM_HEADS = 2
_BLOCK_SIZE = 64
_WORKSPACE_TOKENS = 512
# The first request spans several chunks, so later chunks continue it from a
# non-zero start; the short ones pack together and the last has no context.
_CONTEXT_LENS = [1000, 100, 64, 0]
_QUERY_LENS = [8, 4, 6, 5]


class _RecordingPrefillBackend:
    """Records the Q/K/V each context chunk attends over."""

    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def run_prefill_context_chunk(self, *, chunk, q, k, v):
        self.calls.append((q.float().clone(), k.float().clone(), v.float().clone()))
        num_q = q.shape[0]
        out = torch.full(
            (num_q, _NUM_HEADS, _V_HEAD_DIM),
            (k.float().mean() + v.float().mean()).item(),
            device=q.device,
            dtype=torch.bfloat16,
        )
        lse = torch.full(
            (_NUM_HEADS, num_q), 1.0 + chunk.index, device=q.device, dtype=torch.float32
        )
        return out, lse


class _KVBProj(torch.nn.Module):
    """Stand-in for ``kv_b_proj`` with a BF16 weight (returns (out, bias))."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        weight = torch.randn(
            _NUM_HEADS * (_QK_NOPE + _V_HEAD_DIM),
            _KV_LORA_RANK,
            device=device,
            dtype=torch.bfloat16,
        )
        self.register_buffer("weight", weight * 0.05)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return torch.nn.functional.linear(x, self.weight), None


class _Impl:
    """Only the attributes the context loop reads."""

    _compute_prefill_context = MLACommonBaseImpl._compute_prefill_context
    _concat_k_nope_k_pe = MLACommonBaseImpl._concat_k_nope_k_pe
    _use_flashinfer_concat_mla_k = False

    def __init__(self, kv_b_proj: _KVBProj, kv_cache_dtype: str) -> None:
        self.kv_b_proj = kv_b_proj
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_lora_rank = _KV_LORA_RANK
        self.num_heads = _NUM_HEADS
        self.qk_nope_head_dim = _QK_NOPE
        self.qk_rope_head_dim = _QK_ROPE
        self.v_head_dim = _V_HEAD_DIM


def _build_prefill_metadata(
    device: torch.device,
    q_data_type: torch.dtype,
    block_table: torch.Tensor,
    backend: _RecordingPrefillBackend,
) -> MLACommonPrefillMetadata:
    query_start_loc_cpu = torch.zeros(len(_QUERY_LENS) + 1, dtype=torch.int32)
    query_start_loc_cpu[1:] = torch.tensor(_QUERY_LENS, dtype=torch.int32).cumsum(0)
    # Packed DS-MLA caches are always upconverted into a BF16 workspace.
    workspace = torch.empty(
        (_WORKSPACE_TOKENS, _ENTRY), dtype=torch.bfloat16, device=device
    )
    chunked_context = build_mla_chunked_context_metadata(
        context_lens_cpu=torch.tensor(_CONTEXT_LENS, dtype=torch.int32),
        prefill_query_start_loc_cpu=query_start_loc_cpu,
        chunked_prefill_workspace=workspace,
        chunked_prefill_workspace_size=_WORKSPACE_TOKENS,
        block_size=_BLOCK_SIZE,
        align_chunk_to_block=True,
        device=device,
        dcp_world_size=1,
        dcp_local_block_size=1,
        dcp_virtual_block_size=1,
    )
    assert chunked_context is not None
    assert any(c.is_continuation for c in chunked_context.chunks), (
        "the batch must exercise a continuation chunk"
    )
    return MLACommonPrefillMetadata(
        block_table=block_table,
        query_start_loc=query_start_loc_cpu.to(device),
        max_query_len=max(_QUERY_LENS),
        chunked_context=chunked_context,
        q_data_type=q_data_type,
        output_dtype=torch.bfloat16,
        prefill_backend=backend,
    )


@pytest.mark.parametrize("fp8_query", [False, True], ids=["bf16_query", "fp8_query"])
@torch.inference_mode()
def test_nvfp4_prefill_context_matches_dequantized_cache(fp8_query: bool) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    q_data_type = current_platform.fp8_dtype() if fp8_query else torch.bfloat16

    num_reqs = len(_CONTEXT_LENS)
    max_blocks = (max(_CONTEXT_LENS) + max(_QUERY_LENS)) // _BLOCK_SIZE + 1
    num_blocks = num_reqs * max_blocks
    block_table = (
        torch.randperm(num_blocks, device=device)
        .to(torch.int32)
        .view(num_reqs, max_blocks)
    )

    nvfp4_cache = torch.zeros(
        (num_blocks, _BLOCK_SIZE, _NVFP4_ENTRY), dtype=torch.uint8, device=device
    )
    scale = torch.ones(1, dtype=torch.float32, device=device)
    for req, context_len in enumerate(_CONTEXT_LENS):
        if context_len == 0:
            continue
        pos = torch.arange(context_len, device=device)
        slots = block_table[req, pos // _BLOCK_SIZE].long() * _BLOCK_SIZE
        ops.concat_and_cache_mla(
            torch.randn(
                context_len, _KV_LORA_RANK, device=device, dtype=torch.bfloat16
            ),
            torch.randn(context_len, _QK_ROPE, device=device, dtype=torch.bfloat16),
            nvfp4_cache,
            slots + pos % _BLOCK_SIZE,
            kv_cache_dtype="nvfp4_ds_mla",
            scale=scale,
        )

    # Dequantize every slot once to build the BF16 reference cache.
    bf16_cache = torch.empty(
        (num_blocks * _BLOCK_SIZE, _ENTRY), dtype=torch.bfloat16, device=device
    )
    ops.cp_gather_and_upconvert_nvfp4_kv_cache(
        src_cache=nvfp4_cache,
        dst=bf16_cache,
        block_table=torch.arange(num_blocks, dtype=torch.int32, device=device)[None],
        workspace_starts=torch.zeros(1, dtype=torch.int32, device=device),
        batch_size=1,
    )
    bf16_cache = bf16_cache.view(num_blocks, _BLOCK_SIZE, _ENTRY)

    kv_b_proj = _KVBProj(device)
    q = (
        torch.randn(
            (sum(_QUERY_LENS), _NUM_HEADS, _QK_NOPE + _QK_ROPE),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.2
    ).to(q_data_type)

    backend = _RecordingPrefillBackend()
    out, lse = _Impl(kv_b_proj, "nvfp4_ds_mla")._compute_prefill_context(
        q,
        nvfp4_cache,
        SimpleNamespace(
            prefill=_build_prefill_metadata(device, q_data_type, block_table, backend)
        ),
        scale,
    )
    ref_backend = _RecordingPrefillBackend()
    ref_out, ref_lse = _Impl(kv_b_proj, "auto")._compute_prefill_context(
        q,
        bf16_cache,
        SimpleNamespace(
            prefill=_build_prefill_metadata(
                device, q_data_type, block_table, ref_backend
            )
        ),
        scale,
    )

    assert len(backend.calls) == len(ref_backend.calls) > 1
    for chunk_idx, (call, ref_call) in enumerate(
        zip(backend.calls, ref_backend.calls, strict=True)
    ):
        for name, actual, expected in zip(("q", "k", "v"), call, ref_call, strict=True):
            torch.testing.assert_close(
                actual,
                expected,
                atol=0,
                rtol=0,
                msg=lambda m, n=name, i=chunk_idx: f"chunk {i} {n} differs: {m}",
            )
    torch.testing.assert_close(out, ref_out, atol=0, rtol=0)
    torch.testing.assert_close(lse, ref_lse, atol=0, rtol=0)
