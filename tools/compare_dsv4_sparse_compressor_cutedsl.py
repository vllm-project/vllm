# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare DeepSeek V4 sparse compressor Triton and CuTe DSL paths.

This script checks the two production CuTe DSL paths against the existing vLLM
Triton fused sparse-compressor API:

* C4: CuTe DSL fused compress + RMSNorm + RoPE + FP8 cache store
* C128: CuTe DSL split compress, then RMSNorm + RoPE + FP8 cache store

Run from the vLLM repo root, for example:

    ../.venv/bin/python tools/compare_dsv4_sparse_compressor_cutedsl.py

By default the synthetic KV cache uses DeepSeek V4's vLLM page sizing:
main MLA block size 256, so C4 stores 64 compressed tokens per KV block and
C128 stores 2 compressed tokens per KV block.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch

from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    _fused_kv_compress_norm_rope_insert_sparse_attn,
)
from vllm.models.deepseek_v4.common.ops.sparse_attn_compress_cutedsl import (
    _compress_kv_sparse_attn_cutedsl,
    _fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl,
    _norm_rope_insert_sparse_attn_cutedsl,
)
from vllm.triton_utils import triton

HEAD_SIZE = 512
ROPE_HEAD_DIM = 64
NOPE_HEAD_DIM = HEAD_SIZE - ROPE_HEAD_DIM
QUANT_BLOCK = 64
TOKEN_STRIDE = HEAD_SIZE + ROPE_HEAD_DIM
SCALE_DIM = NOPE_HEAD_DIM // QUANT_BLOCK + 1
FP8_MAX = 448.0
DEEPSEEK_V4_MAIN_BLOCK_SIZE = 256
DEFAULT_NUM_TOKENS = 65


@dataclass
class SparseCase:
    name: str
    compress_ratio: int
    overlap: bool
    state_width: int
    state_block_size: int
    positions: torch.Tensor
    token_to_req_indices: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    state_cache: torch.Tensor
    rms_norm_weight: torch.Tensor
    cos_sin_cache: torch.Tensor
    kv_slot_mapping: torch.Tensor
    kv_cache_block_size: int
    k_cache_shape: tuple[int, int, int]


def make_cos_sin_cache(max_pos: int, rope_dim: int, device: torch.device):
    inv_freq = 1.0 / (
        10000.0
        ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim)
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous()


def storage_block_size(main_block_size: int, compress_ratio: int) -> int:
    if main_block_size % compress_ratio != 0:
        raise ValueError(
            "DeepSeek V4 MLA block size must be divisible by compress_ratio, "
            f"got main_block_size={main_block_size}, "
            f"compress_ratio={compress_ratio}."
        )
    return main_block_size // compress_ratio


def make_case(
    *,
    compress_ratio: int,
    num_tokens: int,
    kv_cache_block_size: int,
    device: torch.device,
    seed: int,
) -> SparseCase:
    torch.manual_seed(seed)
    overlap = compress_ratio == 4
    state_width = HEAD_SIZE * (2 if overlap else 1)
    state_block_size = 4 if compress_ratio == 4 else 8
    window = (1 + int(overlap)) * compress_ratio

    first_position = window - 1
    positions = (
        first_position
        + torch.arange(num_tokens, dtype=torch.int64, device=device) * compress_ratio
    )
    max_pos = int(positions.max().item())
    num_state_blocks = math.ceil((max_pos + 1) / state_block_size)

    state_cache = torch.randn(
        num_state_blocks,
        state_block_size,
        2 * state_width,
        dtype=torch.float32,
        device=device,
    )
    block_table = torch.arange(
        num_state_blocks, dtype=torch.int32, device=device
    ).unsqueeze(0)
    token_to_req_indices = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    kv_slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    num_kv_blocks = math.ceil(num_tokens / kv_cache_block_size)
    rms_norm_weight = torch.randn(HEAD_SIZE, dtype=torch.bfloat16, device=device)
    cos_sin_cache = make_cos_sin_cache(max_pos + 1, ROPE_HEAD_DIM, device)

    return SparseCase(
        name=f"C{compress_ratio}",
        compress_ratio=compress_ratio,
        overlap=overlap,
        state_width=state_width,
        state_block_size=state_block_size,
        positions=positions,
        token_to_req_indices=token_to_req_indices,
        slot_mapping=slot_mapping,
        block_table=block_table,
        state_cache=state_cache,
        rms_norm_weight=rms_norm_weight,
        cos_sin_cache=cos_sin_cache,
        kv_slot_mapping=kv_slot_mapping,
        kv_cache_block_size=kv_cache_block_size,
        k_cache_shape=(num_kv_blocks, kv_cache_block_size, TOKEN_STRIDE + SCALE_DIM),
    )


def empty_k_cache(case: SparseCase):
    return torch.zeros(
        case.k_cache_shape, dtype=torch.uint8, device=case.positions.device
    )


def run_triton(case: SparseCase) -> torch.Tensor:
    k_cache = empty_k_cache(case)
    _fused_kv_compress_norm_rope_insert_sparse_attn[(case.positions.numel(),)](
        case.state_cache,
        case.state_cache.stride(0),
        case.state_cache.stride(1),
        case.token_to_req_indices,
        case.positions,
        case.slot_mapping,
        case.block_table,
        case.block_table.stride(0),
        case.state_block_size,
        case.rms_norm_weight,
        1.0e-6,
        case.cos_sin_cache,
        case.cos_sin_cache.stride(0),
        k_cache,
        case.kv_slot_mapping,
        case.kv_cache_block_size,
        HEAD_SIZE=HEAD_SIZE,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(HEAD_SIZE),
        STATE_WIDTH=case.state_width,
        COMPRESS_RATIO=case.compress_ratio,
        OVERLAP=case.overlap,
        ROPE_HEAD_DIM=ROPE_HEAD_DIM,
        FP8_MAX=FP8_MAX,
        QUANT_BLOCK=QUANT_BLOCK,
        TOKEN_STRIDE=TOKEN_STRIDE,
        SCALE_DIM=SCALE_DIM,
        KV_BLOCK_STRIDE=k_cache.stride(0),
        num_warps=4,
    )
    return k_cache


def run_c4_fused_cutedsl(case: SparseCase) -> torch.Tensor:
    k_cache = empty_k_cache(case)
    _fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl(
        case.state_cache,
        case.token_to_req_indices,
        case.positions,
        case.slot_mapping,
        case.block_table,
        case.state_block_size,
        case.rms_norm_weight,
        1.0e-6,
        case.cos_sin_cache,
        k_cache,
        case.kv_slot_mapping,
        case.kv_cache_block_size,
        k_cache.stride(0),
        head_size=HEAD_SIZE,
        state_width=case.state_width,
        rope_head_dim=ROPE_HEAD_DIM,
        fp8_max=FP8_MAX,
        quant_block=QUANT_BLOCK,
        token_stride=TOKEN_STRIDE,
        scale_dim=SCALE_DIM,
        compress_ratio=case.compress_ratio,
        overlap=case.overlap,
    )
    return k_cache


def run_c128_split_cutedsl(case: SparseCase) -> torch.Tensor:
    k_cache = empty_k_cache(case)
    compressed_kv = torch.empty(
        (case.positions.numel(), HEAD_SIZE),
        dtype=torch.float32,
        device=case.positions.device,
    )
    _compress_kv_sparse_attn_cutedsl(
        case.state_cache,
        case.token_to_req_indices,
        case.positions,
        case.slot_mapping,
        case.block_table,
        case.state_block_size,
        compressed_kv,
        head_size=HEAD_SIZE,
        state_width=case.state_width,
        compress_ratio=case.compress_ratio,
        overlap=case.overlap,
    )
    _norm_rope_insert_sparse_attn_cutedsl(
        compressed_kv,
        case.positions,
        case.slot_mapping,
        case.rms_norm_weight,
        1.0e-6,
        case.cos_sin_cache,
        k_cache,
        case.kv_slot_mapping,
        case.kv_cache_block_size,
        k_cache.stride(0),
        head_size=HEAD_SIZE,
        rope_head_dim=ROPE_HEAD_DIM,
        fp8_max=FP8_MAX,
        quant_block=QUANT_BLOCK,
        token_stride=TOKEN_STRIDE,
        scale_dim=SCALE_DIM,
        compress_ratio=case.compress_ratio,
    )
    return k_cache


def decode_cache(case: SparseCase, k_cache: torch.Tensor) -> torch.Tensor:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("This script requires torch.float8_e4m3fn support.")

    out = torch.empty(
        (case.positions.numel(), HEAD_SIZE),
        dtype=torch.float32,
        device=case.positions.device,
    )
    n_nope_blocks = NOPE_HEAD_DIM // QUANT_BLOCK
    for token_idx in range(case.positions.numel()):
        slot = int(case.kv_slot_mapping[token_idx].item())
        page = slot // case.kv_cache_block_size
        offset = slot % case.kv_cache_block_size
        block_bytes = k_cache[page].reshape(-1)
        data_base = offset * TOKEN_STRIDE
        scale_base = case.kv_cache_block_size * TOKEN_STRIDE + offset * SCALE_DIM

        nope_bytes = block_bytes[data_base : data_base + NOPE_HEAD_DIM].contiguous()
        nope_fp8 = nope_bytes.view(torch.float8_e4m3fn).to(torch.float32)
        scale_bytes = block_bytes[scale_base : scale_base + n_nope_blocks]
        scales = torch.pow(2.0, scale_bytes.to(torch.float32) - 127.0)
        nope = (nope_fp8.reshape(n_nope_blocks, QUANT_BLOCK) * scales[:, None]).reshape(
            NOPE_HEAD_DIM
        )

        rope_base = data_base + NOPE_HEAD_DIM
        rope_bytes = block_bytes[rope_base : rope_base + ROPE_HEAD_DIM * 2].contiguous()
        rope = rope_bytes.view(torch.bfloat16).to(torch.float32)
        out[token_idx] = torch.cat((nope, rope), dim=0)
    return out


def compare(
    case: SparseCase,
    label: str,
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    require_byte_equal: bool,
) -> None:
    torch.cuda.synchronize()
    base_decoded = decode_cache(case, baseline)
    cand_decoded = decode_cache(case, candidate)
    diff = (base_decoded - cand_decoded).abs()
    byte_equal = torch.equal(baseline, candidate)
    print(
        f"{case.name} {label}: byte_equal={byte_equal} "
        f"max_abs={diff.max().item():.6g} "
        f"nope_max={diff[:, :NOPE_HEAD_DIM].max().item():.6g} "
        f"rope_max={diff[:, NOPE_HEAD_DIM:].max().item():.6g}"
    )
    torch.testing.assert_close(cand_decoded, base_decoded, atol=atol, rtol=rtol)
    if require_byte_equal:
        assert byte_equal, f"{case.name} {label} bytes differ from Triton baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens",
        type=int,
        default=DEFAULT_NUM_TOKENS,
        help=(
            "Number of compressed KV output tokens to test. The default crosses "
            "multiple vLLM storage blocks for both C4 and C128."
        ),
    )
    parser.add_argument(
        "--vllm-main-block-size",
        type=int,
        default=DEEPSEEK_V4_MAIN_BLOCK_SIZE,
        help=(
            "DeepSeek V4 sparse MLA block_size before compression. vLLM's "
            "DeepseekV4FlashMLASparseBackend default is 256."
        ),
    )
    parser.add_argument(
        "--kv-cache-block-size",
        type=int,
        default=None,
        help=(
            "Override the compressed KV cache storage block size. If omitted, "
            "uses vllm_main_block_size // compress_ratio, i.e. C4=64 and "
            "C128=2 for the default vLLM block size 256."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atol", type=float, default=0.25)
    parser.add_argument("--rtol", type=float, default=0.15)
    parser.add_argument("--case", choices=["all", "c4", "c128"], default="all")
    parser.add_argument("--require-byte-equal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton and CuTe DSL kernels.")
    device = torch.device(args.device)

    cases: list[tuple[int, str]] = []
    if args.case in ("all", "c4"):
        cases.append((4, "c4_fused_cutedsl"))
    if args.case in ("all", "c128"):
        cases.append((128, "c128_split_cutedsl"))

    for compress_ratio, label in cases:
        kv_cache_block_size = (
            args.kv_cache_block_size
            if args.kv_cache_block_size is not None
            else storage_block_size(args.vllm_main_block_size, compress_ratio)
        )
        case = make_case(
            compress_ratio=compress_ratio,
            num_tokens=args.tokens,
            kv_cache_block_size=kv_cache_block_size,
            device=device,
            seed=args.seed + compress_ratio,
        )
        print(
            f"Running {case.name} Triton baseline "
            f"(kv_cache_block_size={case.kv_cache_block_size})"
        )
        triton_cache = run_triton(case)
        torch.cuda.synchronize()

        if compress_ratio == 4:
            print("Running C4 fused CuTe DSL")
            cutedsl_cache = run_c4_fused_cutedsl(case)
        else:
            print("Running C128 split CuTe DSL")
            cutedsl_cache = run_c128_split_cutedsl(case)
        compare(
            case,
            label,
            triton_cache,
            cutedsl_cache,
            atol=args.atol,
            rtol=args.rtol,
            require_byte_equal=args.require_byte_equal,
        )

    print("All comparisons passed.")


if __name__ == "__main__":
    main()
