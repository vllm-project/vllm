# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark DeepSeek-V4 ROCm gfx950 compressor kernels."""

from dataclasses import dataclass

import torch
from tabulate import tabulate

from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    compress_norm_rope_store_triton,
)
from vllm.models.deepseek_v4.common.ops.save_partial_states import (
    save_partial_states,
)
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser

KV_BLOCK_SIZE = 16
RMS_EPS = 1e-6
SEED = 2026


@dataclass(frozen=True)
class ShapeConfig:
    name: str
    head_dim: int
    rope_head_dim: int
    ratio: int
    overlap: bool
    state_block_size: int
    quant_format: str

    @property
    def state_width(self) -> int:
        return (2 if self.overlap else 1) * self.head_dim

    @property
    def coff(self) -> int:
        return 2 if self.overlap else 1

    @property
    def token_stride(self) -> int:
        if self.quant_format == "indexer_fp8":
            return self.head_dim
        if self.quant_format == "indexer_mxfp4":
            return self.head_dim // 2
        return self.head_dim + self.rope_head_dim

    @property
    def scale_dim(self) -> int:
        if self.quant_format == "indexer_fp8":
            return 4
        if self.quant_format == "indexer_mxfp4":
            return self.head_dim // 32
        return (self.head_dim - self.rope_head_dim) // 64 + 1

    @property
    def quant_block(self) -> int:
        if self.quant_format == "indexer_fp8":
            return self.head_dim
        if self.quant_format == "indexer_mxfp4":
            return 32
        return 64


SHAPES = (
    ShapeConfig("csa_main", 512, 64, 4, True, 4, "csa"),
    ShapeConfig("hca_main", 512, 64, 128, False, 8, "hca"),
    ShapeConfig("indexer_fp8", 128, 64, 4, True, 4, "indexer_fp8"),
    ShapeConfig("indexer_mxfp4", 128, 64, 4, True, 4, "indexer_mxfp4"),
)
SCENARIOS = (
    "decode_boundary",
    "prefill_256",
    "prefill_1024",
    "prefill_4096",
    "prefill_32768",
)


class KVCacheMetadata:
    def __init__(self, slot_mapping: torch.Tensor):
        self.slot_mapping = slot_mapping


@dataclass
class BenchmarkInput:
    name: str
    shape: ShapeConfig
    positions: torch.Tensor
    token_to_req_indices: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    kv_slot_mapping: torch.Tensor
    state_cache_fp32: torch.Tensor
    state_cache_bf16: torch.Tensor
    ape: torch.Tensor
    cos_sin_cache: torch.Tensor
    rms_weight: torch.Tensor
    num_tokens: int
    num_kv_blocks: int
    kv_block_bytes: int

    def new_kv_cache(self) -> torch.Tensor:
        return torch.zeros(
            self.num_kv_blocks,
            KV_BLOCK_SIZE,
            self.kv_block_bytes,
            dtype=torch.uint8,
            device="cuda",
        )

    def common_kwargs(self, kv_cache: torch.Tensor) -> dict:
        shape = self.shape
        return dict(
            num_actual=self.num_tokens,
            token_to_req_indices=self.token_to_req_indices,
            positions=self.positions,
            slot_mapping=self.slot_mapping,
            block_table=self.block_table,
            block_size=shape.state_block_size,
            state_width=shape.state_width,
            cos_sin_cache=self.cos_sin_cache,
            kv_cache=kv_cache,
            k_cache_metadata=KVCacheMetadata(self.kv_slot_mapping),
            pdl_kwargs={},
            head_dim=shape.head_dim,
            rope_head_dim=shape.rope_head_dim,
            compress_ratio=shape.ratio,
            overlap=shape.overlap,
            use_fp4_cache=shape.quant_format == "indexer_mxfp4",
            rms_norm_weight=self.rms_weight,
            rms_norm_eps=RMS_EPS,
            quant_block=shape.quant_block,
            token_stride=shape.token_stride,
            scale_dim=shape.scale_dim,
        )


def _scenario(
    shape: ShapeConfig,
    name: str,
) -> tuple[list[int], list[int], list[int]]:
    if name == "decode_boundary":
        positions, req_indices, slots = [], [], []
        context_len = 4096
        for req_idx, max_position in enumerate([127, 255, 383, 511]):
            for position in range(max_position + 1):
                positions.append(position)
                req_indices.append(req_idx)
                slots.append(req_idx * context_len + position)
        return positions, req_indices, slots

    seq_len = int(name.removeprefix("prefill_"))
    positions = list(range(seq_len))
    return positions, [0] * seq_len, positions


def _kv_slot_mapping(positions: list[int], ratio: int) -> list[int]:
    kv_slots = []
    next_slot = 0
    for position in positions:
        if (position + 1) % ratio == 0:
            kv_slots.append(next_slot)
            next_slot += 1
        else:
            kv_slots.append(-1)
    return kv_slots


def _block_table(
    positions: list[int],
    token_to_req: list[int],
    slot_mapping: list[int],
    block_size: int,
) -> torch.Tensor:
    num_reqs = max(token_to_req) + 1
    num_blocks = max(positions) // block_size + 2
    table = torch.zeros(num_reqs, num_blocks, dtype=torch.int32, device="cuda")
    for req_idx, position, slot in zip(token_to_req, positions, slot_mapping):
        table[req_idx, position // block_size] = slot // block_size
    return table


def _cos_sin_cache(max_position: int, rope_head_dim: int) -> torch.Tensor:
    inv_freq = 1.0 / (
        10000
        ** (
            torch.arange(0, rope_head_dim, 2, dtype=torch.float32, device="cuda")
            / rope_head_dim
        )
    )
    freqs = torch.outer(
        torch.arange(max_position + 1, dtype=torch.float32, device="cuda"),
        inv_freq,
    )
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


def build_input(shape: ShapeConfig, scenario: str) -> BenchmarkInput:
    positions, token_to_req, slot_mapping = _scenario(shape, scenario)
    num_tokens = len(positions)
    kv_slot_mapping = _kv_slot_mapping(positions, shape.ratio)

    generator = torch.Generator(device="cuda").manual_seed(SEED + num_tokens)
    kv = torch.randn(
        num_tokens,
        shape.coff * shape.head_dim,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    score = torch.randn_like(kv)
    ape = torch.randn(
        shape.ratio,
        shape.coff * shape.head_dim,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    rms_weight = torch.rand(
        shape.head_dim,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    ).to(torch.bfloat16)

    positions_t = torch.tensor(positions, dtype=torch.int64, device="cuda")
    token_to_req_t = torch.tensor(token_to_req, dtype=torch.int32, device="cuda")
    slot_mapping_t = torch.tensor(slot_mapping, dtype=torch.int64, device="cuda")
    kv_slot_mapping_t = torch.tensor(kv_slot_mapping, dtype=torch.int64, device="cuda")

    max_slot = max(slot_mapping)
    state_shape = (
        max_slot // shape.state_block_size + 4,
        shape.state_block_size,
        2 * shape.state_width,
    )
    state_cache_fp32 = torch.zeros(state_shape, dtype=torch.float32, device="cuda")
    state_cache_bf16 = torch.zeros(state_shape, dtype=torch.bfloat16, device="cuda")
    save_partial_states(
        kv=kv,
        score=score,
        ape=ape,
        positions=positions_t,
        state_cache=state_cache_fp32,
        slot_mapping=slot_mapping_t,
        block_size=shape.state_block_size,
        state_width=shape.state_width,
        compress_ratio=shape.ratio,
    )
    save_partial_states(
        kv=kv,
        score=score,
        ape=None,
        positions=positions_t,
        state_cache=state_cache_bf16,
        slot_mapping=slot_mapping_t,
        block_size=shape.state_block_size,
        state_width=shape.state_width,
        compress_ratio=shape.ratio,
    )

    num_compressed_tokens = sum(slot >= 0 for slot in kv_slot_mapping)
    return BenchmarkInput(
        name=scenario,
        shape=shape,
        positions=positions_t,
        token_to_req_indices=token_to_req_t,
        slot_mapping=slot_mapping_t,
        block_table=_block_table(
            positions, token_to_req, slot_mapping, shape.state_block_size
        ),
        kv_slot_mapping=kv_slot_mapping_t,
        state_cache_fp32=state_cache_fp32,
        state_cache_bf16=state_cache_bf16,
        ape=ape,
        cos_sin_cache=_cos_sin_cache(max(positions), shape.rope_head_dim),
        rms_weight=rms_weight,
        num_tokens=num_tokens,
        num_kv_blocks=num_compressed_tokens // KV_BLOCK_SIZE + 4,
        kv_block_bytes=shape.token_stride + shape.scale_dim,
    )


def hip_available() -> bool:
    try:
        import vllm._rocm_C  # noqa: F401

        return hasattr(torch.ops._rocm_C, "dsv4_csa_compress")
    except Exception:
        return False


def run_triton(inputs: BenchmarkInput, kv_cache: torch.Tensor) -> None:
    compress_norm_rope_store_triton(
        state_cache=inputs.state_cache_fp32,
        **inputs.common_kwargs(kv_cache),
    )


def run_hip(inputs: BenchmarkInput, kv_cache: torch.Tensor) -> None:
    shape = inputs.shape
    common = (
        inputs.state_cache_bf16,
        inputs.num_tokens,
        inputs.ape,
        inputs.token_to_req_indices,
        inputs.positions,
        inputs.slot_mapping,
        inputs.block_table,
        shape.state_block_size,
        inputs.rms_weight.to(torch.float32),
        RMS_EPS,
        inputs.cos_sin_cache,
        kv_cache,
        inputs.kv_slot_mapping,
        kv_cache.shape[1],
        shape.scale_dim,
    )
    if shape.head_dim == 128:
        torch.ops._rocm_C.dsv4_indexer_compress(
            *common, shape.quant_format == "indexer_mxfp4"
        )
    elif shape.ratio == 128:
        plan_capacity = (
            inputs.num_tokens // shape.ratio + inputs.block_table.shape[0] + 2
        )
        plan_scratch = torch.empty(plan_capacity, dtype=torch.int32, device="cuda")
        counter_scratch = torch.empty(1, dtype=torch.int32, device="cuda")
        torch.ops._rocm_C.dsv4_hca_compress(
            *common,
            plan_scratch,
            counter_scratch,
        )
    else:
        torch.ops._rocm_C.dsv4_csa_compress(*common)


def do_bench_us(fn) -> tuple[float, float, float]:
    ms, min_ms, max_ms = triton.testing.do_bench(
        fn,
        quantiles=[0.5, 0.2, 0.8],
    )
    return 1000 * ms, 1000 * min_ms, 1000 * max_ms


def benchmark_one(inputs: BenchmarkInput) -> tuple[str, str, str, str, str]:
    hip_cache = inputs.new_kv_cache()
    hip_us, _, _ = do_bench_us(lambda: run_hip(inputs, hip_cache))

    if inputs.shape.quant_format == "indexer_mxfp4":
        return inputs.shape.name, inputs.name, "n/a", f"{hip_us:.1f}", "n/a"

    triton_cache = inputs.new_kv_cache()
    triton_us, _, _ = do_bench_us(lambda: run_triton(inputs, triton_cache))
    return (
        inputs.shape.name,
        inputs.name,
        f"{triton_us:.1f}",
        f"{hip_us:.1f}",
        f"{hip_us / triton_us:.3f}",
    )


def parse_args():
    parser = FlexibleArgumentParser(
        description="Benchmark DeepSeek-V4 ROCm gfx950 compressor kernels."
    )
    return parser.parse_args()


def main() -> None:
    parse_args()
    torch.set_default_device("cuda")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP device is not available")
    if not hip_available():
        raise SystemExit("dsv4 compressor ops are not registered in _rocm_C")

    rows = []
    for shape in SHAPES:
        for scenario in SCENARIOS:
            row = benchmark_one(build_input(shape, scenario))
            rows.append(row)

    print(
        tabulate(
            rows,
            headers=["shape", "scenario", "triton (us)", "hip (us)", "hip/triton"],
            tablefmt="github",
        )
    )


if __name__ == "__main__":
    main()
