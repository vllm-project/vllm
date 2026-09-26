# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Losslessly packed BF16 language-model head for single-token decode."""

from dataclasses import dataclass
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_PACK_BLOCK = 256
_MIN_DENSE_BYTES = 64 * 1024 * 1024
_PACK_WORKSPACE_HEADROOM = 256 * 1024 * 1024
_MAX_K = 8192
_MAX_NUMEL = 2**31 - 1
_STATE_PREFIX = "_vllm_lossless_lm_head_"
_STATE_NAMES = (
    "sign_mantissa",
    "exponent_nibbles",
    "base_exponent",
    "fallback_slot",
    "fallback_bits",
)


@dataclass(frozen=True, slots=True)
class PackedLayoutPlan:
    """Storage properties for one packed weight matrix."""

    n: int
    k: int
    numel: int
    padded_numel: int
    block_count: int
    fallback_blocks: int
    dense_bytes: int
    packed_bytes: int

    @property
    def packed_fraction(self) -> float:
        return self.packed_bytes / self.dense_bytes


@dataclass(frozen=True, slots=True)
class PackedLaunchConfig:
    """Triton launch parameters for one hidden dimension."""

    block_n: int
    num_warps: int


def choose_launch_config(k: int) -> PackedLaunchConfig:
    """Bound the per-program register footprint for supported hidden sizes."""
    if k <= 1024:
        return PackedLaunchConfig(block_n=16, num_warps=8)
    if k <= 2048:
        return PackedLaunchConfig(block_n=8, num_warps=8)
    if k <= 4096:
        return PackedLaunchConfig(block_n=4, num_warps=8)
    return PackedLaunchConfig(block_n=2, num_warps=8)


def packed_storage_bytes(
    *, padded_numel: int, block_count: int, fallback_blocks: int
) -> int:
    """Return materialized bytes for the exact base-plus-delta layout."""
    return (
        padded_numel
        + padded_numel // 2
        + block_count
        + block_count * 4
        + fallback_blocks * _PACK_BLOCK * 2
    )


def is_statically_eligible(weight: torch.Tensor) -> tuple[bool, str | None]:
    """Check invariants that do not depend on the weight values."""
    if not current_platform.is_cuda():
        return False, "requires CUDA"
    if not weight.is_cuda:
        return False, "weight is not resident on CUDA"
    if weight.dtype != torch.bfloat16:
        return False, "weight is not BF16"
    if weight.ndim != 2:
        return False, "weight is not a matrix"
    if not weight.is_contiguous():
        return False, "weight is not contiguous row-major"

    n, k = (int(value) for value in weight.shape)
    if n * k > _MAX_NUMEL:
        return False, "head exceeds signed 32-bit indexing"
    if n * k * weight.element_size() < _MIN_DENSE_BYTES:
        return False, "head is too small to amortize packing"
    if k <= 0 or k > _MAX_K:
        return False, f"K={k} is outside the supported range 1..{_MAX_K}"
    return True, None


def _block_bits(weight: torch.Tensor, start: int, end: int) -> torch.Tensor:
    flat = weight.view(torch.int16).reshape(-1)
    valid_end = min(end, flat.numel())
    chunk = flat[start:valid_end]
    if valid_end != end:
        padded = torch.zeros(end - start, dtype=torch.int16, device=weight.device)
        padded[: chunk.numel()] = chunk
        chunk = padded
    return chunk.reshape(-1, _PACK_BLOCK)


@torch.no_grad()
def pack_bf16_weight(
    weight: torch.Tensor,
    *,
    max_packed_fraction: float,
    chunk_blocks: int = 16384,
) -> tuple[PackedLayoutPlan, tuple[torch.Tensor, ...]] | None:
    """Create an exact BF16 encoding with bounded temporary GPU memory."""
    if not 0.0 < max_packed_fraction <= 1.0:
        raise ValueError("max_packed_fraction must be in (0, 1]")
    if chunk_blocks <= 0:
        raise ValueError("chunk_blocks must be positive")
    eligible, reason = is_statically_eligible(weight)
    if not eligible:
        logger.info_once("Lossless packed lm_head skipped: %s", reason)
        return None

    n, k = (int(value) for value in weight.shape)
    numel = weight.numel()
    block_count = cdiv(numel, _PACK_BLOCK)
    padded_numel = block_count * _PACK_BLOCK
    base_exponent = torch.empty(block_count, dtype=torch.uint8, device=weight.device)
    packable = torch.empty(block_count, dtype=torch.bool, device=weight.device)

    for block_start in range(0, block_count, chunk_blocks):
        block_end = min(block_start + chunk_blocks, block_count)
        bits = _block_bits(weight, block_start * _PACK_BLOCK, block_end * _PACK_BLOCK)
        exponents = (bits.to(torch.int32) >> 7) & 0xFF
        minimum = exponents.amin(dim=1)
        maximum = exponents.amax(dim=1)
        base_exponent[block_start:block_end] = minimum.to(torch.uint8)
        packable[block_start:block_end] = maximum - minimum <= 15

    fallback_blocks = int((~packable).sum().item())
    dense_bytes = numel * weight.element_size()
    packed_bytes = packed_storage_bytes(
        padded_numel=padded_numel,
        block_count=block_count,
        fallback_blocks=fallback_blocks,
    )
    plan = PackedLayoutPlan(
        n=n,
        k=k,
        numel=numel,
        padded_numel=padded_numel,
        block_count=block_count,
        fallback_blocks=fallback_blocks,
        dense_bytes=dense_bytes,
        packed_bytes=packed_bytes,
    )
    if plan.packed_fraction > max_packed_fraction:
        logger.info_once(
            "Lossless packed lm_head skipped: packed fraction %.3f exceeds %.3f",
            plan.packed_fraction,
            max_packed_fraction,
        )
        return None

    free_bytes, _ = current_platform.mem_get_info()
    required_free_bytes = plan.packed_bytes + _PACK_WORKSPACE_HEADROOM
    if required_free_bytes > free_bytes:
        logger.info_once(
            "Lossless packed lm_head skipped: needs %.3f MiB free, has %.3f MiB",
            required_free_bytes / (1024 * 1024),
            free_bytes / (1024 * 1024),
        )
        return None

    sign_mantissa = torch.empty(padded_numel, dtype=torch.uint8, device=weight.device)
    exponent_nibbles = torch.empty(
        padded_numel // 2, dtype=torch.uint8, device=weight.device
    )
    fallback_slot = torch.full(
        (block_count,), -1, dtype=torch.int32, device=weight.device
    )
    fallback_bits = torch.empty(
        (fallback_blocks, _PACK_BLOCK), dtype=torch.int16, device=weight.device
    )

    next_fallback = 0
    for block_start in range(0, block_count, chunk_blocks):
        block_end = min(block_start + chunk_blocks, block_count)
        bits = _block_bits(weight, block_start * _PACK_BLOCK, block_end * _PACK_BLOCK)
        flat_bits = bits.reshape(-1).to(torch.int32)
        flat_start = block_start * _PACK_BLOCK
        flat_end = block_end * _PACK_BLOCK
        sign_mantissa[flat_start:flat_end] = (
            (flat_bits & 0x7F) | ((flat_bits >> 8) & 0x80)
        ).to(torch.uint8)

        exponents = (flat_bits >> 7) & 0xFF
        repeated_base = base_exponent[block_start:block_end].repeat_interleave(
            _PACK_BLOCK
        )
        deltas = exponents - repeated_base.to(torch.int32)
        local_packable = packable[block_start:block_end].repeat_interleave(_PACK_BLOCK)
        deltas = torch.where(local_packable, deltas, 0).reshape(-1, 2)
        exponent_nibbles[flat_start // 2 : flat_end // 2] = (
            deltas[:, 0] | (deltas[:, 1] << 4)
        ).to(torch.uint8)

        failed_local = torch.nonzero(
            ~packable[block_start:block_end], as_tuple=False
        ).flatten()
        failed_count = failed_local.numel()
        if failed_count:
            failed_global = failed_local + block_start
            fallback_slot[failed_global] = torch.arange(
                next_fallback,
                next_fallback + failed_count,
                dtype=torch.int32,
                device=weight.device,
            )
            fallback_bits[next_fallback : next_fallback + failed_count] = bits[
                failed_local
            ]
            next_fallback += failed_count

    assert next_fallback == fallback_blocks
    return plan, (
        sign_mantissa,
        exponent_nibbles,
        base_exponent,
        fallback_slot,
        fallback_bits,
    )


@triton.jit
def _lossless_packed_bf16_lm_head_kernel(
    x_ptr,
    sign_mantissa_ptr,
    exponent_nibbles_ptr,
    base_exponent_ptr,
    fallback_slot_ptr,
    fallback_bits_ptr,
    output_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PACK_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    valid = (offsets_n[:, None] < N) & (offsets_k[None, :] < K)
    linear = offsets_n[:, None] * K + offsets_k[None, :]
    block_id = linear // PACK_BLOCK
    in_block = linear % PACK_BLOCK

    sign_mantissa = tl.load(sign_mantissa_ptr + linear, mask=valid, other=0).to(
        tl.int32
    )
    exponent_pair = tl.load(exponent_nibbles_ptr + linear // 2, mask=valid, other=0).to(
        tl.int32
    )
    exponent_delta = (exponent_pair >> ((linear & 1) * 4)) & 0xF
    base_exponent = tl.load(base_exponent_ptr + block_id, mask=valid, other=0).to(
        tl.int32
    )
    fallback_slot = tl.load(fallback_slot_ptr + block_id, mask=valid, other=-1)

    packed_bits = (
        ((sign_mantissa & 0x80) << 8)
        | ((base_exponent + exponent_delta) << 7)
        | (sign_mantissa & 0x7F)
    )
    fallback_bits = (
        tl.load(
            fallback_bits_ptr + fallback_slot * PACK_BLOCK + in_block,
            mask=valid & (fallback_slot >= 0),
            other=0,
        ).to(tl.int32)
        & 0xFFFF
    )
    fp32_bits = tl.where(fallback_slot >= 0, fallback_bits, packed_bits) << 16
    weight = tl.inline_asm_elementwise(
        "mov.b32 $0, $1;",
        "=f,r",
        [fp32_bits],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    x = tl.load(x_ptr + offsets_k, mask=offsets_k < K, other=0.0)
    accum = tl.sum(weight * x[None, :].to(tl.float32), axis=1)
    tl.store(output_ptr + offsets_n, accum, mask=offsets_n < N)


class _PackedBF16LMHeadKernel(VllmJitKernel["_PackedBF16LMHeadKernel.CompileKey"]):
    @dataclass(frozen=True, slots=True)
    class CompileKey:
        n: int
        k: int
        block_n: int
        num_warps: int

    def dispatch(self, **kwargs: Any) -> CompileKey:
        return self.CompileKey(
            n=kwargs["n"],
            k=kwargs["k"],
            block_n=kwargs["block_n"],
            num_warps=kwargs["num_warps"],
        )

    def get_warmup_keys(self, **kwargs) -> list[CompileKey]:
        return [self.dispatch(**kwargs)]

    def compile(self, compile_key: CompileKey) -> None:
        n = compile_key.n
        k = compile_key.k
        block_n = compile_key.block_n
        num_warps = compile_key.num_warps
        _lossless_packed_bf16_lm_head_kernel.warmup(
            TritonWarmupTensor(torch.bfloat16, shape=(1, k)),
            TritonWarmupTensor(torch.uint8),
            TritonWarmupTensor(torch.uint8),
            TritonWarmupTensor(torch.uint8),
            TritonWarmupTensor(torch.int32),
            TritonWarmupTensor(torch.int16),
            TritonWarmupTensor(torch.bfloat16, shape=(1, n)),
            N=n,
            K=k,
            BLOCK_N=block_n,
            BLOCK_K=triton.next_power_of_2(k),
            PACK_BLOCK=_PACK_BLOCK,
            num_warps=num_warps,
            num_stages=1,
            grid=(1,),
        )


_PACKED_KERNEL = _PackedBF16LMHeadKernel()


def _packed_bf16_lm_head_impl(
    x: torch.Tensor,
    sign_mantissa: torch.Tensor,
    exponent_nibbles: torch.Tensor,
    base_exponent: torch.Tensor,
    fallback_slot: torch.Tensor,
    fallback_bits: torch.Tensor,
    n: int,
    k: int,
    block_n: int,
    num_warps: int,
) -> torch.Tensor:
    output = torch.empty((*x.shape[:-1], n), dtype=x.dtype, device=x.device)
    _lossless_packed_bf16_lm_head_kernel[(cdiv(n, block_n),)](
        x,
        sign_mantissa,
        exponent_nibbles,
        base_exponent,
        fallback_slot,
        fallback_bits,
        output,
        N=n,
        K=k,
        BLOCK_N=block_n,
        BLOCK_K=triton.next_power_of_2(k),
        PACK_BLOCK=_PACK_BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


def _packed_bf16_lm_head_fake(
    x: torch.Tensor,
    sign_mantissa: torch.Tensor,
    exponent_nibbles: torch.Tensor,
    base_exponent: torch.Tensor,
    fallback_slot: torch.Tensor,
    fallback_bits: torch.Tensor,
    n: int,
    k: int,
    block_n: int,
    num_warps: int,
) -> torch.Tensor:
    del (
        sign_mantissa,
        exponent_nibbles,
        base_exponent,
        fallback_slot,
        fallback_bits,
        k,
        block_n,
        num_warps,
    )
    return x.new_empty((*x.shape[:-1], n))


direct_register_custom_op(
    op_name="lossless_packed_bf16_lm_head",
    op_func=_packed_bf16_lm_head_impl,
    mutates_args=[],
    fake_impl=_packed_bf16_lm_head_fake,
)


def _set_buffer(layer: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    if name in layer._buffers:
        setattr(layer, name, value)
    else:
        layer.register_buffer(name, value, persistent=False)


def _clear_packed_state(layer: torch.nn.Module) -> None:
    for name in _STATE_NAMES:
        layer._buffers.pop(_STATE_PREFIX + name, None)
    if hasattr(layer, "_vllm_lossless_lm_head_meta"):
        del layer._vllm_lossless_lm_head_meta


def _prepare_packed_weight(layer: torch.nn.Module) -> None:
    from vllm.config import get_current_vllm_config_or_none

    _clear_packed_state(layer)
    config = get_current_vllm_config_or_none()
    if config is None:
        return

    weight = getattr(layer, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return
    packed = pack_bf16_weight(
        weight,
        max_packed_fraction=config.kernel_config.lm_head_max_packed_fraction,
    )
    if packed is None:
        return

    plan, tensors = packed
    launch = choose_launch_config(plan.k)
    for name, tensor in zip(_STATE_NAMES, tensors):
        _set_buffer(layer, _STATE_PREFIX + name, tensor)
    layer._vllm_lossless_lm_head_meta = (
        plan.n,
        plan.k,
        launch.block_n,
        launch.num_warps,
    )
    _PACKED_KERNEL.register_warmup(
        n=plan.n,
        k=plan.k,
        block_n=launch.block_n,
        num_warps=launch.num_warps,
    )
    logger.info_once(
        "Lossless packed BF16 lm_head ready: shape=(%d, %d), %.3f MiB "
        "(%.3f of dense), %d fallback blocks",
        plan.n,
        plan.k,
        plan.packed_bytes / (1024 * 1024),
        plan.packed_fraction,
        plan.fallback_blocks,
    )


def _try_apply_packed_weight(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor | None:
    if envs.VLLM_BATCH_INVARIANT:
        return None
    meta = getattr(layer, "_vllm_lossless_lm_head_meta", None)
    if meta is None or bias is not None:
        return None
    n, k, block_n, num_warps = meta
    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.shape[-1] != k
        or x.numel() != k
        or not x.is_contiguous()
    ):
        return None
    buffers = [getattr(layer, _STATE_PREFIX + name) for name in _STATE_NAMES]
    return torch.ops.vllm.lossless_packed_bf16_lm_head(
        x, *buffers, n, k, block_n, num_warps
    )


class LosslessPackedLMHeadMethod(QuantizeMethodBase):
    """Decorate an unquantized lm-head method with an M=1 packed fast path."""

    supports_pre_processed_weights = True

    def __init__(self, fallback: QuantizeMethodBase) -> None:
        self.fallback = fallback

    def __getattr__(self, name: str) -> Any:
        """Preserve the complete interface of the decorated method."""
        fallback = object.__getattribute__(self, "fallback")
        return getattr(fallback, name)

    def create_weights(
        self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs
    ):
        return self.fallback.create_weights(layer, *weight_args, **extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.fallback.process_weights_after_loading(layer)
        _prepare_packed_weight(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = _try_apply_packed_weight(layer, x, bias)
        if output is not None:
            return output
        return self.fallback.apply(layer, x, bias)

    def tie_weights(self, layer: torch.nn.Module, embed_tokens: torch.nn.Module):
        return self.fallback.tie_weights(layer, embed_tokens)
