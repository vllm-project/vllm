# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ulysses attention for Muse Spark prefill on ROCm.

Enabled with `additional_config={"ulysses_prefill": true}`. It runs on eager steps
whose residual stream is token-sharded across the TP group (`TokenShard` in
`model.py`) and replaces what that stream does around attention and the MoE input:

* attention: every rank projects its token shard with the full (all-gathered, FP8)
  QKV-gate weight, one all-to-all hands every rank all tokens of its own heads
  (the TP layout, so KV insert and attention are unchanged), and a second
  all-to-all returns the attention output of all heads for the rank's tokens, which
  is gated locally and projected by the full O weight with no reduction;
* MoE input: the pre-expert-normed pooled rows are quantized to MXFP4 on the shard
  and gathered as FP4 bytes plus E8M0 scales instead of BF16 rows.

On these steps the QKV-gate and O projections run as FP8 W8A8 (per-channel weights,
per-token activations) from FP8 copies of the BF16/FP16 weights. The copies stay
resident on every rank; they are built after weight loading, so the KV cache
sizing accounts for them.
"""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    from .model import TokenShard

logger = init_logger(__name__)

_OPTION = "ulysses_prefill"


def _enabled_option(vllm_config: VllmConfig) -> bool:
    extra = vllm_config.additional_config
    if not isinstance(extra, dict):
        return False
    for key in extra:
        if key.startswith("ulysses") and key != _OPTION:
            raise ValueError(
                f"Unknown additional_config key {key!r}; the Muse Spark Ulysses "
                f"prefill key is {_OPTION!r}."
            )
    value = extra.get(_OPTION, False)
    # Exact type: a string such as "false" is truthy.
    if type(value) is not bool:
        raise ValueError(
            f"additional_config {_OPTION!r} must be a JSON boolean, got {value!r}."
        )
    return value


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


def _eager_min_tokens(vllm_config: VllmConfig, min_tokens: int) -> int:
    """The smallest step of at least `min_tokens` tokens that runs eagerly: graph
    replays never take the Ulysses path, and piecewise graphs cover the mixed
    steps up to the largest capture size."""
    compilation = vllm_config.compilation_config
    if compilation.cudagraph_mode.has_piecewise_cudagraphs():
        largest = compilation.max_cudagraph_capture_size or 0
        min_tokens = max(min_tokens, largest + 1)
    return min_tokens


def _eager_steps_reach(vllm_config: VllmConfig, min_tokens: int) -> bool:
    """Whether a step of at least `min_tokens` tokens can run eagerly."""
    return (
        _eager_min_tokens(vllm_config, min_tokens)
        <= vllm_config.scheduler_config.max_num_batched_tokens
    )


def ulysses_prefill_enabled(
    vllm_config: VllmConfig, token_shard_min_tokens: int
) -> bool:
    """Whether the token-sharded eager steps run Ulysses
    (`additional_config={"ulysses_prefill": true}`).

    `token_shard_min_tokens` is the model's token-sharded stream threshold (0 when
    the stream is never sharded).
    """
    if not _enabled_option(vllm_config):
        return False
    if token_shard_min_tokens <= 0:
        logger.warning_once(
            "Muse Spark ulysses_prefill is ignored: it runs on the token-sharded "
            "residual stream, which needs VLLM_ROCM_MUSE_SP_MIN_TOKENS > 0, TP > 1 "
            "with expert parallelism or VLLM_ROCM_MUSE_MOE_REDUCE_DTYPE=bf16, only "
            "sparse MLP layers, and no PP, DP, context parallelism or "
            "sequence-parallel MoE."
        )
        return False
    if vllm_config.lora_config is not None:
        raise ValueError(
            "Muse Spark ulysses_prefill does not support LoRA: its steps project "
            "with copies of the base QKV and O weights."
        )
    if not _eager_steps_reach(vllm_config, token_shard_min_tokens):
        logger.warning_once(
            "Muse Spark ulysses_prefill is ignored: no eager step reaches "
            "VLLM_ROCM_MUSE_SP_MIN_TOKENS (%d) within max_num_batched_tokens (%d) "
            "and above the CUDA graph capture sizes.",
            token_shard_min_tokens,
            vllm_config.scheduler_config.max_num_batched_tokens,
        )
        return False
    if not (_on_gfx950() and rocm_aiter_ops.is_enabled()):
        raise ValueError(
            "Muse Spark ulysses_prefill needs a gfx950 GPU with AITER "
            "(VLLM_ROCM_USE_AITER=1)."
        )
    return True


def warmup(worker: "Worker") -> None:
    """Run one eager step at each step size whose MoE expert kernels the startup
    steps have not launched.

    AITER picks the experts' FlyDSL stage tiles by padded token count, and FlyDSL
    compiles a tile on its first launch, which takes seconds. The profile run only
    reaches `max_num_batched_tokens` and CUDA graph capture only the capture sizes,
    so without this the first prompt chunk of another size compiles while serving.
    """
    from .decode_moe import native_routes_token_counts
    from .model import MuseSparkModel, MuseSparkMoE

    model = next(
        (m for m in worker.get_model().modules() if isinstance(m, MuseSparkModel)),
        None,
    )
    if model is None or not model.ulysses_prefill:
        return
    config = worker.vllm_config
    first = _eager_min_tokens(config, model.token_shard_min_tokens)
    last = config.scheduler_config.max_num_batched_tokens
    num_tokens = sorted(
        {
            n
            for moe in model.modules()
            if isinstance(moe, MuseSparkMoE)
            for n in native_routes_token_counts(moe, first, last)
        }
    )
    if not num_tokens:
        return
    start = time.perf_counter()
    for n in num_tokens:
        worker.model_runner._dummy_run(num_tokens=n, skip_eplb=True, is_profile=True)
    logger.info(
        "Muse Spark Ulysses prefill: warmed up the MoE expert kernels of %s-token "
        "steps in %.1fs.",
        "/".join(map(str, num_tokens)),
        time.perf_counter() - start,
    )


_PREQUANT_ATTR = "_muse_ulysses_prequant"


def attach_prequant(x: torch.Tensor, a1: torch.Tensor, scale: torch.Tensor) -> None:
    """Attach one pre-quantized MXFP4 row per row of `x` (FP4 `a1`, E8M0 `scale`)
    for the experts, which take it instead of quantizing `x` (`take_prequant`)."""
    setattr(x, _PREQUANT_ATTR, (a1, scale))


def take_prequant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Detach the pre-quantized rows attached to `x` (None when there are none)."""
    prequant = x.__dict__.pop(_PREQUANT_ATTR, None)
    if prequant is not None and prequant[0].shape[0] != x.shape[0]:
        raise RuntimeError(
            f"{prequant[0].shape[0]} pre-quantized MoE rows for {x.shape[0]} tokens."
        )
    return prequant


def gather_bytes(xs: torch.Tensor, shard: "TokenShard") -> torch.Tensor:
    """All-gather a byte-typed tensor (fp4x2 / e8m0) as bf16 words, so the custom AG
    carries it.
    """
    xs = xs.contiguous()
    words = xs.view(torch.uint8).view(torch.bfloat16)
    return shard.gather(words).view(torch.uint8).view(xs.dtype)


def _group():
    return get_tp_group().device_group


def _pad(x: torch.Tensor, shard: "TokenShard") -> torch.Tensor:
    padded = shard.rows * shard.world
    if padded == x.shape[0]:
        return x
    return torch.cat([x, x.new_zeros((padded - x.shape[0], *x.shape[1:]))])


def _wide(x: torch.Tensor) -> torch.Tensor:
    """View the last dim as 8-byte words so layout copies run vectorized (generic
    byte/half strided copies are much slower).
    """
    if x.shape[-1] * x.element_size() % 8 == 0 and x.stride(-1) == 1:
        return x.view(torch.int64)
    return x


def expand_rows(x: torch.Tensor, k: int) -> torch.Tensor:
    """Repeat each row k times ([T, C] -> [T*k, C]) with wide copies."""
    w = _wide(x.contiguous())
    # contiguous(): for one row, reshape returns a stride-0 view instead of a copy.
    out = w[:, None, :].expand(w.shape[0], k, w.shape[1]).reshape(-1, w.shape[1])
    return out.contiguous().view(x.dtype)


def a2a_tokens_to_heads(
    ys: torch.Tensor, cols: int, shard: "TokenShard"
) -> torch.Tensor:
    """[rows, W*cols] (columns grouped by owning rank) -> [T, cols] (all tokens)."""
    view = _wide(ys.view(shard.rows, shard.world, cols))
    send = view.transpose(0, 1).contiguous().view(ys.dtype)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=_group())
    return recv.view(shard.rows * shard.world, cols)[: shard.num_tokens]


def a2a_heads_to_tokens(y: torch.Tensor, shard: "TokenShard") -> torch.Tensor:
    """[T, cols] (this rank's heads) -> [rows, W*cols] (all heads, rank order)."""
    cols = y.shape[1]
    send = _pad(y, shard).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=_group())
    r = _wide(recv.view(shard.world, shard.rows, cols))
    return r.transpose(0, 1).reshape(shard.rows, -1).view(y.dtype)


FP8Weight = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class UlyssesWeights:
    """One attention layer's full projection weights (all ranks' heads, rank-major)
    as per-output-channel FP8 with `[1, N]` scales."""

    qkv: FP8Weight
    gate: FP8Weight
    o: FP8Weight

    @property
    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (*self.qkv, *self.gate, *self.o)

    @property
    def nbytes(self) -> int:
        return sum(t.nbytes for t in self.tensors)

    def refill(self, new: "UlyssesWeights") -> None:
        """Copy `new` into these tensors, so they keep their storage."""
        for held, t in zip(self.tensors, new.tensors, strict=True):
            held.copy_(t)


def _all_gather_weight(layer: torch.nn.Module) -> list[torch.Tensor]:
    local = layer.weight
    if local.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            "Muse Spark ulysses_prefill needs BF16 or FP16 attention weights, got "
            f"{local.dtype} for {type(layer).__name__}."
        )
    local = local.data.contiguous()
    parts = [
        torch.empty_like(local) for _ in range(get_tensor_model_parallel_world_size())
    ]
    dist.all_gather(parts, local, group=_group())
    return parts


def _quant_weight(full: torch.Tensor) -> FP8Weight:
    weight, scale = ops.scaled_fp8_quant(full, use_per_token_if_dynamic=True)
    return weight, scale.view(1, -1)


def build_weights(
    qkv_proj: torch.nn.Module, o_proj: torch.nn.Module, q: int, kv: int
) -> UlyssesWeights:
    """The full FP8 weights of the TP-sharded fused `[q|k|v|o-gate]` (`q`, `kv`:
    this rank's sizes) and O projections."""
    parts = _all_gather_weight(qkv_proj)
    qkv = torch.cat([p[: q + 2 * kv] for p in parts])
    gate = torch.cat([p[q + 2 * kv : 2 * q + 2 * kv] for p in parts])
    del parts
    o = torch.cat(_all_gather_weight(o_proj), dim=1)
    # Release the cached blocks so the resident copies get blocks of their own size.
    torch.accelerator.empty_cache()
    return UlyssesWeights(_quant_weight(qkv), _quant_weight(gate), _quant_weight(o))


def quant_rows(x: torch.Tensor) -> FP8Weight:
    """Per-token dynamic FP8 quantization of `x` (aiter)."""
    return rocm_aiter_ops.per_token_quant(x.contiguous(), current_platform.fp8_dtype())


def gemm(xq: FP8Weight, w: FP8Weight, out_dtype: torch.dtype) -> torch.Tensor:
    """FP8 GEMM of the `quant_rows` output `xq` with `w` (aiter CK)."""
    return rocm_aiter_ops.w8a8_gemm(xq[0], w[0], xq[1], w[1], None, out_dtype)
