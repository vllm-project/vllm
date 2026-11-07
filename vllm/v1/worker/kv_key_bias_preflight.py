# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup check: a large layer-0 key bias under a quantized KV cache.

Some models carry a large bias on the key projection of their first attention
layer: the Qwen2 and Qwen2.5 families and their fine-tuned and distilled
derivatives. Under a per-token, per-head quantized KV cache
(``int4_per_token_head``; fp8 at default scales on some of them) these models
answer retrieval prompts at chance or worse, and the engine reports success.

Why: the bias adds the same offset to every cached key of that layer. The
quantizer sets its per-token range from that offset instead of from the
signal, so the signal rounds away. The rotated query then reads noise.
Leaving the first layer unquantized (``--kv-cache-dtype-skip-layers 0``)
restores the model and costs a few percent of the quantized option's
throughput.

What this check reads: the largest absolute key-bias entry on the slowest
rotating eighth of the rotary pairs of the first attention layer. Those dims
rotate least with position, so they carry the bias into every key almost
unchanged. On every checkpoint where this statistic was read next to a
collapse count, the models that collapsed read 116 to 456 and the models that
held read 12 to 73. A 49-checkpoint survey on the coarser all-dims maximum
also puts every non-Qwen model at or below 73. The threshold of 100 sits in
that gap.

The check only warns. It changes no behaviour. It is skipped when the cache
is not quantized, when layer 0 is already in the skip list, or when the first
attention layer has no key bias.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

KEY_BIAS_WARN_THRESHOLD: float = 100.0
"""The warning fires above this slowest-band |key bias|.

Measured: collapsed models read 116 to 456, intact ones at most 73 (module
docstring). A constant, not a config option, because the check only warns.
"""


def slowest_band_key_bias(
    k_bias: torch.Tensor,
    head_dim: int,
    rotary_dim: int | None = None,
    is_neox_style: bool = True,
) -> float:
    """Largest |bias| on the slowest-rotating eighth of the rotary pairs.

    ``k_bias`` is the key projection's bias for this rank, with shape
    ``[num_kv_heads * head_dim]``.

    Rotary pairs are numbered by frequency: pair 0 rotates fastest and the
    last pair slowest. The slowest eighth is the last ``rotary_dim // 16``
    pairs. Which two dims form a pair depends on the layout:

    - NeoX style: dim ``d`` pairs with ``d + rotary_dim // 2``.
    - GPT-J style (interleaved): pair ``j`` is dims ``2j`` and ``2j + 1``.

    Dims beyond ``rotary_dim`` (partial rotary) are not rotated and are not
    read.
    """
    if head_dim <= 0 or k_bias.numel() % head_dim != 0:
        raise ValueError(
            f"key bias of {k_bias.numel()} entries is not a whole number of "
            f"heads of head_dim {head_dim}"
        )
    rot = rotary_dim or head_dim
    rot = min(rot, head_dim)
    half = rot // 2
    band = max(half // 8, 1)
    pairs = range(half - band, half)
    if is_neox_style:
        dims = [d for p in pairs for d in (p, p + half)]
    else:
        dims = [d for p in pairs for d in (2 * p, 2 * p + 1)]
    heads = k_bias.detach().to(torch.float32).reshape(-1, head_dim)
    return float(heads[:, dims].abs().max().item())


def _first_attention_key_bias(
    model: torch.nn.Module,
) -> tuple[torch.Tensor, int, int | None, bool] | None:
    """The first attention layer's key bias and its head geometry, or None.

    Walks the module tree in registration order. Stops at the first module
    that has a ``head_dim`` and a biased key projection, in one of two forms:

    - a fused ``qkv_proj`` whose owner exposes ``q_size`` and ``kv_size``
      (the K slice is ``bias[q_size : q_size + kv_size]``);
    - a separate ``k_proj``.

    The returned bias is the K slice this rank holds. Modules without a key
    bias (for example a ``k_norm`` design such as Qwen3) are passed over, so
    the result is None for such models.
    """
    for module in model.modules():
        head_dim = getattr(module, "head_dim", None)
        if not isinstance(head_dim, int) or head_dim <= 0:
            continue
        rotary = getattr(module, "rotary_emb", None)
        rotary_dim = getattr(rotary, "rotary_dim", None)
        is_neox = bool(getattr(rotary, "is_neox_style", True))
        qkv = getattr(module, "qkv_proj", None)
        bias = getattr(qkv, "bias", None) if qkv is not None else None
        q_size = getattr(module, "q_size", None)
        kv_size = getattr(module, "kv_size", None)
        if (
            isinstance(bias, torch.Tensor)
            and isinstance(q_size, int)
            and isinstance(kv_size, int)
            and bias.numel() >= q_size + 2 * kv_size
        ):
            return bias[q_size : q_size + kv_size], head_dim, rotary_dim, is_neox
        k_proj = getattr(module, "k_proj", None)
        bias = getattr(k_proj, "bias", None) if k_proj is not None else None
        if isinstance(bias, torch.Tensor) and bias.numel() % head_dim == 0:
            return bias, head_dim, rotary_dim, is_neox
    return None


def _layer_zero_skipped(skip_layers: Iterable[str]) -> bool:
    return any(str(s).strip() == "0" for s in skip_layers)


def check_key_bias_under_quantized_kv(
    model: torch.nn.Module,
    kv_cache_dtype: str,
    kv_cache_dtype_skip_layers: Iterable[str] = (),
    threshold: float = KEY_BIAS_WARN_THRESHOLD,
) -> float | None:
    """Warn once when a quantized KV cache meets a large layer-0 key bias.

    The check applies when the cache dtype is a quantized one, layer 0 is not
    in the skip list, and the first attention layer has a key bias. It then
    reads the slowest-band statistic and warns if it exceeds ``threshold``.

    Returns the statistic that was read, or None when the check did not
    apply, so callers and tests can see what the model measured.
    """
    if kv_cache_dtype in ("auto", "bfloat16", "float16", "half", "float32"):
        return None
    if _layer_zero_skipped(kv_cache_dtype_skip_layers):
        return None
    found = _first_attention_key_bias(model)
    if found is None:
        return None
    k_bias, head_dim, rotary_dim, is_neox = found
    try:
        stat = slowest_band_key_bias(k_bias, head_dim, rotary_dim, is_neox)
    except ValueError as exc:
        logger.debug("key-bias preflight skipped: %s", exc)
        return None
    if stat > threshold:
        logger.warning_once(
            "The first attention layer has a key bias of %.1f on its slowest "
            "rotary band (threshold %.0f). Models with such a bias (the Qwen2 "
            "and Qwen2.5 families and their derivatives) are known to answer "
            "at chance or worse under kv_cache_dtype=%s: the bias dominates "
            "the per-token quantization range of every cached key in that "
            "layer. Keeping layer 0 unquantized restores the model at a few "
            "percent of throughput: pass --kv-cache-dtype-skip-layers 0. "
            "Otherwise verify your workload against the unquantized cache "
            "before serving.",
            stat,
            threshold,
            kv_cache_dtype,
            scope="global",
        )
    else:
        logger.debug(
            "key-bias preflight: slowest-band |key bias| %.1f under "
            "kv_cache_dtype=%s (threshold %.0f)",
            stat,
            kv_cache_dtype,
            threshold,
        )
    return stat
