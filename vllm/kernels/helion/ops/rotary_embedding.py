# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helion kernel implementations for rotary positional embedding (RoPE).

These kernels register as implementations of ``vllm.ir.ops.rotary_embedding_neox``
and ``vllm.ir.ops.rotary_embedding_gptj`` using the vLLM IR dispatch mechanism.

The semantic contract (signature, cos_sin_cache layout) is defined in
``vllm/ir/ops/rotary_embedding.py``; this file only provides the optimized
Helion/Triton-backed implementation.

Kernel config key format: ``"rotarydim_{rotary_dim}_numtokens_{num_tokens}"``
"""

from typing import Any

import regex as re
import torch
from torch import Tensor

from vllm import ir
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_helion

logger = init_logger(__name__)

_HELION_AVAILABLE = has_helion()
_CUDA_ALIKE = current_platform.is_cuda_alike()

if _HELION_AVAILABLE:
    import helion.language as hl
    from vllm.kernels.helion.register import register_kernel


# ---------------------------------------------------------------------------
# Config picker (shared by both variants)
# ---------------------------------------------------------------------------

def _pick_rope_config(
    args: tuple[Any, ...], config_keys: list[str]
) -> str | None:
    """Pick the best pre-tuned config for the given (rotary_dim, num_tokens).

    Config keys follow the format ``"rotarydim_{int}_numtokens_{int}"``.
    Selection strategy: closest rotary_dim, then ceiling num_tokens.
    """
    if not config_keys:
        return None

    # args = (positions, query, key, head_size, rotary_dim, cos_sin_cache)
    _pos, query, _key, _head_size, rotary_dim, _cache = args
    num_tokens = query.shape[0]

    parsed: dict[int, list[int]] = {}
    for k in config_keys:
        if k == "default":
            continue
        m = re.fullmatch(r"rotarydim_(\d+)_numtokens_(\d+)", k)
        if not m:
            raise ValueError(
                f"Malformed config key '{k}', "
                f"expected format 'rotarydim_{{int}}_numtokens_{{int}}'"
            )
        parsed.setdefault(int(m.group(1)), []).append(int(m.group(2)))

    if not parsed:
        return "default" if "default" in config_keys else None

    best_rdim = min(parsed, key=lambda d: abs(d - rotary_dim))
    avail = sorted(parsed[best_rdim])
    best_n = next((n for n in avail if n >= num_tokens), avail[-1])
    return f"rotarydim_{best_rdim}_numtokens_{best_n}"


# ---------------------------------------------------------------------------
# Input generator (shared by both variants)
# ---------------------------------------------------------------------------

def _generate_rope_inputs() -> dict[str, tuple[Any, ...]]:
    """Representative inputs for autotuning. Covers Llama-3 / DeepSeek shapes.

    We use a sparse set of num_tokens values (powers-of-two plus a few
    cudagraph capture sizes) rather than every possible batch size.  The
    optimal tile configuration changes very little between adjacent token
    counts, so dense sampling would waste autotuning time without improving
    the selected configs.
    """
    rotary_dims = [64, 128]
    num_q_heads, num_kv_heads = 32, 8
    # Sparse coverage: small (1-8), medium (16-128), large (256-512).
    num_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    inputs: dict[str, tuple[Any, ...]] = {}
    for rotary_dim in rotary_dims:
        head_size = rotary_dim
        for num_tokens in num_tokens_list:
            positions = torch.randint(
                0, 2048, (num_tokens,), device="cuda", dtype=torch.int64
            )
            query = torch.randn(
                num_tokens, num_q_heads * head_size,
                device="cuda", dtype=torch.bfloat16,
            )
            key = torch.randn(
                num_tokens, num_kv_heads * head_size,
                device="cuda", dtype=torch.bfloat16,
            )
            cos_sin_cache = torch.randn(
                2048, rotary_dim, device="cuda", dtype=torch.bfloat16,
            )
            config_key = f"rotarydim_{rotary_dim}_numtokens_{num_tokens}"
            inputs[config_key] = (
                positions, query, key, head_size, rotary_dim, cos_sin_cache,
            )
    return inputs


# ---------------------------------------------------------------------------
# Helion kernels (only defined when Helion is available on CUDA)
# ---------------------------------------------------------------------------

if _HELION_AVAILABLE and _CUDA_ALIKE:

    @register_kernel(
        config_picker=_pick_rope_config,
        input_generator=_generate_rope_inputs,
    )
    def _helion_rotary_embedding_neox(
        positions: Tensor,
        query: Tensor,
        key: Tensor,
        head_size: int,
        rotary_dim: int,
        cos_sin_cache: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Helion/Triton RoPE kernel – Neox-style."""
        num_tokens = hl.specialize(query.shape[0])
        num_q_heads = hl.specialize(query.shape[1] // head_size)
        num_kv_heads = hl.specialize(key.shape[1] // head_size)
        rot_half = rotary_dim // 2

        # Reshape to 3D outside the tile loop (static shapes only).
        # Helion requires all in-loop allocations to have compile-time shapes.
        q_3d = query.view(num_tokens, num_q_heads, head_size)
        k_3d = key.view(num_tokens, num_kv_heads, head_size)
        q_out_3d = torch.empty_like(q_3d)
        k_out_3d = torch.empty_like(k_3d)

        # Copy unrotated tail; slice assign is outside hl.tile which causes a
        # TensorOperationInWrapper warning but is the only valid approach since
        # Helion forbids hl.tile inside an `if` block (NestedGridLoop error).
        # This path is only taken for the rare case where rotary_dim < head_size.
        if rotary_dim < head_size:
            q_out_3d[:, :, rotary_dim:] = q_3d[:, :, rotary_dim:]
            k_out_3d[:, :, rotary_dim:] = k_3d[:, :, rotary_dim:]

        for tile_tok, tile_pair in hl.tile([num_tokens, rot_half]):
            pos = positions[tile_tok]
            cos_half = cos_sin_cache[pos, tile_pair]        # [tile_tok, tile_pair]
            sin_half = cos_sin_cache[pos, tile_pair + rot_half]
            cos_h = cos_half.unsqueeze(1)                   # [tile_tok, 1, tile_pair]
            sin_h = sin_half.unsqueeze(1)

            # query: Neox pairs at [tile_pair] and [tile_pair + rot_half]
            q1 = q_3d[tile_tok, :, tile_pair]
            q2 = q_3d[tile_tok, :, tile_pair + rot_half]
            q_out_3d[tile_tok, :, tile_pair] = q1 * cos_h - q2 * sin_h
            q_out_3d[tile_tok, :, tile_pair + rot_half] = q2 * cos_h + q1 * sin_h

            # key
            k1 = k_3d[tile_tok, :, tile_pair]
            k2 = k_3d[tile_tok, :, tile_pair + rot_half]
            k_out_3d[tile_tok, :, tile_pair] = k1 * cos_h - k2 * sin_h
            k_out_3d[tile_tok, :, tile_pair + rot_half] = k2 * cos_h + k1 * sin_h

        return q_out_3d.view_as(query), k_out_3d.view_as(key)

    @register_kernel(
        config_picker=_pick_rope_config,
        input_generator=_generate_rope_inputs,
    )
    def _helion_rotary_embedding_gptj(
        positions: Tensor,
        query: Tensor,
        key: Tensor,
        head_size: int,
        rotary_dim: int,
        cos_sin_cache: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Helion/Triton RoPE kernel – GPT-J-style.

        GPT-J uses interleaved (even/odd) pairs: position i maps to head
        indices (2i, 2i+1).  This requires arithmetic on tile indices
        (tile_pair * 2) which Helion's TileIndexType does not support.

        Work-around: reshape input to [..., rot_half, 2] (4-D view) so even/odd
        pairs become the last dim, indexed with plain ints 0/1.  This avoids
        tile-index arithmetic and strided slices, neither of which Helion
        supports.
        """
        num_tokens = hl.specialize(query.shape[0])
        num_q_heads = hl.specialize(query.shape[1] // head_size)
        num_kv_heads = hl.specialize(key.shape[1] // head_size)
        rot_half = rotary_dim // 2

        # Reshape to 4D so even/odd pairs are the last dim (index 0/1).
        # Helion does not support tile-index arithmetic (tile*2) or strided
        # slices, so this 4D view is the only way to express interleaved pairs.
        # Requires head_size == rotary_dim; when rotary_dim < head_size the
        # caller is expected to pre-split query/key to the rotary portion and
        # handle the tail separately (the IR wrapper does this).
        q_rot4 = query.view(num_tokens, num_q_heads, rot_half, 2)
        k_rot4 = key.view(num_tokens, num_kv_heads, rot_half, 2)
        q_out4 = torch.empty_like(q_rot4)
        k_out4 = torch.empty_like(k_rot4)

        for tile_tok, tile_pair in hl.tile([num_tokens, rot_half]):
            pos = positions[tile_tok]
            cos_h = cos_sin_cache[pos, tile_pair]
            sin_h = cos_sin_cache[pos, tile_pair + rot_half]
            cos_h2 = cos_h.unsqueeze(1)
            sin_h2 = sin_h.unsqueeze(1)

            q1 = q_rot4[tile_tok, :, tile_pair, 0]
            q2 = q_rot4[tile_tok, :, tile_pair, 1]
            q_out4[tile_tok, :, tile_pair, 0] = q1 * cos_h2 - q2 * sin_h2
            q_out4[tile_tok, :, tile_pair, 1] = q2 * cos_h2 + q1 * sin_h2

            k1 = k_rot4[tile_tok, :, tile_pair, 0]
            k2 = k_rot4[tile_tok, :, tile_pair, 1]
            k_out4[tile_tok, :, tile_pair, 0] = k1 * cos_h2 - k2 * sin_h2
            k_out4[tile_tok, :, tile_pair, 1] = k2 * cos_h2 + k1 * sin_h2

        return q_out4.view_as(query), k_out4.view_as(key)

    # ------------------------------------------------------------------
    # Register Helion kernels as vLLM IR implementations
    # ------------------------------------------------------------------

    @ir.ops.rotary_embedding_neox.register_impl(
        "helion", supported=_CUDA_ALIKE
    )
    def _neox_impl(
        positions: Tensor,
        query: Tensor,
        key: Tensor,
        head_size: int,
        rotary_dim: int,
        cos_sin_cache: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return _helion_rotary_embedding_neox(
            positions, query, key, head_size, rotary_dim, cos_sin_cache
        )

    @ir.ops.rotary_embedding_gptj.register_impl(
        "helion", supported=_CUDA_ALIKE
    )
    def _gptj_impl(
        positions: Tensor,
        query: Tensor,
        key: Tensor,
        head_size: int,
        rotary_dim: int,
        cos_sin_cache: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return _helion_rotary_embedding_gptj(
            positions, query, key, head_size, rotary_dim, cos_sin_cache
        )

    # Public aliases so tests can import them directly.
    rotary_embedding_neox = _helion_rotary_embedding_neox
    rotary_embedding_gptj = _helion_rotary_embedding_gptj

else:
    # Stubs so test imports don't fail when Helion is unavailable.
    def rotary_embedding_neox(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("rotary_embedding_neox requires Helion to be installed.")

    def rotary_embedding_gptj(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("rotary_embedding_gptj requires Helion to be installed.")


def rotary_embedding_baseline(
    positions: Tensor,
    query: Tensor,
    key: Tensor | None,
    head_size: int,
    rotary_dim: int,
    cos_sin_cache: Tensor,
    is_neox_style: bool,
) -> tuple[Tensor, Tensor | None]:
    """Pure-PyTorch reference using ``RotaryEmbedding.forward_static``.

    Used in correctness tests to verify the Helion kernel output.
    """
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    return RotaryEmbedding.forward_static(
        positions=positions,
        query=query,
        key=key,
        head_size=head_size,
        rotary_dim=rotary_dim,
        cos_sin_cache=cos_sin_cache,
        is_neox_style=is_neox_style,
    )
