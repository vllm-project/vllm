# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only MiniCPM-RobotTrack model compatible with HuggingFace weights.

MiniCPM-RobotTrack is a vision-language-action policy: a bare MiniCPM4-0.5B
decoder backbone plus an input adapter (vision projector + temporal markers +
learnable control query) and a funnel trajectory head that regresses eight
``[x, y, yaw]`` waypoints. It is non-generative (a single causal forward whose
last token drives the head), so it is served as a vLLM pooling model that
advertises the ``"embed"`` task and returns a flat 24-dim vector per request
(reshape to ``[8, 3]`` on the client).
"""

import hashlib
import math
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoImageProcessor, BatchFeature, PretrainedConfig

from vllm.config import VllmConfig
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.inputs import ModalityData, MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.layers.pooler.seqwise import LastPool, SequencePooler
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import BaseDummyInputsBuilder
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptIndexTargets,
    PromptInsertion,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.pool.metadata import PoolingMetadata

from .interfaces import MultiModalEmbeddings, SupportsMultiModal
from .interfaces_base import default_pooling_type
from .minicpm import MiniCPMModel
from .utils import AutoWeightsLoader, WeightsMapper, maybe_prefix

logger = init_logger(__name__)


def _frame_content_key(frame: Any) -> int:
    """Stable content hash of one raw frame, for the in-tower feature cache.

    Byte-identical frames (the same physical frame re-sent as the window slides)
    map to the same key, so the tower reuses each frame's coarse features.
    """
    arr = np.ascontiguousarray(np.asarray(frame))
    digest = hashlib.blake2b(arr.tobytes(), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


def _stream_id_key(stream_id: object) -> int:
    """Stable integer key for a client-supplied stream identifier.

    String ids are hashed so the batched ``stream_id`` mm field stays an int
    (the vLLM IPC serializer only supports tensors / int / float values).
    """
    if isinstance(stream_id, int):
        return stream_id
    if isinstance(stream_id, str):
        digest = hashlib.blake2b(stream_id.encode(), digest_size=8).digest()
        return int.from_bytes(digest, "big", signed=True)
    raise TypeError(f"stream_id must be an int or str, got {type(stream_id).__name__}")


def _pixel_window_cached(
    frames: Sequence[Any],
    keys: Sequence[int],
    cache: "OrderedDict[int, tuple[torch.Tensor, torch.Tensor]]",
    cache_size: int,
    process_misses: Callable[[list[Any]], tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble a window's normalized pixels, reusing per-frame cache entries.

    ``process_misses`` resizes+normalizes a list of frames (all cache misses)
    and returns ``(dino, siglip)`` batches in the same order. Frames already in
    ``cache`` (content-addressed by ``keys``) are served without reprocessing.
    Returns the full window's ``(dino, siglip)`` tensors in window order.

    ``cache_size == 0`` disables reuse: every frame goes through
    ``process_misses``, matching the uncached path exactly.
    """
    by_frame: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    miss_idx: list[int] = []
    for i, key in enumerate(keys):
        cached = cache.get(key) if cache_size > 0 else None
        if cached is not None:
            by_frame[i] = cached
            cache.move_to_end(key)
        else:
            miss_idx.append(i)

    if miss_idx:
        miss_dino, miss_siglip = process_misses([frames[i] for i in miss_idx])
        for j, i in enumerate(miss_idx):
            dino_pixel = miss_dino[j : j + 1]
            siglip_pixel = miss_siglip[j : j + 1]
            by_frame[i] = (dino_pixel, siglip_pixel)
            if cache_size > 0:
                cache[keys[i]] = (dino_pixel, siglip_pixel)
                cache.move_to_end(keys[i])
        while cache_size > 0 and len(cache) > cache_size:
            cache.popitem(last=False)

    dino = torch.cat([by_frame[i][0] for i in range(len(frames))], dim=0)
    siglip = torch.cat([by_frame[i][1] for i in range(len(frames))], dim=0)
    return dino, siglip


@dataclass
class RobotTrackStreamState:
    """Server-side rolling window for one explicitly named client stream.

    ``coarse_history`` holds the pooled coarse tokens (``[coarse_per, C]`` each)
    of the trailing history frames, newest last, excluding the current frame;
    ``current_coarse`` is the latest frame's coarse pool, promoted into history
    when the next frame arrives. ``fine`` is the latest frame's fine pool and
    ``frame_index`` the latest committed camera frame index.
    """

    coarse_history: tuple[torch.Tensor, ...] = ()
    current_coarse: torch.Tensor | None = None
    fine: torch.Tensor | None = None
    frame_index: int | None = None


def _classify_stream_request(
    frame_count: int,
    history_frames: int,
    state: RobotTrackStreamState | None,
    frame_index: int | None,
) -> str:
    """Classify a request against the stream's committed state.

    A ``history_frames + 1``-frame request replaces the stream's window
    (establish or re-sync); a single-frame request appends to an established
    stream when ``frame_index`` is consecutive, or reuses the committed result
    when it repeats the last index (idempotent retry). Returns
    ``"replace"`` / ``"append"`` / ``"reuse"``.
    """
    if frame_count == history_frames + 1:
        if frame_index is not None and frame_index < 0:
            raise ValueError("frame_index must be non-negative.")
        return "replace"
    if frame_count != 1:
        raise ValueError(
            f"frames must contain 1 incremental frame or {history_frames + 1} "
            f"complete-window frames, got {frame_count}."
        )
    if state is None:
        raise ValueError(
            "A single-frame request requires an existing stream. Send a complete "
            f"{history_frames + 1}-frame window first."
        )
    if frame_index is None or state.frame_index is None:
        raise ValueError(
            "Single-frame requests require frame_index, and the preceding "
            "complete window must also specify its final frame_index."
        )
    if frame_index == state.frame_index:
        return "reuse"
    if frame_index != state.frame_index + 1:
        raise ValueError(
            f"Out-of-order frame_index={frame_index}; expected "
            f"{state.frame_index + 1} for this stream."
        )
    return "append"


def _advance_stream_state(
    state: RobotTrackStreamState | None,
    mode: str,
    coarse_by_frame: list[torch.Tensor],
    fine: torch.Tensor,
    frame_index: int,
    history_frames: int,
) -> RobotTrackStreamState:
    """Roll the stream window forward after encoding the received frame(s)."""
    if mode == "replace":
        history = tuple(coarse_by_frame[:-1])[-history_frames:]
        current_coarse = coarse_by_frame[-1]
    else:  # "append"
        # Promote the previous current frame into history, drop the oldest, and
        # make the new frame current — otherwise the window would skip a frame.
        history = (*state.coarse_history[1:], state.current_coarse)[-history_frames:]
        current_coarse = coarse_by_frame[0]
    return RobotTrackStreamState(
        coarse_history=history,
        current_coarse=current_coarse,
        fine=fine,
        frame_index=frame_index,
    )


def _assemble_window_tensors(
    coarse_history: torch.Tensor,
    fine: torch.Tensor,
    history_frames: int,
    coarse_tokens_per_frame: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble ``(coarse, coarse_time, fine, fine_time)`` for the visual bundle.

    ``coarse_history`` is the stacked ``[n, coarse_per, C]`` coarse pools of the
    trailing history frames (newest last); ``fine`` is the current frame's fine
    pool. The stateless ``_encode_window`` and the stateful stream path both
    build exactly this bundle for the same window, so the construction lives
    here once instead of being kept in sync across two copies.
    """
    history = _pad_history_frames(coarse_history, history_frames)
    coarse = history.reshape(-1, history.size(-1))
    device = fine.device
    coarse_time = torch.arange(history_frames, device=device).repeat_interleave(
        coarse_tokens_per_frame
    )
    fine_time = torch.full(
        (fine.size(0),), history_frames, dtype=torch.long, device=device
    )
    return coarse, coarse_time, fine, fine_time


def _assemble_stream_window(
    state: RobotTrackStreamState,
    history_frames: int,
    coarse_tokens_per_frame: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble ``(coarse, coarse_time, fine, fine_time)`` from stream history.

    Re-stacks the committed coarse history through the shared
    ``_assemble_window_tensors``, so the stateful path produces exactly the
    policy input the stateless ``_encode_window`` would for the same window.
    """
    if not state.coarse_history:
        raise RuntimeError("empty stream history")
    return _assemble_window_tensors(
        torch.stack(state.coarse_history),
        state.fine,
        history_frames,
        coarse_tokens_per_frame,
    )


def _square_side(token_count: int) -> int:
    side = int(round(token_count**0.5))
    if side * side != token_count:
        raise ValueError(f"token count {token_count} is not a square grid")
    return side


def _square_side_or_none(token_count: int) -> int | None:
    if token_count <= 0:
        return None
    side = int(round(token_count**0.5))
    return side if side * side == token_count else None


def _grid_pool(tokens: torch.Tensor, grid: int, out_tokens: int) -> torch.Tensor:
    """Average-pool a ``[B, grid*grid, C]`` token grid to ``out_tokens`` tokens."""
    out_side = _square_side(out_tokens)
    batch, _, channels = tokens.shape
    feats = tokens.transpose(1, 2).reshape(batch, channels, grid, grid)
    feats = F.adaptive_avg_pool2d(feats, (out_side, out_side))
    return feats.flatten(2).transpose(1, 2).contiguous()


def _pad_history_frames(history: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Pad ``[n, coarse_per, C]`` history to ``num_frames`` frames.

    Matches the upstream window assembly: repeat the oldest frame to the left,
    then keep the most recent ``num_frames``.
    """
    n = history.size(0)
    if n < num_frames:
        pad = history[:1].expand(num_frames - n, -1, -1)
        history = torch.cat([pad, history], dim=0)
    return history[-num_frames:]


def _encode_frames_cached(
    tower: Any,
    cache: "OrderedDict[int, torch.Tensor]",
    cache_size: int,
    dino_pixels: torch.Tensor,
    siglip_pixels: torch.Tensor,
    frame_keys: Sequence[int],
    coarse_tokens: int,
    fine_tokens: int,
) -> tuple[list[torch.Tensor], torch.Tensor, int]:
    """Encode a window's frames, reusing cached per-frame coarse features.

    Only the coarse pool is cached: it is the per-frame feature reused across
    the sliding window (every frame but the current one feeds history). The
    current frame's fine pool is consumed once, when it is the newest frame, so
    it is always recomputed rather than stored. A coarse cache hit that is also
    the current frame still re-runs the tower on that single frame for its fine.
    Returns the per-frame ``coarse`` pooled tensors (window order), the current
    frame's ``fine`` pool, and the number of frames run through the tower.

    ``cache_size == 0`` disables reuse: every frame is re-encoded, matching the
    uncached path exactly (used as an A/B parity toggle).
    """
    num_frames = len(frame_keys)
    coarse_map: dict[int, torch.Tensor] = {}

    miss_idx: list[int] = []
    for i, key in enumerate(frame_keys):
        cached = cache.get(key) if cache_size > 0 else None
        if cached is not None:
            coarse_map[i] = cached
            cache.move_to_end(key)
        else:
            miss_idx.append(i)
    # The current frame's fine is never cached; a coarse hit that is also the
    # current frame must still run the tower for its fine.
    if num_frames and num_frames - 1 not in miss_idx:
        miss_idx.append(num_frames - 1)

    if not miss_idx:
        # Only an empty window reaches here; the current frame is always encoded.
        raise RuntimeError("empty frame window")

    index = torch.tensor(miss_idx, device=dino_pixels.device)
    fused, grid = tower(
        dino_pixels.index_select(0, index),
        siglip_pixels.index_select(0, index),
    )
    coarse_pool = _grid_pool(fused, grid, coarse_tokens)
    fine_pool = _grid_pool(fused, grid, fine_tokens)
    for j, i in enumerate(miss_idx):
        coarse_map[i] = coarse_pool[j]
        if cache_size > 0:
            cache[frame_keys[i]] = coarse_pool[j]
            cache.move_to_end(frame_keys[i])
            while len(cache) > cache_size:
                cache.popitem(last=False)

    fine = fine_pool[miss_idx.index(num_frames - 1)]
    coarse = [coarse_map[i] for i in range(num_frames)]
    return coarse, fine, len(miss_idx)


def _normalize_windows(frames: Sequence[Any]) -> list[list[Any]]:
    """Normalize processor ``frames`` into a list of per-item windows.

    A frame is one image (PIL or an HxWxC array); a window is the list of frames
    for one mm item. Different code paths hand this over either as a single flat
    window (``[frame, ...]``) or already grouped by item (``[[frame, ...], ...]``
    or a stacked 4-D array per item); normalize both to ``[[frame, ...], ...]``.
    """
    first = frames[0]
    if isinstance(first, (list, tuple)) or getattr(first, "ndim", 0) >= 4:
        return [list(window) for window in frames]
    return [list(frames)]


# ---------------------------------------------------------------------------
# DINOv3 ViT port of transformers ``dinov3_vit``. The DINOv3 half of the
# RobotTrack vision tower runs from this in-tree port instead of a
# ``transformers`` encoder dependency.
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_dinov3_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_prefix_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to patch tokens only; CLS + register tokens are left as-is.

    ``q``/``k`` are ``[B, S, num_heads, head_dim]``; ``cos``/``sin`` are
    ``[num_patches, head_dim]`` and broadcast over batch and heads.
    """
    q_prefix, q_patch = q.split((num_prefix_tokens, q.shape[1] - num_prefix_tokens), 1)
    k_prefix, k_patch = k.split((num_prefix_tokens, k.shape[1] - num_prefix_tokens), 1)
    # cos/sin: [num_patches, head_dim] -> [1, num_patches, 1, head_dim]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    q_patch = (q_patch * cos) + (_rotate_half(q_patch) * sin)
    k_patch = (k_patch * cos) + (_rotate_half(k_patch) * sin)
    q = torch.cat((q_prefix, q_patch), dim=1)
    k = torch.cat((k_prefix, k_patch), dim=1)
    return q, k


class DINOv3ViTEmbeddings(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.cls_token = nn.Parameter(torch.empty(1, 1, hidden))
        self.register_tokens = nn.Parameter(
            torch.empty(1, config.num_register_tokens, hidden)
        )
        self.patch_embeddings = nn.Conv2d(
            config.num_channels,
            hidden,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        target_dtype = self.patch_embeddings.weight.dtype
        patches = self.patch_embeddings(pixel_values.to(target_dtype))
        patches = patches.flatten(2).transpose(1, 2)
        batch = patches.shape[0]
        cls = self.cls_token.expand(batch, -1, -1)
        register = self.register_tokens.expand(batch, -1, -1)
        return torch.cat([cls, register, patches], dim=1)


class DINOv3ViTRopePositionEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.config = config
        self.base = config.rope_theta
        self.head_dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / self.base ** torch.arange(
            0, 1, 4 / self.head_dim, dtype=torch.float32
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, height, width = pixel_values.shape
        patch = self.config.patch_size
        num_h, num_w = height // patch, width // patch
        device = pixel_values.device

        coords_h = torch.arange(0.5, num_h, dtype=torch.float32, device=device) / num_h
        coords_w = torch.arange(0.5, num_w, dtype=torch.float32, device=device) / num_w
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = coords.flatten(0, 1)
        coords = 2.0 * coords - 1.0  # [num_patches, 2] in [-1, 1]

        inv_freq = self.inv_freq.to(device)
        angles = 2 * math.pi * coords[:, :, None] * inv_freq[None, None, :]
        angles = angles.flatten(1, 2).tile(2)  # [num_patches, head_dim]
        cos, sin = torch.cos(angles), torch.sin(angles)
        return cos.to(pixel_values.dtype), sin.to(pixel_values.dtype)


def _create_fake_bias_for_k_proj(
    weights: Iterable[tuple[str, torch.Tensor]], fake_bias_key_name: str
) -> Iterable[tuple[str, torch.Tensor]]:
    """Inject a zeros bias for a bias-less projection into a weight stream.

    Same trick as ``whisper._create_fake_bias_for_k_proj``: a fused
    ``QKVParallelLinear`` has a single bias over q/k/v, but DINOv3's k_proj has
    no bias. The fused bias starts uninitialized, so a zeros k-bias is fed to
    the loader to make the assembled bias ``[q_bias, 0, v_bias]``.
    """
    for name, weight in weights:
        yield name, weight
        if name.endswith(fake_bias_key_name):
            bias = torch.zeros(weight.size(0))
            bias_name = name.replace("weight", "bias")
            yield bias_name, bias


class DINOv3ViTAttention(nn.Module):
    """DINOv3 attention with RoPE on patch tokens only.

    q/k/v are a fused ``QKVParallelLinear`` with one bias; DINOv3's k_proj has
    no bias, so a zeros bias is faked for it at load time (see
    ``_create_fake_bias_for_k_proj``) and the fused bias holds ``[q, 0, v]``.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.num_heads_per_partition = divide(
            self.num_heads, get_tensor_model_parallel_world_size()
        )

        # A fused qkv has one bias; DINOv3's published configs use
        # query_bias=True / key_bias=False / value_bias=True, with the missing
        # k bias faked to zero at load time.
        assert config.query_bias and config.value_bias and not config.key_bias
        self.qkv_proj = QKVParallelLinear(
            self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.embed_dim,
            self.embed_dim,
            bias=config.proj_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.attn = MMEncoderAttention(
            self.num_heads_per_partition,
            self.head_dim,
            scale=self.scale,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        num_prefix_tokens: int,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            (self.num_heads_per_partition * self.head_dim,) * 3, dim=-1
        )
        batch, seq = q.shape[:2]
        q = q.view(batch, seq, self.num_heads_per_partition, self.head_dim)
        k = k.view(batch, seq, self.num_heads_per_partition, self.head_dim)
        q, k = _apply_dinov3_rotary_pos_emb(q, k, cos, sin, num_prefix_tokens)
        out = self.attn(
            q.reshape(batch, seq, -1),
            k.reshape(batch, seq, -1),
            v,
        )
        out, _ = self.o_proj(out)
        return out


class DINOv3ViTLayerScale(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(config.hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.lambda1


class DINOv3ViTMLP(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.mlp_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.mlp_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        # Exact (erf) GELU matches HF ACT2FN["gelu"], not the tanh approx.
        self.act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.up_proj(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class DINOv3ViTLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        eps = config.layer_norm_eps
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=eps)
        self.attention = DINOv3ViTAttention(
            config, quant_config, prefix=f"{prefix}.attention"
        )
        self.layer_scale1 = DINOv3ViTLayerScale(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=eps)
        self.mlp = DINOv3ViTMLP(config, quant_config, prefix=f"{prefix}.mlp")
        self.layer_scale2 = DINOv3ViTLayerScale(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        num_prefix_tokens: int,
    ) -> torch.Tensor:
        attn = self.attention(self.norm1(hidden_states), cos, sin, num_prefix_tokens)
        hidden_states = hidden_states + self.layer_scale1(attn)
        mlp = self.mlp(self.norm2(hidden_states))
        hidden_states = hidden_states + self.layer_scale2(mlp)
        return hidden_states


class DINOv3VisionModel(nn.Module):
    """DINOv3 ViT encoder returning post-norm ``last_hidden_state``."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.num_prefix_tokens = 1 + config.num_register_tokens
        self.embeddings = DINOv3ViTEmbeddings(config)
        self.rope_embeddings = DINOv3ViTRopePositionEmbedding(config)
        self.layer = nn.ModuleList(
            [
                DINOv3ViTLayer(config, quant_config, prefix=f"{prefix}.layer.{i}")
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        cos, sin = self.rope_embeddings(pixel_values)
        for layer in self.layer:
            hidden_states = layer(hidden_states, cos, sin, self.num_prefix_tokens)
        return self.norm(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # mask_token is inference-irrelevant; inv_freq is computed, not stored.
        # DINOv3's k_proj has no bias: fake a zeros bias so the fused qkv_proj
        # bias loads as [q, 0, v] (same trick as whisper).
        weights = _create_fake_bias_for_k_proj(weights, ".k_proj.weight")
        mapper = WeightsMapper(
            orig_to_new_stacked={
                ".attention.q_proj": (".attention.qkv_proj", "q"),
                ".attention.k_proj": (".attention.qkv_proj", "k"),
                ".attention.v_proj": (".attention.qkv_proj", "v"),
            }
        )
        loader = AutoWeightsLoader(self, skip_substrs=["mask_token"])
        return loader.load_weights(weights, mapper=mapper)


class DualVisionTower(nn.Module):
    """Frozen DINOv3 + SigLIP encoder.

    Stateless per-frame function: ``(dino_pixels, siglip_pixels) -> fused grid``.
    DINOv3 patch tokens (CLS + register dropped) are channel-concatenated in
    front of the SigLIP tokens (adaptive-pooled to the DINOv3 grid), yielding a
    1536-dim fused grid per frame. The encoder weights live outside the RobotTrack
    checkpoint and are loaded here from ``dino_model`` / ``siglip_model``.
    """

    def __init__(
        self,
        dino_model: str,
        siglip_model: str,
        quant_config: Any = None,
    ) -> None:
        super().__init__()
        from .siglip import SiglipVisionModel

        dino_config = AutoConfig.from_pretrained(dino_model)
        siglip_config = AutoConfig.from_pretrained(siglip_model)
        siglip_vision_config = getattr(siglip_config, "vision_config", siglip_config)

        self.dino = DINOv3VisionModel(dino_config, quant_config, prefix="dino")
        self.siglip = SiglipVisionModel(
            siglip_vision_config,
            quant_config,
            require_post_norm=True,
            use_head=False,
            prefix="siglip",
        )
        self.num_register = int(getattr(dino_config, "num_register_tokens", 0) or 0)

        self._load_encoder_weights(dino_model, siglip_model)

    def _load_encoder_weights(self, dino_model: str, siglip_model: str) -> None:
        self.dino.load_weights(_iter_safetensors(dino_model))

        def siglip_vision_weights() -> Iterable[tuple[str, torch.Tensor]]:
            prefix = "vision_model."
            for name, weight in _iter_safetensors(siglip_model):
                if not name.startswith(prefix) or name.startswith(prefix + "head."):
                    continue
                yield name[len(prefix) :], weight

        self.siglip.vision_model.load_weights(siglip_vision_weights())

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self, dino_pixels: torch.Tensor, siglip_pixels: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        dino_pixels = dino_pixels.to(self.device, self.dtype)
        siglip_pixels = siglip_pixels.to(self.device, self.dtype)

        dino_tokens = self.dino(dino_pixels)
        dino_tokens = dino_tokens[:, 1 + self.num_register :, :]
        grid = _square_side(dino_tokens.size(1))

        siglip_tokens = self.siglip(siglip_pixels)
        if _square_side_or_none(siglip_tokens.size(1)) is None:
            siglip_tokens = siglip_tokens[:, 1:, :]
        s_grid = _square_side(siglip_tokens.size(1))
        siglip_tokens = siglip_tokens.transpose(1, 2).reshape(
            siglip_tokens.size(0), siglip_tokens.size(2), s_grid, s_grid
        )
        siglip_tokens = (
            F.adaptive_avg_pool2d(siglip_tokens, (grid, grid))
            .flatten(2)
            .transpose(1, 2)
        )
        fused = torch.cat((dino_tokens, siglip_tokens), dim=-1)
        return fused, grid


def _iter_safetensors(model_path: str) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield ``(name, tensor)`` for every weight in a local HF checkpoint dir."""
    import glob
    import os

    from safetensors import safe_open

    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors found under {model_path}")
    for path in files:
        with safe_open(path, framework="pt") as handle:
            keys = list(handle.keys())
            for key in keys:
                yield key, handle.get_tensor(key)


class VisionProjector(nn.Module):
    """Map fused DINOv3+SigLIP features into the MiniCPM hidden space."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class TemporalMarkerEncoder(nn.Module):
    """Build one marker token per represented frame (time+stream+camera)."""

    def __init__(self, hidden_dim: int, max_time_steps: int) -> None:
        super().__init__()
        self.time_embedding = nn.Embedding(max_time_steps, hidden_dim)
        self.stream_embedding = nn.Embedding(2, hidden_dim)
        self.camera_embedding = nn.Embedding(1, hidden_dim)

    def forward(
        self, time_step: int, stream_id: int, device: torch.device
    ) -> torch.Tensor:
        time = torch.tensor([time_step], dtype=torch.long, device=device)
        stream = torch.tensor([stream_id], dtype=torch.long, device=device)
        camera = torch.zeros(1, dtype=torch.long, device=device)
        return (
            self.time_embedding(time)
            + self.stream_embedding(stream)
            + self.camera_embedding(camera)
        ).squeeze(0)


class FunnelTrajectoryHead(nn.Module):
    """Six-layer funnel MLP predicting a fixed waypoint trajectory."""

    def __init__(
        self,
        hidden_dim: int,
        num_waypoints: int,
        action_dim: int,
        dropout: float,
        use_tanh: bool,
    ) -> None:
        super().__init__()
        output_dim = num_waypoints * action_dim
        self.num_waypoints = num_waypoints
        self.action_dim = action_dim
        self.use_tanh = use_tanh
        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 4096),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4096, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(128),
            nn.Linear(128, output_dim),
        )

    def forward(self, control_state: torch.Tensor) -> torch.Tensor:
        trajectory = self.layers(control_state)
        if self.use_tanh:
            trajectory = torch.tanh(trajectory)
        return trajectory.view(-1, self.num_waypoints, self.action_dim)


def _robottrack_field_config(
    hf_inputs: Mapping[str, torch.Tensor],
) -> Mapping[str, MultiModalFieldConfig]:
    coarse_lengths = hf_inputs.get("coarse_lengths", torch.empty(0, dtype=torch.long))
    fine_lengths = hf_inputs.get("fine_lengths", torch.empty(0, dtype=torch.long))
    return dict(
        coarse_tokens=MultiModalFieldConfig.flat_from_sizes("image", coarse_lengths),
        coarse_time_indices=MultiModalFieldConfig.flat_from_sizes(
            "image", coarse_lengths
        ),
        fine_tokens=MultiModalFieldConfig.flat_from_sizes("image", fine_lengths),
        fine_time_indices=MultiModalFieldConfig.flat_from_sizes("image", fine_lengths),
        coarse_lengths=MultiModalFieldConfig.batched("image"),
        fine_lengths=MultiModalFieldConfig.batched("image"),
    )


class MiniCPMRobotTrackImageItems(DictEmbeddingItems):
    """One item = the full visual bundle (coarse/fine tokens + time indices)."""

    def __init__(
        self,
        data: Mapping[str, torch.Tensor],
        fields_factory: Any,
    ) -> None:
        super().__init__(
            data,
            modality="image",
            required_fields={
                "coarse_tokens",
                "coarse_time_indices",
                "fine_tokens",
                "fine_time_indices",
            },
            fields_factory=fields_factory,
        )


class MiniCPMRobotTrackPixelItems(ModalityDataItems[list[Any], list[Any]]):
    """One item = a full rolling window of raw frames (pixels-in path).

    The window (oldest first, current last) is a single multimodal item; the
    frames are routed to the HF processor, which resizes/normalizes them for
    DINOv3 and SigLIP. When the client names a ``stream_id`` (plus a monotonic
    ``frame_index``), the server keeps the window state and the client only
    sends the newly-arrived frame each step.
    """

    def __init__(
        self,
        frames: list[Any],
        stream_id: str | None = None,
        frame_index: int | None = None,
    ) -> None:
        super().__init__(frames, "image")
        self.stream_id = stream_id
        self.frame_index = frame_index

    def get_count(self) -> int:
        return 1

    def get(self, index: int) -> Any:
        if index != 0:
            raise IndexError(index)
        # Return the metadata-carrying dict so the framework's cache-miss
        # re-parse (`parse_mm_data` on this item's data) does not drop the
        # stream_id/frame_index that distinguish a stateful request.
        if self.stream_id is None:
            return self.data
        return {
            "frames": self.data,
            "stream_id": self.stream_id,
            "frame_index": self.frame_index,
        }

    def get_processor_data(self) -> Mapping[str, object]:
        data: dict[str, object] = {"frames": self.data}
        if self.stream_id is not None:
            data["stream_id"] = self.stream_id
        if self.frame_index is not None:
            data["frame_index"] = self.frame_index
        return data

    def get_passthrough_data(self) -> Mapping[str, object]:
        return {}


class MiniCPMRobotTrackDataParser(MultiModalDataParser):
    def _parse_image_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[Any],
    ) -> ModalityDataItems[Any, Any] | None:
        # pixels-in: a rolling window of raw frames (list, or {"frames": [...]}).
        # A dict may also carry stream_id/frame_index for the stateful window.
        # The framework's cache-miss re-parse wraps one item's data in a list;
        # unwrap a single frames-dict so its stream metadata is preserved.
        if (
            isinstance(data, (list, tuple))
            and len(data) == 1
            and isinstance(data[0], dict)
            and "frames" in data[0]
        ):
            item = data[0]
            stream_id = item.get("stream_id")
            if stream_id is not None:
                stream_id = _stream_id_key(stream_id)
            return MiniCPMRobotTrackPixelItems(
                list(item["frames"]),
                stream_id=stream_id,
                frame_index=item.get("frame_index"),
            )
        if isinstance(data, (list, tuple)):
            return MiniCPMRobotTrackPixelItems(list(data))
        if isinstance(data, dict) and "frames" in data:
            stream_id = data.get("stream_id")
            if stream_id is not None:
                stream_id = _stream_id_key(stream_id)
            return MiniCPMRobotTrackPixelItems(
                list(data["frames"]),
                stream_id=stream_id,
                frame_index=data.get("frame_index"),
            )
        # features-in: precomputed DINOv3+SigLIP tokens (backward compatible).
        if not isinstance(data, dict):
            raise ValueError(
                "MiniCPM-RobotTrack expects either a window of raw frames "
                "(pixels-in) or precomputed visual features as a dict of "
                "tensors (features-in) under the 'image' modality."
            )
        data = _with_visual_lengths(data)
        return MiniCPMRobotTrackImageItems(
            data,
            fields_factory=_robottrack_field_config,
        )


def _with_visual_lengths(
    data: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Attach per-item token counts so the flat fields can be sliced back."""
    out = dict(data)
    out["coarse_tokens"] = torch.as_tensor(out["coarse_tokens"])
    out["fine_tokens"] = torch.as_tensor(out["fine_tokens"])
    out["coarse_time_indices"] = torch.as_tensor(
        out["coarse_time_indices"], dtype=torch.long
    )
    out["fine_time_indices"] = torch.as_tensor(
        out["fine_time_indices"], dtype=torch.long
    )
    out["coarse_lengths"] = torch.tensor(
        [out["coarse_tokens"].shape[0]], dtype=torch.long
    )
    out["fine_lengths"] = torch.tensor([out["fine_tokens"].shape[0]], dtype=torch.long)
    return out


def _count_marker_runs(time_indices: torch.Tensor) -> int:
    """Number of maximal equal-value runs (one temporal marker per run)."""
    values = time_indices.tolist()
    if not values:
        return 0
    runs = 1
    for prev, cur in zip(values, values[1:]):
        if cur != prev:
            runs += 1
    return runs


class MiniCPMRobotTrackProcessingInfo(BaseProcessingInfo):
    def __init__(self, ctx: Any) -> None:
        super().__init__(ctx)
        # Per-frame normalized-pixel cache (pixels-in path, CPU side). Keyed by
        # the same content hash as the tower feature cache: a rolling window
        # resize/normalizes only frames it has not seen before, so steady-state
        # steps re-process just the new frame. Bound by ``pixel_cache_size``
        # (0 disables reuse, matching the original full-window processing).
        self._pixel_cache: OrderedDict[int, tuple[torch.Tensor, torch.Tensor]] = (
            OrderedDict()
        )
        self._pixel_cache_size = int(
            getattr(self.get_hf_config(), "pixel_cache_size", 0)
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 1}

    def get_data_parser(self) -> MultiModalDataParser:
        return MiniCPMRobotTrackDataParser()

    def get_image_token_id(self) -> int:
        # A reserved in-vocab token id (last vocab slot, a special token that
        # never appears in a natural-language instruction) marks the appended
        # visual bundle; those positions are overwritten by the visual embeds.
        return self.get_hf_config().backbone_config.vocab_size - 1

    def _get_image_processor(self, attr: str, loader: Any, model: str) -> Any:
        cached = getattr(self, attr, None)
        if cached is None:
            cached = loader.from_pretrained(model)
            setattr(self, attr, cached)
        return cached

    def get_dino_processor(self) -> Any:
        return self._get_image_processor(
            "_dino_processor", AutoImageProcessor, self.get_hf_config().dino_model
        )

    def get_siglip_processor(self) -> Any:
        from transformers import SiglipImageProcessor

        return self._get_image_processor(
            "_siglip_processor", SiglipImageProcessor, self.get_hf_config().siglip_model
        )

    @staticmethod
    def _resize_frame(frame: Any, size: int, resample: Any) -> Any:
        from PIL import Image

        return (
            (
                frame
                if isinstance(frame, Image.Image)
                else Image.fromarray(np.asarray(frame))
            )
            .convert("RGB")
            .resize((size, size), resample)
        )

    def prepare_pixels(
        self, frames: Sequence[Any]
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Resize + normalize a window of raw frames for DINOv3 and SigLIP.

        Faithful to upstream ``DualVisionEncoder._prepare``: each frame is
        RGB-converted and BICUBIC-resized to ``image_size`` before the
        encoder-specific processor normalization.

        Per-frame normalized pixels are memoized by content hash (same key as
        the tower feature cache), so a rolling window only resize/normalizes
        frames it has not seen before; unchanged history frames are served from
        the cache. Returns ``(dino, siglip, keys)``; ``keys`` are the per-frame
        content hashes, reused by the caller so frames are hashed exactly once.
        """
        from PIL import Image

        size = self.get_hf_config().image_size
        resample = Image.Resampling.BICUBIC
        keys = [_frame_content_key(frame) for frame in frames]

        def process_misses(
            miss_frames: list[Any],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            images = [
                self._resize_frame(frame, size, resample) for frame in miss_frames
            ]
            size_arg = {"height": size, "width": size}
            miss_dino = self.get_dino_processor()(
                images=images, return_tensors="pt", size=size_arg
            )["pixel_values"]
            miss_siglip = self.get_siglip_processor()(
                images=images, return_tensors="pt", size=size_arg
            )["pixel_values"]
            return miss_dino, miss_siglip

        if self._pixel_cache_size <= 0:
            dino, siglip = process_misses(list(frames))
        else:
            dino, siglip = _pixel_window_cached(
                frames,
                keys,
                self._pixel_cache,
                self._pixel_cache_size,
                process_misses,
            )
        return dino, siglip, keys

    def get_num_image_tokens(
        self,
        coarse_time_indices: torch.Tensor,
        fine_time_indices: torch.Tensor,
    ) -> int:
        coarse = int(coarse_time_indices.shape[0]) + _count_marker_runs(
            coarse_time_indices
        )
        fine = int(fine_time_indices.shape[0]) + _count_marker_runs(fine_time_indices)
        return coarse + fine + 1  # +1 for the control query

    def get_num_pixel_image_tokens(self) -> int:
        # The tower pads history to exactly ``history_frames`` and pools the
        # current frame to ``fine_tokens_current_frame``, so the placeholder
        # count is fixed regardless of how many frames the client sends.
        hf_config = self.get_hf_config()
        coarse_time = torch.arange(hf_config.history_frames).repeat_interleave(
            hf_config.coarse_tokens_per_frame
        )
        fine_time = torch.full(
            (hf_config.fine_tokens_current_frame,),
            hf_config.history_frames,
            dtype=torch.long,
        )
        return self.get_num_image_tokens(coarse_time, fine_time)

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        hf_config = self.get_hf_config()
        # Worst case: every visual token sits in its own frame -> a marker each.
        coarse = hf_config.history_frames * hf_config.coarse_tokens_per_frame
        fine = hf_config.fine_tokens_current_frame
        return {"image": 2 * coarse + 2 * fine + 1}


class MiniCPMRobotTrackDummyInputsBuilder(
    BaseDummyInputsBuilder[MiniCPMRobotTrackProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        if num_images == 0:
            return {}
        # Use the pixels-in window so engine profiling exercises the vision
        # tower (the heavier of the two input paths).
        hf_config = self.info.get_hf_config()
        size = hf_config.image_size
        num_frames = hf_config.history_frames + 1
        # Distinct per-frame fill so the content cache does not collapse the
        # window during profiling; the tower must encode the full batch to size
        # its peak activation memory correctly.
        frames = [
            np.full((size, size, 3), i % 256, dtype=np.uint8) for i in range(num_frames)
        ]
        return {"image": {"frames": frames}}


class MiniCPMRobotTrackMultiModalProcessor(
    BaseMultiModalProcessor[MiniCPMRobotTrackProcessingInfo]
):
    def _get_data_parser(self) -> MultiModalDataParser:
        return self.info.get_data_parser()

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # No HF multimodal processor: tokenize the instruction here, and for the
        # pixels-in path resize/normalize each window into DINOv3/SigLIP-ready
        # pixels. Visual placeholders are appended at the end by
        # `_get_prompt_updates` (HF's [text, history, current, control] order).
        tokenizer = self.info.get_tokenizer()
        input_ids = tokenizer(prompt, add_special_tokens=True).input_ids
        outputs: dict[str, object] = {"input_ids": [input_ids]}

        frames = mm_data.get("frames")
        if isinstance(frames, list) and frames:
            # Each mm item is one window; the framework may hand over a single
            # window or a list of windows. Normalize to per-window groups so each
            # window's frame count is recorded for the flat field slicing.
            windows = _normalize_windows(frames)
            dino_list, siglip_list, lengths = [], [], []
            keys: list[int] = []
            for window in windows:
                dino_pixels, siglip_pixels, window_keys = self.info.prepare_pixels(
                    window
                )
                dino_list.append(dino_pixels)
                siglip_list.append(siglip_pixels)
                lengths.append(len(window))
                keys.extend(window_keys)
            outputs["dino_pixels"] = torch.cat(dino_list)
            outputs["siglip_pixels"] = torch.cat(siglip_list)
            outputs["frame_lengths"] = torch.tensor(lengths, dtype=torch.long)
            outputs["frame_keys"] = torch.tensor(keys, dtype=torch.long)
            # Stateful stream metadata: one value per window, routed to the model
            # as batched fields (not split by the per-frame flat config).
            if "stream_id" in mm_data:
                outputs["stream_id"] = [mm_data["stream_id"]]
            if "frame_index" in mm_data:
                outputs["frame_index"] = [mm_data["frame_index"]]

        return BatchFeature(outputs, tensor_type="pt")

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        # Placeholders are always inserted by `_get_prompt_updates`, never by
        # the text-only HF processor call.
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        if "dino_pixels" in hf_inputs:
            frame_lengths = hf_inputs.get(
                "frame_lengths", torch.empty(0, dtype=torch.long)
            )
            fields = dict(
                dino_pixels=MultiModalFieldConfig.flat_from_sizes(
                    "image", frame_lengths
                ),
                siglip_pixels=MultiModalFieldConfig.flat_from_sizes(
                    "image", frame_lengths
                ),
                frame_keys=MultiModalFieldConfig.flat_from_sizes(
                    "image", frame_lengths
                ),
                frame_lengths=MultiModalFieldConfig.batched("image"),
            )
            # Stateful stream metadata is a batched (per-window) non-tensor value.
            if "stream_id" in hf_inputs:
                fields["stream_id"] = MultiModalFieldConfig.batched("image")
                fields["frame_index"] = MultiModalFieldConfig.batched("image")
            return fields
        return _robottrack_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: Any,
    ) -> Sequence[PromptUpdate]:
        image_items = mm_items.get_items(
            "image",
            (MiniCPMRobotTrackImageItems, MiniCPMRobotTrackPixelItems),
        )
        image_token_id = self.info.get_image_token_id()

        def get_insertion(item_idx: int) -> list[int]:
            if isinstance(image_items, MiniCPMRobotTrackPixelItems):
                num_tokens = self.info.get_num_pixel_image_tokens()
            else:
                item = image_items.get(item_idx)
                num_tokens = self.info.get_num_image_tokens(
                    item["coarse_time_indices"],
                    item["fine_time_indices"],
                )
            return [image_token_id] * num_tokens

        return [
            PromptInsertion(
                modality="image",
                target=PromptIndexTargets.end(),
                insertion=get_insertion,
            )
        ]


@default_pooling_type(seq_pooling_type="LAST")
@MULTIMODAL_REGISTRY.register_processor(
    MiniCPMRobotTrackMultiModalProcessor,
    info=MiniCPMRobotTrackProcessingInfo,
    dummy_inputs=MiniCPMRobotTrackDummyInputsBuilder,
)
class MiniCPMRobotTrackModel(nn.Module, SupportsMultiModal):
    is_pooling_model = True

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"backbone.": "model."})

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return None
        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        backbone_config = config.backbone_config
        hidden_dim = backbone_config.hidden_size

        with self._mark_language_model(vllm_config):
            self.model = MiniCPMModel(
                vllm_config=vllm_config.with_hf_config(backbone_config),
                prefix=maybe_prefix(prefix, "model"),
            )

        # The vision tower (pixels-in path) loads its own DINOv3/SigLIP weights
        # from external encoder dirs; they are absent from the RobotTrack
        # checkpoint, so `load_weights` skips it and reports its params as loaded.
        with self._mark_tower_model(vllm_config, "image"):
            self.vision_tower = DualVisionTower(config.dino_model, config.siglip_model)
        self.vision_tower.to(
            device=vllm_config.device_config.device,
            dtype=vllm_config.model_config.dtype,
        )
        # Per-frame coarse-feature cache (pixels-in path): the tower encodes each
        # unique frame once and reuses its coarse pool while it stays in the
        # sliding window, so a rolling request re-encodes only the new frame each
        # step. The current frame's fine pool is never cached.
        self._frame_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._frame_cache_size = int(getattr(config, "frame_cache_size", 0))

        # Stateful stream windows (pixels-in path): each stream holds its rolling
        # 31-frame coarse history server-side, so a client only sends the new
        # frame per step (with ``stream_id`` / ``frame_index``). Bounded LRU.
        self._stream_states: OrderedDict[str, RobotTrackStreamState] = OrderedDict()
        self._max_cached_streams = int(getattr(config, "max_cached_streams", 8))

        self.vision_projector = VisionProjector(config.vision_feature_dim, hidden_dim)
        self.temporal_markers = TemporalMarkerEncoder(hidden_dim, config.max_time_steps)
        self.control_query = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.trajectory_head = FunnelTrajectoryHead(
            hidden_dim=hidden_dim,
            num_waypoints=config.num_waypoints,
            action_dim=config.action_dim,
            dropout=config.trajectory_dropout,
            use_tanh=config.use_tanh_actions,
        )

        # Recomputed rather than loaded so it never depends on stored dtype.
        output_scale = torch.ones(1, 1, config.action_dim)
        output_scale[..., :2] = config.xy_scale
        self.register_buffer("output_scale", output_scale, persistent=False)

        self.pooler = DispatchPooler(
            {"embed": SequencePooler(pooling=LastPool(), head=self._pool_trajectory)}
        )

    def _pool_trajectory(
        self,
        pooled_data: torch.Tensor | list[torch.Tensor],
        pooling_metadata: PoolingMetadata,
    ) -> torch.Tensor:
        if isinstance(pooled_data, list):
            pooled_data = torch.stack(pooled_data)
        head_dtype = self.trajectory_head.layers[-1].weight.dtype
        trajectory = self.trajectory_head(pooled_data.to(head_dtype))
        trajectory = trajectory * self.output_scale.to(trajectory.dtype)
        return trajectory.flatten(1)

    def _embed_text_input_ids(
        self,
        input_ids: torch.Tensor,
        embed_input_ids: Any,
        *,
        is_multimodal: torch.Tensor | None,
    ) -> torch.Tensor:
        # HF RobotTrack feeds RAW token embeddings (no scale_emb) as
        # inputs_embeds, so bypass MiniCPMModel.embed_input_ids (which multiplies
        # by scale_emb) and use the plain embedding table instead.
        return super()._embed_text_input_ids(
            input_ids,
            self.model.embed_tokens,
            is_multimodal=is_multimodal,
        )

    def _insert_temporal_markers(
        self,
        tokens: torch.Tensor,
        time_indices: torch.Tensor,
        stream_id: int,
    ) -> torch.Tensor:
        if tokens.shape[0] == 0:
            return tokens
        device = tokens.device
        time_row = time_indices.tolist()
        pieces: list[torch.Tensor] = []
        start = 0
        while start < len(time_row):
            time_step = int(time_row[start])
            end = start + 1
            while end < len(time_row) and int(time_row[end]) == time_step:
                end += 1
            marker = self.temporal_markers(time_step, stream_id, device)
            pieces.append(marker.unsqueeze(0).to(tokens.dtype))
            pieces.append(tokens[start:end])
            start = end
        return torch.cat(pieces, dim=0)

    def _embed_visual_bundle(
        self,
        coarse_tokens: torch.Tensor,
        coarse_time_indices: torch.Tensor,
        fine_tokens: torch.Tensor,
        fine_time_indices: torch.Tensor,
    ) -> torch.Tensor:
        device = self.control_query.device
        proj_dtype = self.vision_projector.layers[1].weight.dtype
        coarse_tokens = coarse_tokens.to(device=device, dtype=proj_dtype)
        fine_tokens = fine_tokens.to(device=device, dtype=proj_dtype)
        history = self.vision_projector(coarse_tokens)
        current = self.vision_projector(fine_tokens)
        history = self._insert_temporal_markers(
            history, coarse_time_indices.to(device), stream_id=0
        )
        current = self._insert_temporal_markers(
            current, fine_time_indices.to(device), stream_id=1
        )
        control_query = self.control_query.reshape(1, -1).to(history.dtype)
        sequence = torch.cat((history, current, control_query), dim=0)
        return sequence.to(self.model.embed_tokens.weight.dtype)

    def _encode_window(
        self,
        dino_pixels: torch.Tensor,
        siglip_pixels: torch.Tensor,
        frame_keys: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode one raw-frame window into (coarse, coarse_time, fine, fine_time).

        The last frame is the current frame (fine pool); earlier frames are
        history (coarse pool), padded to ``history_frames``. Per-frame features
        are served from ``self._frame_cache`` so unchanged history frames are not
        re-encoded across the sliding window.
        """
        cfg = self.config
        coarse_by_frame, fine, num_encoded = _encode_frames_cached(
            self.vision_tower,
            self._frame_cache,
            self._frame_cache_size,
            dino_pixels,
            siglip_pixels,
            list(frame_keys),
            cfg.coarse_tokens_per_frame,
            cfg.fine_tokens_current_frame,
        )
        logger.debug(
            "RobotTrack tower: encoded %d/%d frames (cache=%d/%d)",
            num_encoded,
            len(frame_keys),
            len(self._frame_cache),
            self._frame_cache_size,
        )

        if len(coarse_by_frame) > 1:
            history = torch.stack(coarse_by_frame[:-1], dim=0)
        else:
            history = coarse_by_frame[-1].unsqueeze(0)
        return _assemble_window_tensors(
            history,
            fine,
            cfg.history_frames,
            cfg.coarse_tokens_per_frame,
        )

    def _commit_stream(self, stream_id: str, state: RobotTrackStreamState) -> None:
        self._stream_states[stream_id] = state
        self._stream_states.move_to_end(stream_id)
        while len(self._stream_states) > self._max_cached_streams:
            self._stream_states.popitem(last=False)

    def _encode_stream_window(
        self,
        dino_pixels: torch.Tensor,
        siglip_pixels: torch.Tensor,
        frame_keys: Sequence[int],
        stream_id: str,
        frame_index: int | None,
    ) -> torch.Tensor:
        """Encode one stream update and roll the server-side window forward.

        ``"replace"`` (a full window) establishes/re-syncs the stream; ``"append"``
        (one consecutive frame) encodes it and rolls the history; ``"reuse"``
        (the same ``frame_index``) is an idempotent retry served from the
        committed state without re-encoding. Returns the visual bundle for the
        assembled window.
        """
        cfg = self.config
        state = self._stream_states.get(stream_id)
        mode = _classify_stream_request(
            dino_pixels.shape[0], cfg.history_frames, state, frame_index
        )

        if mode == "reuse":
            next_state = state
        else:
            coarse_by_frame, fine, _ = _encode_frames_cached(
                self.vision_tower,
                self._frame_cache,
                self._frame_cache_size,
                dino_pixels,
                siglip_pixels,
                list(frame_keys),
                cfg.coarse_tokens_per_frame,
                cfg.fine_tokens_current_frame,
            )
            next_state = _advance_stream_state(
                state,
                mode,
                coarse_by_frame,
                fine,
                frame_index,
                cfg.history_frames,
            )
            self._commit_stream(stream_id, next_state)

        coarse, coarse_time, fine, fine_time = _assemble_stream_window(
            next_state,
            cfg.history_frames,
            cfg.coarse_tokens_per_frame,
        )
        return self._embed_visual_bundle(coarse, coarse_time, fine, fine_time)

    def _embed_pixel_windows(
        self,
        dino_pixels: object,
        siglip_pixels: object,
        frame_lengths: object,
        frame_keys: object,
        stream_ids: object | None = None,
        frame_indices: object | None = None,
    ) -> MultiModalEmbeddings:
        """Embed raw-frame windows, optionally advancing server-side streams.

        With ``stream_ids`` (plus per-window ``frame_indices``) each window is a
        stateful stream update (one new frame per step); otherwise windows are
        encoded statelessly. Both paths share the same split pipeline and only
        differ in the per-window encode call.
        """
        dino_pixels = _as_4d(dino_pixels)
        siglip_pixels = _as_4d(siglip_pixels)
        lengths = _as_length_list(frame_lengths)
        keys = _as_length_list(frame_keys)
        dino_split = torch.split(dino_pixels, lengths)
        siglip_split = torch.split(siglip_pixels, lengths)
        key_split = _split_by_lengths(keys, lengths)

        if stream_ids is None:
            return [
                self._embed_visual_bundle(*self._encode_window(dp, sp, wk))
                for dp, sp, wk in zip(dino_split, siglip_split, key_split)
            ]

        ids = [_scalar(s) for s in stream_ids]
        indices = (
            [_scalar(i) for i in frame_indices]
            if frame_indices is not None
            else [None] * len(ids)
        )
        return [
            self._encode_stream_window(dp, sp, wk, sid, fidx)
            for dp, sp, wk, sid, fidx in zip(
                dino_split, siglip_split, key_split, ids, indices
            )
        ]

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        # pixels-in: raw-frame windows are encoded by the in-tree tower. With a
        # ``stream_id`` the request advances a server-side window (one frame per
        # step); otherwise it is a stateless full-window request.
        dino_pixels = kwargs.get("dino_pixels")
        if dino_pixels is not None:
            return self._embed_pixel_windows(
                dino_pixels,
                kwargs["siglip_pixels"],
                kwargs["frame_lengths"],
                kwargs["frame_keys"],
                kwargs.get("stream_id"),
                kwargs.get("frame_index"),
            )

        # features-in: precomputed DINOv3+SigLIP tokens (backward compatible).
        coarse_tokens = kwargs.get("coarse_tokens")
        if coarse_tokens is None:
            return []
        coarse_time_indices = kwargs["coarse_time_indices"]
        fine_tokens = kwargs["fine_tokens"]
        fine_time_indices = kwargs["fine_time_indices"]
        coarse_lengths = _as_length_list(kwargs["coarse_lengths"])
        fine_lengths = _as_length_list(kwargs["fine_lengths"])

        coarse_tokens = _as_2d(coarse_tokens)
        fine_tokens = _as_2d(fine_tokens)
        coarse_time_indices = _as_1d(coarse_time_indices)
        fine_time_indices = _as_1d(fine_time_indices)

        coarse_tok = torch.split(coarse_tokens, coarse_lengths)
        coarse_time = torch.split(coarse_time_indices, coarse_lengths)
        fine_tok = torch.split(fine_tokens, fine_lengths)
        fine_time = torch.split(fine_time_indices, fine_lengths)

        return [
            self._embed_visual_bundle(ct, cti, ft, fti)
            for ct, cti, ft, fti in zip(coarse_tok, coarse_time, fine_tok, fine_time)
        ]

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = (
            (name, weight)
            for name, weight in weights
            if not name.startswith("output_scale")
        )
        # The vision tower self-loads in __init__; skip it here but report its
        # params as loaded so the missing-weights check passes.
        loader = AutoWeightsLoader(self, skip_prefixes=["vision_tower."])
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        loaded |= {
            name
            for name, _ in self.vision_tower.named_parameters(prefix="vision_tower")
        }
        return loaded


def _scalar(value: object) -> object:
    # Batched stream fields arrive as 0-dim CUDA tensors after the H2D
    # transfer; dict-key matching needs a plain int (tensor hashing is by
    # identity, so a fresh tensor never equals the stored key).
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return value


def _as_4d(tensor: object) -> torch.Tensor:
    if isinstance(tensor, (list, tuple)):
        tensor = torch.cat([_as_4d(t) for t in tensor], dim=0)
    assert isinstance(tensor, torch.Tensor)
    return tensor.reshape(-1, *tensor.shape[-3:])


def _as_2d(tensor: object) -> torch.Tensor:
    if isinstance(tensor, (list, tuple)):
        tensor = torch.cat([_as_2d(t) for t in tensor], dim=0)
    assert isinstance(tensor, torch.Tensor)
    return tensor.reshape(-1, tensor.shape[-1])


def _as_1d(tensor: object) -> torch.Tensor:
    if isinstance(tensor, (list, tuple)):
        tensor = torch.cat([_as_1d(t) for t in tensor], dim=0)
    assert isinstance(tensor, torch.Tensor)
    return tensor.reshape(-1)


def _as_length_list(lengths: object) -> list[int]:
    if isinstance(lengths, (list, tuple)):
        flat: list[int] = []
        for item in lengths:
            flat.extend(_as_length_list(item))
        return flat
    assert isinstance(lengths, torch.Tensor)
    return [int(x) for x in lengths.reshape(-1).tolist()]


def _split_by_lengths(values: list[int], lengths: list[int]) -> list[list[int]]:
    out: list[list[int]] = []
    start = 0
    for length in lengths:
        out.append(values[start : start + length])
        start += length
    return out
