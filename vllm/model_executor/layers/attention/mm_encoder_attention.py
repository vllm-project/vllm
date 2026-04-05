# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import json
from pathlib import Path

import numpy as np
import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp, maybe_get_oot_by_class
from vllm.model_executor.layers.quantization.input_quant_fp8 import (
    QuantFP8,
    quantize_fp8_pad_head_dim_triton,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    get_fp8_min_max,
)
from vllm.model_executor.models.vision import get_vit_attn_backend
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    is_flashinfer_cudnn_fp8_prefill_attn_supported,
)
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.ops.vit_attn_wrappers import (
    vit_flash_attn_wrapper,
    vit_flashinfer_wrapper,
    vit_torch_sdpa_wrapper,
    vit_triton_attn_wrapper,
)

logger = init_logger(__name__)

_, _FP8_MAX = get_fp8_min_max()
_FP8_AMAX_HISTORY_LEN = 16


@functools.cache
def _load_fp8_scales_file(path: str | None) -> dict[str, dict[str, float]]:
    """Load per-layer FP8 Q/K/V scales from a JSON file. Results are cached.

    Expected format example (keys like ``q_scale`` also accepted)::

        {
            "visual.blocks.0.attn": {"q": 224.0, "k": 198.0, "v": 210.0},
            "visual.blocks.1.attn": {"q": 218.0, "k": 195.0, "v": 207.0},
            ...
        }

    To generate a scale file, enable dynamic scaling (enabled by default when
    no scale file is provided) and run some forward passes. Then dump::

        scales = {}
        for name, module in model.named_modules():
            if hasattr(module, "_fp8_q_scale"):
                scales[name] = {
                    "q": module._fp8_q_scale.item(),
                    "k": module._fp8_k_scale.item(),
                    "v": module._fp8_v_scale.item(),
                }
        with open("fp8_vit_scales.json", "w") as f:
            json.dump(scales, f, indent=2)
    """
    if path is None:
        return {}

    path = str(Path(path).expanduser())
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Handle nested "layers" format
    if "layers" in data and isinstance(data["layers"], dict):
        data = data["layers"]

    scales: dict[str, dict[str, float]] = {}
    for layer_name, layer_scales in data.items():
        if not isinstance(layer_scales, dict):
            continue
        q = layer_scales.get("q", layer_scales.get("q_scale"))
        k = layer_scales.get("k", layer_scales.get("k_scale"))
        v = layer_scales.get("v", layer_scales.get("v_scale"))
        if q is not None and k is not None and v is not None:
            q_f, k_f, v_f = float(q), float(k), float(v)
            if q_f <= 0 or k_f <= 0 or v_f <= 0:
                raise ValueError(
                    f"FP8 scales must be positive, got q={q_f}, "
                    f"k={k_f}, v={v_f} for layer '{layer_name}'"
                )
            scales[layer_name] = {"q": q_f, "k": k_f, "v": v_f}

    logger.info_once(
        "Loaded FP8 attention scales from %s (%d layers)", path, len(scales)
    )
    return scales


def _get_mm_encoder_fp8_config() -> tuple[str | None, str | None]:
    """Read FP8 encoder config from multimodal_config, if available.

    Returns (mm_encoder_attn_dtype, mm_encoder_fp8_scale_path).
    """
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None or vllm_config.model_config is None:
        return None, None
    mm_config = vllm_config.model_config.multimodal_config
    if mm_config is None:
        return None, None
    return mm_config.mm_encoder_attn_dtype, mm_config.mm_encoder_fp8_scale_path


# Batch buckets for cuDNN graph caching.
# Graphs use batch size and max sequence length as cache key.
# This avoids creating a new graph for each unique set of
# batch size and max sequence length at runtime.
# From the cuDNN team's performance measurements, there
# is no significant kernel performance difference between padding
# to a smaller batch size/seq length and padding to larger
# ones. The bucketing here is solely used to avoid memory
# operation overhead, which won't be needed if we have CUDA
# graph support in the future.
# TODO: Remove buckets after issue #34763
# (cuda graph support) is addressed.
FLASHINFER_BATCH_BUCKETS = [8, 16, 32, 64]
FLASHINFER_MAX_SEQLEN_BUCKETS = [
    1 * 1024,
    2 * 1024,
    4 * 1024,
    8 * 1024,
    16 * 1024,
    32 * 1024,
    64 * 1024,
    128 * 1024,
]

# Workspace buffer for FlashInfer CuDNN backend
FLASHINFER_CUDNN_WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024
_flashinfer_workspace_buffer: torch.Tensor | None = None


def _get_flashinfer_workspace_buffer() -> torch.Tensor:
    global _flashinfer_workspace_buffer
    if _flashinfer_workspace_buffer is None:
        _flashinfer_workspace_buffer = torch.zeros(
            FLASHINFER_CUDNN_WORKSPACE_SIZE_BYTES,
            dtype=torch.uint8,
            device="cuda",
        )
    return _flashinfer_workspace_buffer


def add_padding_to_seqlens(
    seq: np.ndarray,
    batch_size: int,
    padding_value: int,
) -> np.ndarray:
    batch_size_padded = next(
        (b for b in FLASHINFER_BATCH_BUCKETS if b >= batch_size),
        round_up(batch_size, FLASHINFER_BATCH_BUCKETS[0]),
    )
    if batch_size_padded == batch_size:
        return seq
    return np.concatenate(
        [
            seq,
            np.full((batch_size_padded - batch_size,), padding_value, dtype=seq.dtype),
        ]
    )


def bucket_flashinfer_max_seqlen(
    real_max_seqlen: int,
) -> int:
    if real_max_seqlen <= 0:
        return FLASHINFER_MAX_SEQLEN_BUCKETS[0]
    return next(
        (s for s in FLASHINFER_MAX_SEQLEN_BUCKETS if s >= real_max_seqlen),
        round_up(real_max_seqlen, FLASHINFER_MAX_SEQLEN_BUCKETS[-1]),
    )


# --8<-- [start:mm_encoder_attn]
@CustomOp.register("mm_encoder_attn")
class MMEncoderAttention(CustomOp):
    """Multi-headed attention without any cache, used for multimodal encoder."""

    # --8<-- [end:mm_encoder_attn]
    @classmethod
    def compute_max_seqlen(
        cls,
        attn_backend: AttentionBackendEnum,
        cu_seqlens: np.ndarray,
    ) -> int:
        max_seqlen = 0
        if (
            attn_backend
            in (
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.ROCM_AITER_FA,
                AttentionBackendEnum.TRITON_ATTN,
                AttentionBackendEnum.FLASHINFER,
            )
            and len(cu_seqlens) >= 2
        ):
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
        if attn_backend == AttentionBackendEnum.FLASHINFER:
            max_seqlen = bucket_flashinfer_max_seqlen(max_seqlen)
        return max_seqlen

    @classmethod
    def maybe_compute_seq_lens(
        cls,
        attn_backend: AttentionBackendEnum,
        cu_seqlens: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor | None:
        if (oot_class := maybe_get_oot_by_class(cls)) is not cls:
            return oot_class.maybe_compute_seq_lens(attn_backend, cu_seqlens, device)  # type: ignore[attr-defined]

        if attn_backend != AttentionBackendEnum.FLASHINFER:
            return None

        sequence_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        sequence_lengths = add_padding_to_seqlens(
            sequence_lengths, len(sequence_lengths), 0
        )
        sequence_lengths = torch.from_numpy(sequence_lengths).to(
            device, non_blocking=True
        )
        return sequence_lengths

    @classmethod
    def maybe_recompute_cu_seqlens(
        cls,
        attn_backend: AttentionBackendEnum,
        cu_seqlens: np.ndarray,
        hidden_size: int,
        tp_size: int,
        device: torch.device,
        fp8_padded_hidden_size: int | None = None,
    ) -> torch.Tensor:
        if (oot_class := maybe_get_oot_by_class(cls)) is not cls:
            return oot_class.maybe_recompute_cu_seqlens(  # type: ignore[attr-defined]
                attn_backend,
                cu_seqlens,
                hidden_size,
                tp_size,
                device,
                fp8_padded_hidden_size=fp8_padded_hidden_size,
            )

        if attn_backend == AttentionBackendEnum.FLASHINFER:
            batch_size = len(cu_seqlens) - 1

            if fp8_padded_hidden_size is not None:
                # FP8 path: after quantization Q/K/V are each independent
                # contiguous tensors with stride H * padded_D per token.
                # All sections use the same element stride.
                scale = fp8_padded_hidden_size // tp_size
                cu_seqlens = cu_seqlens * scale
                cu_seqlens_padded = add_padding_to_seqlens(
                    cu_seqlens, batch_size, cu_seqlens[-1]
                )
                cu_seqlens = np.concatenate([cu_seqlens_padded, cu_seqlens_padded])
            else:
                # BF16 path: Q/K/V are non-contiguous views into shared
                # buffers. V section has 3x stride from interleaved QKV.
                scale = hidden_size // tp_size
                cu_seqlens = cu_seqlens * scale

                cu_seqlens_qko = cu_seqlens
                cu_seqlens_v = cu_seqlens * 3

                cu_seqlens_qko = add_padding_to_seqlens(
                    cu_seqlens_qko, batch_size, cu_seqlens_qko[-1]
                )
                cu_seqlens_v = add_padding_to_seqlens(
                    cu_seqlens_v, batch_size, cu_seqlens_v[-1]
                )
                cu_seqlens = np.concatenate([cu_seqlens_qko, cu_seqlens_v])

        cu_seqlens = torch.from_numpy(cu_seqlens).to(device, non_blocking=True)
        return cu_seqlens

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float | None = None,
        num_kv_heads: int | None = None,
        prefix: str = "",
    ) -> None:
        """
        Args:
            num_heads: number of attention heads per partition.
            head_size: hidden_size per attention head.
            scale: scale factor.
            num_kv_heads: number of kv heads.
            prefix: This has no effect, it is only here to make it easier to
                    swap between Attention and MultiHeadAttention
        """
        super().__init__()

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = 1.0 / (head_size**0.5) if scale is None else scale
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.layer_name = prefix
        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({self.num_heads}) is not "
            f"divisible by num_kv_heads ({self.num_kv_heads})"
        )
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        # During model initialization, the default dtype is set as the model
        # weight and activation dtype.
        dtype = torch.get_default_dtype()
        self.dtype = dtype

        # Get device-specific vision attention backend.
        self.attn_backend = get_vit_attn_backend(
            head_size=head_size,
            dtype=dtype,
        )

        self.is_flash_attn_backend = self.attn_backend in {
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.ROCM_AITER_FA,
        }

        self._fa_version = (
            get_flash_attn_version(head_size=head_size)
            if self.is_flash_attn_backend
            else None
        )

        if self.attn_backend == AttentionBackendEnum.FLASHINFER:
            _get_flashinfer_workspace_buffer()

        logger.info_once(
            f"Using {self.attn_backend} for MMEncoderAttention.", scope="local"
        )

        # FP8 attention support (currently only FlashInfer cuDNN backend)
        self.fp8_enabled = False
        self._fp8_dynamic_scale = False
        self.fp8_quant: QuantFP8 | None = None

        attn_dtype, fp8_scale_path = _get_mm_encoder_fp8_config()
        if attn_dtype == "fp8":
            if not is_flashinfer_cudnn_fp8_prefill_attn_supported():
                raise ValueError(
                    "mm_encoder_attn_dtype='fp8' requires the FlashInfer "
                    "cuDNN backend with cuDNN >= 9.17.1. "
                    "Try upgrading cuDNN (nvidia-cudnn-cu1x) via pip, "
                    "e.g.: pip install -U nvidia-cudnn-cu13==9.18.1"
                )
            self._init_fp8_attention(prefix, fp8_scale_path)

    def _init_fp8_attention(self, layer_name: str, scale_path: str | None) -> None:
        """Initialize FP8 quantization state for this layer.

        If *scale_path* is provided, static per-layer scales are loaded from
        the JSON file. Otherwise, dynamic scaling is used: a circular buffer
        of the last 16 observed Q/K/V amax values is maintained and scales
        are recomputed each forward pass.
        """
        all_scales = _load_fp8_scales_file(scale_path)

        if scale_path is not None:
            # Static scaling from calibration file.
            layer_scales = all_scales.get(layer_name)
            if layer_scales is None:
                raise ValueError(
                    "FP8 attention enabled but scales not found for layer "
                    f"'{layer_name}' in {scale_path}. "
                    f"Available layers: {list(all_scales.keys())}"
                )
            init_scales = layer_scales
        else:
            # Dynamic scaling (auto when no scale file provided).
            init_scales = {"q": 1.0, "k": 1.0, "v": 1.0}
            self._fp8_dynamic_scale = True
            logger.info_once(
                "FP8 attention enabled with dynamic scaling "
                "(no scale file provided). Scales will adapt from "
                "observed Q/K/V amax values (history_len=%d).",
                _FP8_AMAX_HISTORY_LEN,
            )

        # Shape (1, 1, 1, 1) as required by cuDNN.
        for attr, key in (
            ("_fp8_q_scale", "q"),
            ("_fp8_k_scale", "k"),
            ("_fp8_v_scale", "v"),
        ):
            self.register_buffer(
                attr,
                torch.tensor([init_scales[key]], dtype=torch.float32).view(1, 1, 1, 1),
            )

        # Circular amax history buffers for dynamic scaling.
        if self._fp8_dynamic_scale:
            for attr in ("_fp8_q_amax", "_fp8_k_amax", "_fp8_v_amax"):
                self.register_buffer(
                    attr,
                    torch.zeros(_FP8_AMAX_HISTORY_LEN, dtype=torch.float32),
                    persistent=False,
                )
            self._fp8_amax_pos = 0

        self.fp8_quant = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR)
        self.fp8_enabled = True

        self.skip_scale_q = not self._fp8_dynamic_scale and init_scales["q"] == 1.0
        self.skip_scale_k = not self._fp8_dynamic_scale and init_scales["k"] == 1.0
        self.skip_scale_v = not self._fp8_dynamic_scale and init_scales["v"] == 1.0

        logger.debug(
            "FP8 attention enabled for %s: q=%.4f, k=%.4f, v=%.4f%s",
            layer_name if layer_name else "MMEncoderAttention",
            init_scales["q"],
            init_scales["k"],
            init_scales["v"],
            " (dynamic scaling)" if self._fp8_dynamic_scale else "",
        )

    @classmethod
    def enabled(cls) -> bool:
        return True

    def view_qkv_to_4d(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bsz: int,
        q_len: int,
        kv_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reshape query, key, value to 4D tensors:
        (batch_size, seq_len, num_heads, head_size)
        """
        query = query.view(bsz, q_len, self.num_heads, self.head_size)
        key = key.view(bsz, kv_len, self.num_kv_heads, self.head_size)
        value = value.view(bsz, kv_len, self.num_kv_heads, self.head_size)

        return query, key, value

    def _forward_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Input shape:
        (batch_size x seq_len x hidden_size) or
        (batch_size x seq_len x num_heads x head_size)
        """
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() != 4

        query, key, value = self.view_qkv_to_4d(query, key, value, bsz, q_len, kv_len)

        output = vit_torch_sdpa_wrapper(
            q=query,
            k=key,
            v=value,
            scale=self.scale,
            cu_seqlens=cu_seqlens,
            enable_gqa=self.num_heads > self.num_kv_heads,
        )
        if is_reshaped:
            output = output.reshape(bsz, q_len, -1)
        return output

    def _forward_fa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
    ) -> torch.Tensor:
        """Input shape:
        (batch_size x seq_len x hidden_size) or
        (batch_size x seq_len x num_heads x head_size)
        """
        assert (cu_seqlens is not None and max_seqlen is not None) or (
            cu_seqlens is None and max_seqlen is None
        ), "cu_seqlens and max_seqlen should be both set or both None."

        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() != 4

        query, key, value = self.view_qkv_to_4d(query, key, value, bsz, q_len, kv_len)

        output = vit_flash_attn_wrapper(
            q=query,
            k=key,
            v=value,
            batch_size=bsz,
            is_rocm_aiter=(self.attn_backend == AttentionBackendEnum.ROCM_AITER_FA),
            fa_version=self._fa_version,
            scale=self.scale,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        if is_reshaped:
            output = output.reshape(bsz, q_len, -1)
        return output

    def _forward_triton(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
    ) -> torch.Tensor:
        """Input shape:
        (batch_size x seq_len x hidden_size) or
        (batch_size x seq_len x num_heads x head_size)
        """
        assert (cu_seqlens is not None and max_seqlen is not None) or (
            cu_seqlens is None and max_seqlen is None
        ), "cu_seqlens and max_seqlen should be both set or both None."

        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() != 4

        query, key, value = self.view_qkv_to_4d(query, key, value, bsz, q_len, kv_len)

        output = vit_triton_attn_wrapper(
            q=query,
            k=key,
            v=value,
            batch_size=bsz,
            scale=self.scale,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        if is_reshaped:
            output = output.reshape(bsz, q_len, -1)
        return output

    def _quantize_to_fp8(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        skip_scale: bool = False,
    ) -> torch.Tensor:
        """Quantize a 3D/4D tensor to FP8.

        Accepts (S, H, D) or (B, S, H, D) input. Uses QuantFP8 CustomOp
        when head_dim is aligned to 16; otherwise falls back to a
        stride-aware Triton kernel that pads head_dim to a multiple of 16.
        """
        assert self.fp8_quant is not None
        orig_shape = tensor.shape
        head_dim = orig_shape[-1]

        if head_dim % 16 == 0:
            if skip_scale:
                return tensor.to(current_platform.fp8_dtype())

            # QuantFP8 expects 2D: flatten all dims except (H, D).
            total_tokens = tensor.numel() // (tensor.shape[-1] * tensor.shape[-2])
            tensor_2d = tensor.reshape(total_tokens, -1)
            fp8_tensor, _ = self.fp8_quant(tensor_2d, scale=scale)
            return fp8_tensor.reshape(orig_shape)

        # Triton kernel pads head_dim to a multiple of 16
        return quantize_fp8_pad_head_dim_triton(tensor, scale, skip_scale=skip_scale)

    @torch.no_grad()
    def _record_amax_and_update_scales(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Record Q/K/V amax into circular history and recompute scales.

        All work stays on GPU with no device-to-host sync. The Python-side
        history position counter is mutated, so this method must NOT be
        called inside CUDA graph capture/replay. When CUDA graphs are
        used for the encoder, dynamic scaling should be disabled by
        providing a static scale file via --mm-encoder-fp8-scale-path.
        """
        pos = self._fp8_amax_pos
        self._fp8_amax_pos = (pos + 1) % _FP8_AMAX_HISTORY_LEN

        for tensor, amax_buf, scale_buf in (
            (query, self._fp8_q_amax, self._fp8_q_scale),
            (key, self._fp8_k_amax, self._fp8_k_scale),
            (value, self._fp8_v_amax, self._fp8_v_scale),
        ):
            amax_buf[pos] = tensor.amax()
            max_amax = amax_buf.max()
            scale_buf.fill_(
                torch.clamp(max_amax, min=torch.finfo(torch.float32).tiny) / _FP8_MAX
            )

    def _forward_flashinfer(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor
        | None = None,  # Only used for FlashInfer CuDNN backend
    ) -> torch.Tensor:
        if self.fp8_enabled:
            assert self.fp8_quant is not None

            if self._fp8_dynamic_scale:
                self._record_amax_and_update_scales(query, key, value)

            query = self._quantize_to_fp8(
                query, self._fp8_q_scale, skip_scale=self.skip_scale_q
            )
            key = self._quantize_to_fp8(
                key, self._fp8_k_scale, skip_scale=self.skip_scale_k
            )
            value = self._quantize_to_fp8(
                value, self._fp8_v_scale, skip_scale=self.skip_scale_v
            )

        output = vit_flashinfer_wrapper(
            q=query,
            k=key,
            v=value,
            scale=self.scale,
            workspace_buffer=_get_flashinfer_workspace_buffer(),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
            q_scale=self._fp8_q_scale if self.fp8_enabled else None,
            k_scale=self._fp8_k_scale if self.fp8_enabled else None,
            v_scale=self._fp8_v_scale if self.fp8_enabled else None,
            o_data_type=self.dtype if self.fp8_enabled else None,
        )

        if self.fp8_enabled and output.shape[-1] != self.head_size:
            output = output[..., : self.head_size].contiguous()

        return output

    def forward_native(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
        sequence_lengths: torch.Tensor
        | None = None,  # Only used for FlashInfer CuDNN backend
    ) -> torch.Tensor:
        return self._forward_sdpa(query, key, value, cu_seqlens)

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
        sequence_lengths: torch.Tensor
        | None = None,  # Only used for FlashInfer CuDNN backend
    ) -> torch.Tensor:
        if self.is_flash_attn_backend:
            return self._forward_fa(query, key, value, cu_seqlens, max_seqlen)
        elif self.attn_backend == AttentionBackendEnum.TRITON_ATTN:
            return self._forward_triton(query, key, value, cu_seqlens, max_seqlen)
        elif self.attn_backend == AttentionBackendEnum.FLASHINFER:
            return self._forward_flashinfer(
                query, key, value, cu_seqlens, max_seqlen, sequence_lengths
            )
        elif self.attn_backend == AttentionBackendEnum.TORCH_SDPA:
            return self._forward_sdpa(query, key, value, cu_seqlens)
        else:
            raise ValueError(
                f"Unsupported multi-modal encoder attention backend for CUDA: "
                f"{self.attn_backend}."
            )

    def forward_cpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
        sequence_lengths: torch.Tensor
        | None = None,  # Only used for FlashInfer CuDNN backend
    ) -> torch.Tensor:
        return self._forward_sdpa(query, key, value, cu_seqlens)

    def forward_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
        sequence_lengths: torch.Tensor
        | None = None,  # Only used for FlashInfer CuDNN backend
    ) -> torch.Tensor:
        if self.attn_backend == AttentionBackendEnum.FLASH_ATTN:
            return self._forward_fa(query, key, value, cu_seqlens, max_seqlen)
        elif self.attn_backend == AttentionBackendEnum.TRITON_ATTN:
            return self._forward_triton(query, key, value, cu_seqlens, max_seqlen)
        elif self.attn_backend == AttentionBackendEnum.TORCH_SDPA:
            return self._forward_sdpa(query, key, value, cu_seqlens)
        else:
            raise ValueError(
                f"Unsupported multi-modal encoder attention backend for XPU: "
                f"{self.attn_backend}."
            )
