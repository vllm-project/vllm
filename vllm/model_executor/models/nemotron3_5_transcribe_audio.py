# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Audio tower of Nemotron-3.5-Transcribe.

A native port of the inference path of NeMo's ``AudioPerceptionModule`` with a
``ParallelExpertEncoder``:

    waveform -> log-mel -> Parallel Expert Encoder -> Linear -> decoder width

The Parallel Expert Encoder runs an ASR ``TransformerEncoder`` and a streaming
Sortformer speaker encoder over the same mel features. Both run in windows of
``window_frames`` encoder frames with left and right context, and the speaker
encoder carries a speaker cache and a FIFO between windows. Thresholded speaker
activity is added to the normalized ASR states through a fixed sinusoidal
kernel. Module names follow NeMo so the checkpoint loads by prefix.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils.torch_utils import async_tensor_h2d

from .utils import maybe_prefix

# Checkpoint tensors that the inference path does not use. The speaker
# encoder's own preprocessor is unused because both encoders read the mel
# features of the perception preprocessor; ``hidden_to_spks`` is a frozen
# legacy head.
_UNUSED_CHECKPOINT_PREFIXES = (
    "encoder.diarization_model.preprocessor.",
    "encoder.diarization_model.sortformer_modules.hidden_to_spks.",
)


class _TowerLinear(ReplicatedLinear):
    """A replicated linear layer over the last dimension of any input shape.

    vLLM's quantized kernels take ``(tokens, features)``; the tower's activations
    are ``(batch, frames, features)``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Built with return_bias=False, so the layer returns only the output.
        out = cast(torch.Tensor, super().forward(x.reshape(-1, x.shape[-1])))
        return out.view(*x.shape[:-1], out.shape[-1])


def _linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool,
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> _TowerLinear:
    """A linear layer the checkpoint's quantization config applies to by name."""
    return _TowerLinear(
        in_features,
        out_features,
        bias=bias,
        quant_config=quant_config,
        prefix=prefix,
        return_bias=False,
        disable_tp=True,
    )


class MelSpectrogram(nn.Module):
    """Log-mel features as NeMo ``FilterbankFeatures`` computes them at inference.

    Dither is off and no normalization is applied; frames past each row's
    length are set to ``pad_value``. The filterbank and window come from the
    checkpoint and are held in the model dtype, as in the reference, and are
    upcast to float32 for the computation.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.n_fft = cfg["n_fft"]
        self.hop_length = cfg["hop_length"]
        self.win_length = cfg["win_length"]
        self.preemph = cfg["preemph"]
        self.log_zero_guard_value = cfg["log_zero_guard_value"]
        self.mag_power = cfg["mag_power"]
        self.pad_value = cfg["pad_value"]
        self.register_buffer("fb", torch.zeros(1, cfg["n_mels"], self.n_fft // 2 + 1))
        self.register_buffer("window", torch.zeros(self.win_length))

    def get_seq_len(self, num_samples: torch.Tensor) -> torch.Tensor:
        pad_amount = self.n_fft // 2 * 2
        seq_len = torch.floor_divide(
            num_samples + pad_amount - self.n_fft, self.hop_length
        ).to(torch.long)
        return torch.where(num_samples == 0, torch.zeros_like(seq_len), seq_len)

    @torch.no_grad()
    def forward(
        self, audio: torch.Tensor, num_samples: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = self.get_seq_len(num_samples)
        x = audio.float()
        if self.preemph:
            time_mask = torch.arange(x.shape[1], device=x.device).unsqueeze(
                0
            ) < num_samples.unsqueeze(1)
            x = torch.cat((x[:, :1], x[:, 1:] - self.preemph * x[:, :-1]), dim=1)
            x = x.masked_fill(~time_mask, 0.0)
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=True,
            window=self.window.float(),
            return_complex=True,
            pad_mode="constant",
        )
        x = torch.sqrt(torch.view_as_real(x).pow(2).sum(-1))
        if self.mag_power != 1.0:
            x = x.pow(self.mag_power)
        x = torch.matmul(self.fb.float(), x)
        x = torch.log(x + self.log_zero_guard_value)
        pad_mask = torch.arange(x.shape[-1], device=x.device).unsqueeze(
            0
        ) >= seq_len.unsqueeze(1)
        x = x.masked_fill(pad_mask.unsqueeze(1), self.pad_value)
        return x, seq_len


class AudioPreprocessor(nn.Module):
    """Holds the featurizer under NeMo's ``preprocessor.featurizer`` name."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.featurizer = MelSpectrogram(cfg)

    def forward(
        self, audio: torch.Tensor, num_samples: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.featurizer(audio, num_samples)


def normalize_per_feature(x: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    """NeMo ``normalize_batch(x, seq_len, "per_feature")`` on ``(B, C, T)`` features."""
    batch_size, _, max_time = x.shape
    time_steps = (
        torch.arange(max_time, device=x.device)
        .unsqueeze(0)
        .expand(batch_size, max_time)
    )
    valid_mask = time_steps < seq_len.unsqueeze(1)
    denominator = valid_mask.sum(dim=1)
    reference = x[:, :, 0] if max_time else x.new_zeros((batch_size, x.shape[1]))
    reference = reference.masked_fill((denominator == 0).unsqueeze(1), 0.0)
    centered = torch.where(valid_mask.unsqueeze(1), x - reference.unsqueeze(2), 0.0)
    mean = reference + centered.sum(dim=2) / denominator.clamp_min(1).unsqueeze(1)
    variance_sum = torch.sum(
        torch.where(valid_mask.unsqueeze(1), x - mean.unsqueeze(2), 0.0) ** 2, dim=2
    )
    std = torch.sqrt(variance_sum / (denominator.unsqueeze(1) - 1.0))
    std = std.masked_fill(std.isnan(), 0.0) + 1e-5
    normalized = (x - mean.unsqueeze(2)) / std.unsqueeze(2)
    return normalized.masked_fill(~valid_mask.unsqueeze(1), 0.0)


class RotaryEmbedding:
    """NeMo ``RotaryPositionalEncoding``: NeoX rotation of the first ``rot_dim`` dims.

    Angles are computed in float32 on the CPU and stored in the activation
    dtype before use, as the reference does, so rotations match bit for bit.
    """

    def __init__(self, head_dim: int, rotary_fraction: float, base: float) -> None:
        self.rot_dim = int(head_dim * rotary_fraction)
        if self.rot_dim < 2 or self.rot_dim % 2:
            raise ValueError(
                f"Rotary dim must be a positive even number, got {self.rot_dim}."
            )
        self.base = base
        self._cache: dict[
            tuple[torch.dtype, torch.device], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def cos_sin(
        self, length: int, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (dtype, device)
        cached = self._cache.get(key)
        if cached is None or cached[0].shape[0] < length:
            inv_freq = 1.0 / (
                self.base
                ** (
                    torch.arange(0, self.rot_dim, 2, dtype=torch.float32, device="cpu")
                    / self.rot_dim
                )
            )
            positions = torch.arange(
                0, max(length, 1024), dtype=torch.float32, device="cpu"
            )
            freqs = torch.outer(positions, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cached = (
                emb.cos().to(device=device, dtype=dtype),
                emb.sin().to(device=device, dtype=dtype),
            )
            self._cache[key] = cached
        return cached[0][:length], cached[1][:length]

    def apply(
        self, t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        def rotate(x: torch.Tensor) -> torch.Tensor:
            half = x.shape[-1] // 2
            return x * cos + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sin

        if self.rot_dim == t.shape[-1]:
            return rotate(t)
        return torch.cat(
            (rotate(t[..., : self.rot_dim]), t[..., self.rot_dim :]), dim=-1
        )


class FeatureStacking(nn.Module):
    """Stack ``factor`` consecutive frames and project them to the model width."""

    def __init__(
        self,
        factor: int,
        feat_in: int,
        feat_out: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.factor = factor
        self.proj = _linear(
            factor * feat_in,
            feat_out,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
        )

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        batch_size, num_frames, channels = x.shape
        pad = (self.factor - num_frames % self.factor) % self.factor
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        x = x.reshape(
            batch_size, (num_frames + pad) // self.factor, channels * self.factor
        )
        return self.proj(x), (lengths + self.factor - 1) // self.factor


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        cfg: dict[str, Any],
        rope: RotaryEmbedding,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.d_model = cfg["d_model"]
        self.n_heads = cfg["n_heads"]
        self.head_dim = self.d_model // self.n_heads
        self.rope = rope
        self.w_qkv = _linear(
            self.d_model,
            3 * self.d_model,
            bias=cfg["qkv_bias"],
            quant_config=quant_config,
            prefix=f"{prefix}.w_qkv",
        )
        self.out_proj = _linear(
            self.d_model,
            self.d_model,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        self.qk_norm = cfg["qk_norm"]
        if self.qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_frames, _ = x.shape
        qkv = self.w_qkv(x).view(batch_size, num_frames, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if self.qk_norm:
            q = self.q_norm(q).to(v.dtype)
            k = self.k_norm(k).to(v.dtype)
        q = self.rope.apply(q, cos.to(q.dtype), sin.to(q.dtype))
        k = self.rope.apply(k, cos.to(k.dtype), sin.to(k.dtype))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(batch_size, num_frames, self.d_model)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        cfg: dict[str, Any],
        rope: RotaryEmbedding,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        d_model = cfg["d_model"]
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(cfg, rope, quant_config, f"{prefix}.attn")
        self.norm2 = nn.LayerNorm(d_model)
        ff_hidden = int(cfg["ff_expansion"] * d_model)
        # Indices 0 and 3 match the checkpoint's ``ffn.net.{0,3}``.
        self.ffn = nn.Module()
        self.ffn.net = nn.Sequential(
            _linear(
                d_model,
                ff_hidden,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.ffn.net.0",
            ),
            nn.GELU(),
            nn.Identity(),
            _linear(
                ff_hidden,
                d_model,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.ffn.net.3",
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask, cos, sin)
        return x + self.ffn.net(self.norm2(x))


class TransformerEncoder(nn.Module):
    """NeMo ``TransformerEncoder`` with RoPE and full attention, at inference.

    ``forward`` takes ``(B, C, T)`` mel features, or ``(B, T, d_model)``
    pre-encoded embeddings with ``bypass_pre_encode=True``, and returns
    ``(B, d_model, T')`` states with their lengths. Only keys are masked, so
    padded frames never influence valid ones.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.d_model = cfg["d_model"]
        self.subsampling_factor = cfg["subsampling_factor"]
        self.pre_encode = FeatureStacking(
            self.subsampling_factor,
            cfg["feat_in"],
            self.d_model,
            quant_config,
            f"{prefix}.pre_encode",
        )
        self.xscale = math.sqrt(self.d_model) if cfg["xscaling"] else None
        self.rope = RotaryEmbedding(
            self.d_model // cfg["n_heads"], cfg["rotary_fraction"], cfg["rope_base"]
        )
        self.embed_norm = (
            nn.LayerNorm(self.d_model) if cfg["pre_block_norm"] else nn.Identity()
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(cfg, self.rope, quant_config, f"{prefix}.layers.{i}")
                for i in range(cfg["n_layers"])
            ]
        )
        self.final_norm = nn.LayerNorm(self.d_model)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor, bypass_pre_encode: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not bypass_pre_encode:
            x, lengths = self.pre_encode(x, lengths)
        lengths = lengths.to(torch.int64)
        if self.xscale:
            x = x * self.xscale
        x = self.embed_norm(x)
        num_frames = x.shape[1]
        positions = torch.arange(num_frames, device=x.device).unsqueeze(0)
        # A row with no valid frame (a short clip in a later window) attends to
        # frame 0 to stay finite; its outputs are discarded.
        key_mask = (positions < lengths.unsqueeze(1)) | (
            (positions == 0) & (lengths.unsqueeze(1) == 0)
        )
        attn_mask = key_mask[:, None, None, :]
        cos, sin = self.rope.cos_sin(num_frames, x.dtype, x.device)
        for layer in self.layers:
            x = layer(x, attn_mask, cos, sin)
        return self.final_norm(x).transpose(1, 2), lengths


def downsample_preds(preds: torch.Tensor, factor: int) -> torch.Tensor:
    """Average ``(B, T, S)`` probabilities over windows of ``factor`` frames."""
    if factor == 1 or preds.shape[1] == 0:
        return preds
    pooled = F.avg_pool1d(
        preds.transpose(1, 2),
        kernel_size=factor,
        stride=factor,
        ceil_mode=True,
        count_include_pad=False,
    )
    return pooled.transpose(1, 2)


def align_frames(preds: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad with the last frame, or truncate, ``(B, T, S)`` to ``target_len`` frames."""
    current_len = preds.shape[1]
    if current_len < target_len:
        if current_len == 0:
            raise ValueError(
                "Speaker activity cannot be empty when the encoder output is not."
            )
        preds = torch.cat(
            [preds, preds[:, -1:].repeat(1, target_len - current_len, 1)], dim=1
        )
    return preds[:, :target_len]


@dataclass
class SpeakerCacheState:
    """Synchronous streaming state of the Sortformer speaker encoder."""

    spkcache: torch.Tensor
    fifo: torch.Tensor
    spkcache_preds: torch.Tensor | None = None
    spkcache_compressed: bool = False


class SortformerModules(nn.Module):
    """Speaker head and speaker cache of NeMo ``SortformerModules`` at inference."""

    def __init__(
        self,
        cfg: dict[str, Any],
        online: dict[str, Any],
        upsample_factor: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.n_spk = cfg["num_speakers"]
        self.upsample_factor = upsample_factor
        hidden = cfg["tf_d_model"]
        self.encoder_proj = _linear(
            cfg["fc_d_model"],
            hidden,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder_proj",
        )
        self.subpixel_upsample = nn.Conv1d(
            hidden, hidden * upsample_factor, kernel_size=3, padding=1
        )
        self.first_hidden_to_hidden = _linear(
            hidden,
            hidden,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.first_hidden_to_hidden",
        )
        self.single_hidden_to_spks = _linear(
            hidden,
            self.n_spk,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.single_hidden_to_spks",
        )
        self.learnable_sil_emb = nn.Parameter(torch.zeros(cfg["fc_d_model"]))

        self.fifo_len = online["fifo_len"]
        self.spkcache_len = online["spkcache_len"]
        self.spkcache_update_period = online["spkcache_update_period"]
        self.sil_frames_per_spk = cfg["spkcache_sil_frames_per_spk"]
        self.pred_score_threshold = cfg["pred_score_threshold"]
        self.max_index = cfg["max_index"]
        self.scores_boost_latest = cfg["scores_boost_latest"]
        self.strong_boost_rate = cfg["strong_boost_rate"]
        self.weak_boost_rate = cfg["weak_boost_rate"]
        self.min_pos_scores_rate = cfg["min_pos_scores_rate"]
        if self.spkcache_len < (1 + self.sil_frames_per_spk) * self.n_spk:
            raise ValueError("spkcache_len is too small for the number of speakers.")

    def speaker_probs(
        self, hidden: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Upsampled speaker probabilities, ``(B, T * upsample_factor, n_spk)``."""
        batch_size, num_frames, width = hidden.shape
        mask = torch.arange(num_frames, device=hidden.device).unsqueeze(
            0
        ) < lengths.unsqueeze(1)
        # The convolution reads neighbouring frames: zero the padding so a clip's
        # last frame does not depend on the clips batched with it.
        hidden = hidden.masked_fill(~mask.unsqueeze(-1), 0.0)
        upsampled = self.subpixel_upsample(hidden.transpose(1, 2)).transpose(1, 2)
        upsampled = upsampled.reshape(
            batch_size, num_frames, self.upsample_factor, width
        ).reshape(batch_size, num_frames * self.upsample_factor, width)
        logits = self.single_hidden_to_spks(
            F.relu(self.first_hidden_to_hidden(F.relu(upsampled)))
        )
        output_mask = mask.repeat_interleave(self.upsample_factor, dim=1)
        return torch.sigmoid(logits) * output_mask.unsqueeze(-1)

    def update(
        self,
        state: SpeakerCacheState,
        chunk: torch.Tensor,
        preds: torch.Tensor,
        lc: int,
        rc: int,
    ) -> SpeakerCacheState:
        """NeMo ``SortformerModules.streaming_update`` (synchronous)."""
        spkcache_len, fifo_len = state.spkcache.shape[1], state.fifo.shape[1]
        chunk_len = chunk.shape[1] - lc - rc
        fifo_preds = preds[:, spkcache_len : spkcache_len + fifo_len]
        chunk = chunk[:, lc : chunk_len + lc]
        chunk_preds = preds[
            :, spkcache_len + fifo_len + lc : spkcache_len + fifo_len + chunk_len + lc
        ]
        fifo = torch.cat([state.fifo, chunk], dim=1)
        fifo_preds = torch.cat([fifo_preds, chunk_preds], dim=1)
        spkcache, spkcache_preds, compressed = (
            state.spkcache,
            state.spkcache_preds,
            state.spkcache_compressed,
        )

        if fifo_len + chunk_len > self.fifo_len:
            pop_out_len = max(
                self.spkcache_update_period, chunk_len - self.fifo_len + fifo_len
            )
            pop_out_len = min(pop_out_len, fifo_len + chunk_len)
            pop_out_embs, pop_out_preds = (
                fifo[:, :pop_out_len],
                fifo_preds[:, :pop_out_len],
            )
            fifo = fifo[:, pop_out_len:]
            spkcache = torch.cat([spkcache, pop_out_embs], dim=1)
            if compressed:
                spkcache_preds = torch.cat([spkcache_preds, pop_out_preds], dim=1)
            else:
                spkcache_preds = torch.cat(
                    [preds[:, :spkcache_len], pop_out_preds], dim=1
                )
            if spkcache.shape[1] > self.spkcache_len:
                spkcache, spkcache_preds = self._compress_spkcache(
                    spkcache, spkcache_preds
                )
                compressed = True
        return SpeakerCacheState(spkcache, fifo, spkcache_preds, compressed)

    def _compress_spkcache(
        self, emb_seq: torch.Tensor, preds: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep the ``spkcache_len`` most informative frames, as NeMo does."""
        batch_size, _, n_spk = preds.shape
        per_spk = self.spkcache_len // n_spk - self.sil_frames_per_spk
        strong_boost = math.floor(per_spk * self.strong_boost_rate)
        weak_boost = math.floor(per_spk * self.weak_boost_rate)
        min_pos_scores = math.floor(per_spk * self.min_pos_scores_rate)

        threshold = self.pred_score_threshold
        log_probs = torch.log(torch.clamp(preds, min=threshold))
        log_1_probs = torch.log(torch.clamp(1.0 - preds, min=threshold))
        log_1_probs_sum = log_1_probs.sum(dim=2).unsqueeze(-1).expand(-1, -1, n_spk)
        scores = log_probs - log_1_probs + log_1_probs_sum - math.log(0.5)

        is_speech = preds > 0.5
        scores = torch.where(is_speech, scores, float("-inf"))
        is_pos = scores > 0
        replace = (
            (~is_pos) * is_speech * (is_pos.sum(dim=1).unsqueeze(1) >= min_pos_scores)
        )
        scores = torch.where(replace, float("-inf"), scores)

        if self.scores_boost_latest > 0:
            scores[:, self.spkcache_len :, :] += self.scores_boost_latest
        scores = self._boost_topk_scores(scores, strong_boost, scale_factor=2)
        scores = self._boost_topk_scores(scores, weak_boost, scale_factor=1)
        if self.sil_frames_per_spk > 0:
            pad = torch.full(
                (batch_size, self.sil_frames_per_spk, n_spk),
                float("inf"),
                dtype=scores.dtype,
                device=scores.device,
            )
            scores = torch.cat([scores, pad], dim=1)

        num_frames = scores.shape[1]
        flat = scores.permute(0, 2, 1).reshape(batch_size, -1)
        topk_values, topk_indices = torch.topk(
            flat, self.spkcache_len, dim=1, sorted=False
        )
        topk_indices = torch.where(
            topk_values != float("-inf"), topk_indices, self.max_index
        )
        topk_indices, _ = torch.sort(topk_indices, dim=1)
        is_disabled = topk_indices == self.max_index
        topk_indices = torch.remainder(topk_indices, num_frames)
        is_disabled += topk_indices >= num_frames - self.sil_frames_per_spk
        topk_indices[is_disabled] = 0

        sil_emb = (
            self.learnable_sil_emb.to(dtype=emb_seq.dtype)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        emb_gathered = torch.gather(
            emb_seq, 1, topk_indices.unsqueeze(-1).expand(-1, -1, emb_seq.shape[2])
        )
        emb_gathered = torch.where(
            is_disabled.unsqueeze(-1),
            sil_emb.unsqueeze(1).expand(-1, self.spkcache_len, -1),
            emb_gathered,
        )
        preds_gathered = torch.gather(
            preds, 1, topk_indices.unsqueeze(-1).expand(-1, -1, n_spk)
        )
        preds_gathered = torch.where(is_disabled.unsqueeze(-1), 0.0, preds_gathered)
        return emb_gathered, preds_gathered

    @staticmethod
    def _boost_topk_scores(
        scores: torch.Tensor, n_boost_per_spk: int, scale_factor: float
    ) -> torch.Tensor:
        batch_size, _, n_spk = scores.shape
        _, topk_indices = torch.topk(
            scores, n_boost_per_spk, dim=1, largest=True, sorted=False
        )
        batch_indices = torch.arange(batch_size, device=scores.device).view(-1, 1, 1)
        speaker_indices = torch.arange(n_spk, device=scores.device).view(1, 1, -1)
        scores[batch_indices, topk_indices, speaker_indices] -= scale_factor * math.log(
            0.5
        )
        return scores


class SortformerSpeakerEncoder(nn.Module):
    """Streaming inference of the high-resolution Sortformer in NeMo's PEE.

    Mirrors ``SortformerEncLabelModel.forward_streaming_step`` for synchronous
    streaming without speaker permutation. Each step returns speaker
    probabilities on the encoder frame grid for the window's core frames.
    """

    def __init__(
        self,
        encoder_cfg: dict[str, Any],
        sortformer_cfg: dict[str, Any],
        online: dict[str, Any],
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.encoder = TransformerEncoder(
            encoder_cfg, quant_config, f"{prefix}.encoder"
        )
        self.subsampling_factor = encoder_cfg["subsampling_factor"]
        self.sortformer_modules = SortformerModules(
            sortformer_cfg,
            online,
            self.subsampling_factor,
            quant_config,
            f"{prefix}.sortformer_modules",
        )
        self.fc_d_model = sortformer_cfg["fc_d_model"]
        self.n_spk = sortformer_cfg["num_speakers"]

    def init_state(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> SpeakerCacheState:
        empty = torch.zeros(
            (batch_size, 0, self.fc_d_model), dtype=dtype, device=device
        )
        return SpeakerCacheState(spkcache=empty, fifo=empty.clone())

    def step(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        state: SpeakerCacheState,
        left_offset: int,
        right_offset: int,
    ) -> tuple[SpeakerCacheState, torch.Tensor]:
        """Process one window of ``(B, C, T)`` mel features.

        Returns the new state and the core frames' speaker probabilities on the
        encoder grid.
        """
        modules = self.sortformer_modules
        factor = self.subsampling_factor
        chunk_embs, chunk_lengths = self.encoder.pre_encode(features, lengths)
        embs = torch.cat([state.spkcache, state.fifo, chunk_embs], dim=1)
        context_len = state.spkcache.shape[1] + state.fifo.shape[1]
        emb_lengths = context_len + chunk_lengths
        states, state_lengths = self.encoder(embs, emb_lengths, bypass_pre_encode=True)
        hidden = modules.encoder_proj(states.transpose(1, 2))
        high_res_preds = modules.speaker_probs(hidden, state_lengths)

        lc = round(left_offset / factor)
        rc = math.ceil(right_offset / factor)
        preds = downsample_preds(high_res_preds, modules.upsample_factor)
        valid = torch.arange(preds.shape[1], device=preds.device).view(
            1, -1, 1
        ) < state_lengths.view(-1, 1, 1)
        preds = torch.where(valid, preds, 0.0)
        state = modules.update(state, chunk_embs, preds, lc, rc)

        chunk_len = chunk_embs.shape[1] - lc - rc
        start = (context_len + lc) * modules.upsample_factor
        chunk_preds = high_res_preds[
            :, start : start + chunk_len * modules.upsample_factor
        ]
        return state, downsample_preds(chunk_preds, factor)


class ParallelExpertEncoder(nn.Module):
    """NeMo ``ParallelExpertEncoder`` online inference with a transformer ASR encoder.

    ``forward`` takes unnormalized ``(B, C, T)`` mel features and returns the
    fused ``(B, d_model, T')`` states.
    """

    def __init__(
        self,
        audio_config: dict[str, Any],
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        online = audio_config["online"]
        fusion = audio_config["fusion"]
        self.asr_encoder = TransformerEncoder(
            audio_config["asr_encoder"],
            quant_config,
            f"{prefix}.asr_encoder",
        )
        self.diarization_model = SortformerSpeakerEncoder(
            audio_config["speaker_encoder"],
            audio_config["sortformer"],
            online,
            quant_config,
            f"{prefix}.diarization_model",
        )
        self.subsampling_factor = self.asr_encoder.subsampling_factor
        self.window_feat_len = online["window_frames"] * self.subsampling_factor
        self.left_ctx_feat_len = online["left_context_frames"] * self.subsampling_factor
        self.right_ctx_feat_len = (
            online["right_context_frames"] * self.subsampling_factor
        )
        self.speaker_feature_mode = fusion["speaker_feature_mode"]
        self.speaker_activity_threshold = fusion["speaker_activity_threshold"]
        self.spk_kernel_scale = fusion["spk_kernel_scale"]
        n_spk = self.diarization_model.n_spk
        self.asr_norm = nn.LayerNorm(self.asr_encoder.d_model)
        self.diar_norm = nn.LayerNorm(n_spk)
        self.register_buffer(
            "diar_kernel",
            self._sinusoid_kernel(n_spk, self.asr_encoder.d_model),
            persistent=False,
        )

    @staticmethod
    def _sinusoid_kernel(num_speakers: int, width: int) -> torch.Tensor:
        position = torch.arange(num_speakers, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, width, 2, dtype=torch.float32)
            * -(math.log(10000.0) / width)
        )
        kernel = torch.zeros(num_speakers, width, dtype=torch.float32)
        kernel[:, 0::2] = torch.sin(position * div_term)
        kernel[:, 1::2] = torch.cos(position * div_term)
        return kernel

    def _output_boundary(self, feat_boundary: int) -> int:
        return (feat_boundary + self.subsampling_factor - 1) // self.subsampling_factor

    @property
    def dtype(self) -> torch.dtype:
        """The model dtype, which quantized layers' weights may not have."""
        return self.asr_norm.weight.dtype

    def forward(
        self, mel: torch.Tensor, mel_lengths: torch.Tensor, max_feat_len: int
    ) -> torch.Tensor:
        total_feat_len = min(mel.shape[-1], max_feat_len)
        num_windows = max(1, math.ceil(total_feat_len / self.window_feat_len))
        asr_signal = normalize_per_feature(mel, mel_lengths).to(self.dtype)
        diar_signal = mel.to(self.dtype)
        state = self.diarization_model.init_state(mel.shape[0], self.dtype, mel.device)

        asr_windows: list[torch.Tensor] = []
        speaker_windows: list[torch.Tensor] = []
        for index in range(num_windows):
            start = index * self.window_feat_len
            end = min(start + self.window_feat_len, total_feat_len)
            context_start = max(start - self.left_ctx_feat_len, 0)
            context_end = min(end + self.right_ctx_feat_len, total_feat_len)
            left_offset, right_offset = start - context_start, context_end - end
            window_lengths = (mel_lengths - context_start).clamp(
                min=0, max=context_end - context_start
            )

            encoded, _ = self.asr_encoder(
                asr_signal[:, :, context_start:context_end], window_lengths
            )
            left_drop = left_offset // self.subsampling_factor
            core_len = self._output_boundary(end) - self._output_boundary(start)
            core_len = max(0, min(core_len, encoded.shape[-1] - left_drop))
            asr_windows.append(encoded[:, :, left_drop : left_drop + core_len])

            state, window_preds = self.diarization_model.step(
                diar_signal[:, :, context_start:context_end],
                window_lengths,
                state,
                left_offset,
                right_offset,
            )
            speaker_windows.append(align_frames(window_preds, core_len))

        asr_encoded = torch.cat(asr_windows, dim=2)
        speaker_probs = torch.cat(speaker_windows, dim=1)
        return self._fuse(asr_encoded, speaker_probs)

    def _fuse(
        self, asr_encoded: torch.Tensor, speaker_probs: torch.Tensor
    ) -> torch.Tensor:
        states = asr_encoded.transpose(1, 2)
        speaker_probs = align_frames(speaker_probs, states.shape[1]).to(
            device=states.device, dtype=states.dtype
        )
        if self.speaker_feature_mode == "thresholded":
            speaker_features = (speaker_probs > self.speaker_activity_threshold).to(
                states.dtype
            )
        else:
            speaker_features = speaker_probs
        normalized_states = self.asr_norm(states)
        normalized_speakers = self.diar_norm(speaker_features)
        infusion = torch.matmul(
            normalized_speakers, self.diar_kernel.to(normalized_speakers.dtype)
        )
        return (normalized_states + self.spk_kernel_scale * infusion).transpose(1, 2)


class Nemotron3_5TranscribeAudioTower(nn.Module):
    """Waveform to decoder-width embeddings: mel, Parallel Expert Encoder, Linear."""

    def __init__(
        self,
        audio_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.preprocessor = AudioPreprocessor(audio_config["preprocessor"])
        self.encoder = ParallelExpertEncoder(
            audio_config, quant_config, maybe_prefix(prefix, "encoder")
        )
        projection = audio_config["projection"]
        self.proj = _linear(
            projection["input_dim"],
            projection["output_dim"],
            bias=True,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "proj"),
        )
        self.subsampling_factor = self.encoder.subsampling_factor

    def encode(
        self, signals: Sequence[torch.Tensor], num_samples_cpu: torch.Tensor
    ) -> list[torch.Tensor]:
        """Encode each clip into its embeddings, returned in input order.

        Every row of a batch runs every window of the batch's longest clip, so a
        clip is batched only with clips that span as many windows.
        """
        window = self.encoder.window_feat_len
        num_mel_frames = self.preprocessor.featurizer.get_seq_len(num_samples_cpu)
        groups: dict[int, list[int]] = {}
        for index, num_windows in enumerate(
            ((num_mel_frames + window - 1) // window).tolist()
        ):
            groups.setdefault(num_windows, []).append(index)

        device = self.encoder.asr_norm.weight.device
        embeddings: dict[int, torch.Tensor] = {}
        for indices in groups.values():
            lengths_cpu = num_samples_cpu[indices]
            audio = torch.zeros(
                len(indices), int(lengths_cpu.max()), dtype=torch.float32, device=device
            )
            for row, index in enumerate(indices):
                length = int(lengths_cpu[row])
                audio[row, :length] = (
                    signals[index]
                    .flatten()[:length]
                    .to(device=device, dtype=torch.float32)
                )
            lengths = (
                lengths_cpu
                if device.type == "cpu"
                else async_tensor_h2d(lengths_cpu, device=device, dtype=torch.long)
            )
            for index, emb in zip(indices, self(audio, lengths, lengths_cpu)):
                embeddings[index] = emb
        return [embeddings[index] for index in range(len(signals))]

    def forward(
        self,
        audio: torch.Tensor,
        num_samples: torch.Tensor,
        num_samples_cpu: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Encode padded ``(B, N)`` float32 audio into one embedding tensor per row.

        ``num_samples_cpu`` holds the lengths on the host, so the window bounds
        and the output lengths need no device-to-host copy.
        """
        mel, mel_lengths = self.preprocessor(audio, num_samples)
        # NeMo's AudioPreprocessor computes in float32 but returns features in
        # the module dtype, so the encoders normalize model-dtype features.
        mel = mel.to(self.encoder.dtype)
        mel_lengths_cpu = self.preprocessor.featurizer.get_seq_len(
            num_samples_cpu
        ).clamp(max=mel.shape[-1])
        encoded = self.encoder(mel, mel_lengths, int(mel_lengths_cpu.max()))
        encoded_lengths = (
            mel_lengths_cpu + self.subsampling_factor - 1
        ) // self.subsampling_factor
        embeddings = self.proj(encoded.transpose(1, 2))
        return [
            emb[:length] for emb, length in zip(embeddings, encoded_lengths.tolist())
        ]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load NeMo ``perception.*`` tensors, named without the ``perception.``."""
        targets = dict(self.named_parameters())
        targets.update(dict(self.named_buffers()))
        targets.pop("encoder.diar_kernel", None)
        loaded: set[str] = set()
        for name, tensor in weights:
            if name.endswith("._extra_state") or name.startswith(
                _UNUSED_CHECKPOINT_PREFIXES
            ):
                continue
            target = targets.get(name)
            if target is None:
                raise ValueError(f"Unexpected audio tower weight {name!r}.")
            weight_loader = getattr(target, "weight_loader", None)
            if weight_loader is not None:
                # vLLM's linear layers: quantized parameters and scalar scales.
                weight_loader(target, tensor)
            elif target.shape != tensor.shape:
                raise ValueError(
                    f"Shape mismatch for {name!r}: {tuple(tensor.shape)} vs "
                    f"{tuple(target.shape)}."
                )
            else:
                with torch.no_grad():
                    target.copy_(tensor.to(device=target.device, dtype=target.dtype))
            loaded.add(name)
        missing = sorted(set(targets) - loaded)
        if missing:
            raise ValueError(
                f"Audio tower weights missing from the checkpoint ({len(missing)}): "
                f"{missing[:8]}"
            )
        return loaded
