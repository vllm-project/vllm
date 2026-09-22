# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from vllm.third_party.deep_gemm import per_block_cast_to_fp8

from .vision_attention import (
    VisionRMSNorm,
    VisionRotaryEmbedding,
    VisionRotaryPositionEmbedding,
    apply_vision_attention_residual,
    attn_uses_seqlens,
    build_vision_attention,
    prepare_rotary_pos_emb_vision,
    resolve_attn_implementation,
)
from .vision_moe import note_vision_fused_moe_fp8


class DotsMoEVitConfig(PretrainedConfig):
    model_type: str = "dots_moe_vit"

    def __init__(
        self,
        embed_dim: int = 1536,
        hidden_size: int = 2048,
        intermediate_size: int = 4224,
        moe_intermediate_size: int = 2112,
        num_hidden_layers: int = 42,
        num_attention_heads: int = 24,
        num_channels: int = 3,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 1,
        rms_norm_eps: float = 1e-5,
        use_bias: bool = False,
        use_qk_norm: bool = True,
        attn_implementation="flash_attention_3",  # "eager","eager_v2","sdpa","flash_attention_2","flash_attention_3"
        initializer_range=0.02,
        is_causal=False,
        post_norm=True,
        gradient_checkpointing=False,
        pyramid_num_routed: list[int] | None = None,
        capacity_factor: float = 2.0,
        router_scoring_func: str = "sigmoid",
        router_scale: float = 1.0,
        adapter_in_dim: int = 1536,
        adapter_out_dim: int = 2048,
        adapter_merge_size: int = 2,
        # ``pixel_shuffle_mlp`` reshapes 2x2 spatial neighbours via NHWC
        # pixel-shuffle, then runs layer norm and a two-layer MLP.
        # ``patch_merger`` (cybertron PatchMerger, e.g. fireall_iter02275) skips the pixel-shuffle permutation and instead views every 4 consecutive 2x2-grouped tokens as one row.
        adapter_type: str = "pixel_shuffle_mlp",
        # If True the preprocessor already emits patches in 2x2-grouped order (qwen ``merge_size=2`` flatten path), so RoPE positions must also be regrouped
        # to match the in-block layout. Older checkpoints were trained with
        # ``pre_pixel_shuffle=False``, so the default keeps row-major RoPE.
        pre_pixel_shuffle: bool = False,
        # Keep the native encoder's post-load compile behavior.  Compiling in
        # ``__init__`` would rename checkpoint keys through ``_orig_mod``.
        enable_torch_compile: bool = True,
        # If True, use FP8 MoE implementation
        enable_fp8_moe: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.rms_norm_eps = rms_norm_eps
        self.use_bias = use_bias
        self.use_qk_norm = use_qk_norm
        self.attn_implementation = attn_implementation
        self.initializer_range = initializer_range
        self.is_causal = is_causal
        self.post_norm = post_norm
        self.gradient_checkpointing = gradient_checkpointing
        self.pyramid_num_routed = pyramid_num_routed or []
        self.capacity_factor = capacity_factor
        self.router_scoring_func = router_scoring_func
        self.router_scale = router_scale
        self.adapter_in_dim = adapter_in_dim
        self.adapter_out_dim = adapter_out_dim
        self.adapter_merge_size = adapter_merge_size
        if adapter_type not in ("pixel_shuffle_mlp", "patch_merger"):
            raise ValueError(
                f"adapter_type must be 'pixel_shuffle_mlp' or 'patch_merger', got {adapter_type!r}"
            )
        self.adapter_type = adapter_type
        self.pre_pixel_shuffle = pre_pixel_shuffle
        self.enable_torch_compile = enable_torch_compile
        self.enable_fp8_moe = enable_fp8_moe


# ---- FFN modules ----


class DotsSwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features, bias=False):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        self.fc3 = nn.Linear(in_features, hidden_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)) * self.fc3(x))


class MoESwiGLUFFN(nn.Module):
    """MoE FFN with per-expert SwiGLU experts, sigmoid/softmax gating, top-k routing."""

    def __init__(self, config: DotsMoEVitConfig, layer_number: int):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.hidden_size = config.embed_dim
        self.num_routed = config.pyramid_num_routed[layer_number]
        self.capacity_factor = config.capacity_factor
        self.router_scoring_func = config.router_scoring_func
        self.router_scale = config.router_scale

        self.register_buffer(
            "router_bias", torch.zeros(self.num_routed, dtype=torch.float32)
        )

        self.experts = nn.ModuleList(
            [
                DotsSwiGLUFFN(
                    self.hidden_size, config.moe_intermediate_size, bias=config.use_bias
                )
                for _ in range(self.num_routed)
            ]
        )

        self.gate_weight = nn.Parameter(
            torch.empty((self.num_routed, self.hidden_size), dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mirror cybertron's ``AIMv2MoEGEMMSwiGLUFFN.forward``: keep ``gating_prob`` /
        # router-bias add in fp32 and run topk in fp32. The legacy ``.type_as(x_flat)``
        # path on bf16 made expert routing diverge whenever two routed-expert scores
        # tied within bf16 precision, which is the dominant source of numerical drift
        # observed against latest cybertron checkpoints.
        epsilon = 1e-9
        x_flat = x.contiguous()
        num_tokens = x_flat.shape[0]

        gate_logits = F.linear(x_flat.float(), self.gate_weight.float())

        if self.router_scoring_func == "sigmoid":
            gating_prob = torch.sigmoid(gate_logits)
        else:
            gating_prob = torch.softmax(gate_logits, dim=-1, dtype=torch.float32)

        aggregated_output = torch.zeros_like(x_flat)
        aggregated_gate = torch.zeros(num_tokens, dtype=x.dtype, device=x.device)

        topk = min(int(self.capacity_factor), self.num_routed)

        gating_with_bias = gating_prob + self.router_bias.to(torch.float32).unsqueeze(0)
        _, topk_indices = torch.topk(gating_with_bias, k=topk, dim=-1, sorted=False)

        routed_weights = gating_prob.gather(1, topk_indices)
        if self.router_scoring_func == "sigmoid" and topk > 1:
            routed_weights = routed_weights / (
                routed_weights.sum(dim=-1, keepdim=True) + epsilon
            )
        routed_weights = (routed_weights * self.router_scale).to(x_flat.dtype)

        for expert_idx in range(self.num_routed):
            selected_mask = topk_indices == expert_idx
            if selected_mask.sum() == 0:
                continue
            n_idx, top = torch.where(selected_mask)
            # Fancy indexing can yield non-contiguous rows; cuBLAS bf16 GEMM may then fail
            # with ``CUBLAS_STATUS_INVALID_VALUE`` inside ``F.linear``.
            x_selected = x_flat[n_idx].contiguous()
            expert_output = self.experts[expert_idx](x_selected)
            contrib = expert_output * routed_weights[n_idx, top].unsqueeze(-1)
            aggregated_output[n_idx] = aggregated_output[n_idx] + contrib
            aggregated_gate[n_idx] = aggregated_gate[n_idx] + routed_weights[n_idx, top]

        aggregated_output = aggregated_output / (
            aggregated_gate.unsqueeze(-1) + epsilon
        )
        return aggregated_output


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _per_block_cast_to_fp8_padded(
    weight: torch.Tensor,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = weight.shape
    padded = weight.new_zeros(
        _ceil_to_multiple(rows, block_size),
        _ceil_to_multiple(columns, block_size),
    )
    padded[:rows, :columns] = weight
    return per_block_cast_to_fp8(
        padded.contiguous(),
        use_ue8m0=False,
        gran_k=block_size,
    )


class MoESwiGLUFFNFP8(MoESwiGLUFFN):
    """NOTE vision MoE using the checkpoint's local block-FP8 semantics."""

    @torch.no_grad()
    def process_weights_after_loading(self) -> None:
        if hasattr(self, "_fused_w13_fp8"):
            return

        w13_weights = []
        w13_scales = []
        w2_weights = []
        w2_scales = []
        for expert in self.experts:
            w1, s1 = _per_block_cast_to_fp8_padded(expert.fc1.weight)
            w3, s3 = _per_block_cast_to_fp8_padded(expert.fc3.weight)
            w2, s2 = _per_block_cast_to_fp8_padded(expert.fc2.weight)
            w13_weights.append(torch.cat((w1, w3), dim=0))
            w13_scales.append(torch.cat((s1, s3), dim=0))
            w2_weights.append(w2)
            w2_scales.append(s2)

        self.register_buffer(
            "_fused_w13_fp8",
            torch.stack(w13_weights).contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "_fused_w13_scale",
            torch.stack(w13_scales).contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "_fused_w2_fp8",
            torch.stack(w2_weights).contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "_fused_w2_scale",
            torch.stack(w2_scales).contiguous(),
            persistent=False,
        )
        del self.experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "_fused_w13_fp8"):
            raise RuntimeError("NOTE vision FP8 weights were not initialized")

        gate_logits = F.linear(x.float(), self.gate_weight.float())
        if self.router_scoring_func == "sigmoid":
            scores = torch.sigmoid(gate_logits)
        else:
            scores = torch.softmax(gate_logits, dim=-1, dtype=torch.float32)

        topk = min(int(self.capacity_factor), self.num_routed)
        biased_scores = scores + self.router_bias.float().unsqueeze(0)
        topk_ids = torch.topk(biased_scores, k=topk, dim=-1, sorted=False).indices
        topk_weights = scores.gather(1, topk_ids)
        if self.router_scoring_func == "sigmoid" and topk > 1:
            topk_weights = topk_weights / (
                topk_weights.sum(dim=-1, keepdim=True) + 1e-9
            )
        topk_weights = topk_weights * self.router_scale

        output = note_vision_fused_moe_fp8(
            x.contiguous(),
            self._fused_w13_fp8,
            self._fused_w2_fp8,
            topk_weights,
            topk_ids.to(torch.int32),
            self._fused_w13_scale,
            self._fused_w2_scale,
        )
        denominator = topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return (output / denominator).type_as(x)


# ---- PatchEmbed ----


class DotsPatchEmbed(nn.Module):
    def __init__(self, config: DotsMoEVitConfig):
        super().__init__()
        self.num_channels = config.num_channels
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.embed_dim = config.embed_dim
        self.proj = nn.Conv2d(
            config.num_channels,
            config.embed_dim,
            kernel_size=(config.patch_size, config.patch_size),
            stride=(config.patch_size, config.patch_size),
        )
        self.norm = VisionRMSNorm(config.embed_dim, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(
            -1,
            self.num_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )[:, :, 0]
        x = self.proj(x).view(-1, self.embed_dim)
        x = self.norm(x)
        return x


# ---- Block ----


class MoEVisionBlock(nn.Module):
    def __init__(self, config: DotsMoEVitConfig, layer_number: int):
        super().__init__()
        attn_impl = resolve_attn_implementation(
            config.attn_implementation, eager_fallback="eager_v2"
        )
        self.attn = build_vision_attention(attn_impl, config, eager_fallback="eager_v2")
        self._attn_uses_seqlens = attn_uses_seqlens(attn_impl)
        self.norm_1 = VisionRMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.norm_2 = VisionRMSNorm(config.embed_dim, eps=config.rms_norm_eps)

        is_moe = (
            config.pyramid_num_routed
            and layer_number < len(config.pyramid_num_routed)
            and config.pyramid_num_routed[layer_number] > 0
        )
        if is_moe:
            mlp_cls = MoESwiGLUFFNFP8 if config.enable_fp8_moe else MoESwiGLUFFN
            self.mlp = mlp_cls(config, layer_number)
        else:
            self.mlp = DotsSwiGLUFFN(
                config.embed_dim, config.intermediate_size, bias=config.use_bias
            )

    def forward(
        self,
        hidden_states,
        cu_seqlens,
        rotary_pos_emb,
        max_seqlen: int,
        seqlens: list[int] | None = None,
    ) -> torch.Tensor:
        hidden_states = apply_vision_attention_residual(
            self.attn,
            self.norm_1,
            hidden_states,
            cu_seqlens,
            max_seqlen,
            rotary_pos_emb,
            seqlens=seqlens,
            uses_seqlens=self._attn_uses_seqlens,
        )
        hidden_states = hidden_states + self.mlp(self.norm_2(hidden_states))
        return hidden_states


# ---- Adapter (pixel_shuffle + MLP) ----


def _pixel_shuffle(x, scale_factor=0.5):
    if x.size(1) % 2 == 1:
        x = torch.cat([x[:, :1], x], dim=1)
    if x.size(2) % 2 == 1:
        x = torch.cat([x[:, :, :1], x], dim=2)
    n, h, w, c = x.size()
    x = x.reshape(n, h, int(w * scale_factor), int(c / scale_factor))
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.reshape(
        n,
        int(w * scale_factor),
        int(h * scale_factor),
        int(c / (scale_factor * scale_factor)),
    )
    x = x.permute(0, 2, 1, 3).contiguous()
    return x


class PixelShuffleAdapter(nn.Module):
    """Legacy adapter: NHWC pixel-shuffle spatial merge + LayerNorm + 2-layer MLP.

    Mirrors ``cybertron`` ``FCAdapter(pool_kind='pixel_shuffle', proj_kind='mlp2x_ln_gelu')``.
    State-dict keys: ``proj.0`` (LayerNorm of in_dim*merge**2), ``proj.1`` / ``proj.3`` (Linear).
    """

    def __init__(self, config: DotsMoEVitConfig):
        super().__init__()
        in_dim = config.adapter_in_dim
        out_dim = config.adapter_out_dim
        merge_size = config.adapter_merge_size
        merged_dim = in_dim * merge_size**2
        self.proj = nn.Sequential(
            LayerNorm(merged_dim),
            nn.Linear(merged_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(
        self,
        patch_embed: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        assert patch_embed.dim() == 2 and grid_thw is not None
        image_features = []
        token_index = 0
        for i in range(grid_thw.shape[0]):
            grid_t, grid_h, grid_w = grid_thw[i]
            images_token_length = grid_t * grid_h * grid_w
            _pe = patch_embed[token_index : token_index + images_token_length]
            token_index += images_token_length
            if grid_t == 1:
                _pe = _pe.reshape(int(grid_h), int(grid_w), -1).unsqueeze(0)
            else:
                _pe = _pe.reshape(int(grid_t), int(grid_h), int(grid_w), -1)
            _pe = _pixel_shuffle(_pe, scale_factor=0.5)
            _pe = _pe.squeeze(0) if grid_t == 1 else _pe.reshape(-1, _pe.shape[-1])
            image_features.append(_pe.reshape(-1, _pe.shape[-1]))
        out = torch.cat(image_features, dim=0)
        out = self.proj(out)
        return out


class PatchMergerAdapter(nn.Module):
    """Cybertron ``PatchMerger`` (``pool_kind='patch_merger', proj_kind='identity'``).

    Assumes the encoder output is already laid out in ``merge_size``x``merge_size`` groups
    (qwen ``pre_pixel_shuffle`` preprocessor + RoPE grouped accordingly), so merging is a
    simple ``view(-1, merge**2 * in_dim)`` of consecutive tokens. State-dict layout matches
    cybertron's ``PatchMerger`` (``ln_q`` over the per-token dim, ``mlp.0`` / ``mlp.2`` Linear).
    """

    def __init__(self, config: DotsMoEVitConfig):
        super().__init__()
        in_dim = config.adapter_in_dim
        out_dim = config.adapter_out_dim
        merge_size = config.adapter_merge_size
        merged_dim = in_dim * merge_size**2
        self.merge_size = merge_size
        self.merged_dim = merged_dim
        self.ln_q = LayerNorm(in_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(merged_dim, merged_dim),
            nn.GELU(),
            nn.Linear(merged_dim, out_dim),
        )

    def forward(
        self,
        patch_embed: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        assert patch_embed.dim() == 2 and grid_thw is not None
        x = self.ln_q(patch_embed)
        x = x.reshape(-1, self.merged_dim)
        return self.mlp(x)


_ADAPTER_CLASSES: dict[str, type[nn.Module]] = {
    "pixel_shuffle_mlp": PixelShuffleAdapter,
    "patch_merger": PatchMergerAdapter,
}


@dataclass(frozen=True)
class VitForwardMeta:
    cu_seqlens: torch.Tensor
    max_seqlen: int
    rotary_pos_emb: VisionRotaryPositionEmbedding
    grid_thw_cpu: torch.Tensor
    seqlens: list[int] | None = None


# ---- Full Model ----


class DotsMoEVitModel(PreTrainedModel):
    config_class = DotsMoEVitConfig

    def __init__(self, config: DotsMoEVitConfig) -> None:
        super().__init__(config)
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = DotsPatchEmbed(config)

        head_dim = config.embed_dim // config.num_attention_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2, cache_seq_len=100000)

        self.blocks = nn.ModuleList(
            [MoEVisionBlock(config, i) for i in range(config.num_hidden_layers)]
        )

        if config.post_norm:
            self.post_trunk_norm = VisionRMSNorm(
                config.embed_dim, eps=config.rms_norm_eps
            )

        adapter_cls = _ADAPTER_CLASSES.get(config.adapter_type)
        if adapter_cls is None:
            raise ValueError(f"Unknown adapter_type {config.adapter_type!r}")
        self.adapter = adapter_cls(config)

        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint

    @property
    def dtype(self) -> torch.dtype:
        mlp = self.blocks[0].mlp
        if hasattr(mlp, "fc13"):
            return mlp.fc13.weight.dtype
        if hasattr(mlp, "fc1"):
            return mlp.fc1.weight.dtype
        expert = mlp.experts[0]
        if hasattr(expert, "fc13"):
            return expert.fc13.weight.dtype
        return expert.fc1.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def process_weights_after_loading(self) -> None:
        for block in self.blocks:
            if isinstance(block.mlp, MoESwiGLUFFNFP8):
                block.mlp.process_weights_after_loading()
        if self.config.enable_torch_compile:
            self.compile_block_modules()

    def compile_block_modules(self, **compile_kwargs: Any) -> None:
        for block in self.blocks:
            block.attn = torch.compile(block.attn, **compile_kwargs)
            block.norm_1 = torch.compile(block.norm_1, **compile_kwargs)
            block.norm_2 = torch.compile(block.norm_2, **compile_kwargs)

    def get_pos_ids_by_grid(self, grid_thw_cpu: torch.Tensor):
        # Mirrors ``cybertron`` ``AIMv2NativeModel.rot_pos_emb``: when ``pre_pixel_shuffle``
        # is set, RoPE positions follow the qwen ``merge_size`` grouped layout (default 2x2);
        # otherwise positions are flat row-major regardless of ``spatial_merge_size``.
        if self.config.pre_pixel_shuffle:
            rope_merge_size = (
                self.spatial_merge_size if self.spatial_merge_size > 1 else 2
            )
        else:
            rope_merge_size = 1
        pos_ids = []
        for t, h, w in grid_thw_cpu:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // rope_merge_size,
                rope_merge_size,
                w // rope_merge_size,
                rope_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // rope_merge_size,
                rope_merge_size,
                w // rope_merge_size,
                rope_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        return pos_ids

    def rot_pos_emb(self, grid_thw_cpu: torch.Tensor):
        pos_ids = torch.cat(self.get_pos_ids_by_grid(grid_thw_cpu), dim=0)
        pos_ids = pos_ids.to(self.device, non_blocking=True)
        max_grid_size = int(grid_thw_cpu[:, 1:].max())
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        return rotary_pos_emb_full[pos_ids].flatten(1)

    def prepare_meta(self, grid_thw_cpu: torch.Tensor) -> VitForwardMeta:
        if grid_thw_cpu.device.type != "cpu":
            raise ValueError("NOTE vision grid_thw must remain on CPU")

        seq_per_frame = grid_thw_cpu[:, 1] * grid_thw_cpu[:, 2]
        frame_lengths = torch.repeat_interleave(seq_per_frame, grid_thw_cpu[:, 0])
        cu_seqlens_cpu = F.pad(
            frame_lengths.cumsum(dim=0, dtype=torch.int32), (1, 0), value=0
        )
        resolved_attn = resolve_attn_implementation(
            self.config.attn_implementation, eager_fallback="eager_v2"
        )
        return VitForwardMeta(
            cu_seqlens=cu_seqlens_cpu.to(self.device, non_blocking=True),
            max_seqlen=int(seq_per_frame.max()),
            rotary_pos_emb=prepare_rotary_pos_emb_vision(
                self.rot_pos_emb(grid_thw_cpu)
            ),
            grid_thw_cpu=grid_thw_cpu,
            seqlens=(
                frame_lengths.tolist() if attn_uses_seqlens(resolved_attn) else None
            ),
        )

    def forward(
        self, hidden_states: torch.Tensor, meta: VitForwardMeta, bf16=True
    ) -> torch.Tensor:
        if bf16:
            hidden_states = hidden_states.to(
                device=self.device, dtype=self.dtype, non_blocking=True
            )
        hidden_states = self.patch_embed(hidden_states)

        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__,
                    hidden_states,
                    meta.cu_seqlens,
                    meta.rotary_pos_emb,
                    meta.max_seqlen,
                    meta.seqlens,
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    meta.cu_seqlens,
                    meta.rotary_pos_emb,
                    meta.max_seqlen,
                    seqlens=meta.seqlens,
                )

        if self.config.post_norm:
            hidden_states = self.post_trunk_norm(hidden_states)

        hidden_states = self.adapter(hidden_states, meta.grid_thw_cpu)
        return hidden_states
