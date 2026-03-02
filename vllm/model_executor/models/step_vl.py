# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""This is basically a copy from perception_models/core/vision_encoder/pe.py"""

from collections.abc import Callable
from functools import partial

import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
from vllm.model_executor.layers.conv import Conv2dLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig

from .step3_vl import Step3VLForConditionalGeneration
from .utils import WeightsMapper, init_vllm_registered_model, maybe_prefix
from .vision import is_vit_use_data_parallel, run_dp_sharded_vision_model

_DEFAULT_NORM_LAYER = partial(nn.LayerNorm, eps=1e-5)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    dtype = t.dtype

    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[-1], (
        "feature dimension {} is not of sufficient size to rotate in all the "
        "positions {}".format(t.shape[-1], rot_dim)
    )

    t_left, t, t_right = (
        t[..., :start_index],
        t[..., start_index:end_index],
        t[..., end_index:],
    )
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    out = torch.cat((t_left, t, t_right), dim=-1)

    return out.type(dtype)


class PerceptionEncoderRope2D(nn.Module):
    def __init__(
        self,
        dim: int,
        max_grid_height: int,
        max_grid_width: int,
        use_cls_token: bool = False,
        theta=10000,
        max_freq=10,
        num_freqs=1,
        theta_rescale_factor=1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_grid_height = max_grid_height
        self.max_grid_width = max_grid_width
        self.use_cls_token = use_cls_token
        self.theta = theta * theta_rescale_factor ** (dim / (dim - 2))
        self.max_freq = max_freq
        self.num_freqs = num_freqs
        cache = self._compute_2d_freqs()
        self.register_buffer("freqs_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: int | float, dim: int) -> torch.Tensor:
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        return freqs

    def _compute_freqs(self, t: torch.Tensor, inv_freq: torch.Tensor):
        freqs = torch.einsum("..., f -> ... f", t.type(inv_freq.dtype), inv_freq)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        return freqs

    def _compute_2d_freqs(self) -> torch.Tensor:
        grid_h_range = torch.arange(self.max_grid_height, dtype=torch.float)
        grid_w_range = torch.arange(self.max_grid_width, dtype=torch.float)
        if self.use_cls_token:
            grid_h_range += 1
            grid_w_range += 1
        inv_freq = self._compute_inv_freq(self.theta, self.dim // 2)
        freqs_h = self._compute_freqs(grid_h_range, inv_freq)[:, None].expand(
            self.max_grid_height, self.max_grid_width, -1
        )
        freqs_w = self._compute_freqs(grid_w_range, inv_freq)[None, :].expand(
            self.max_grid_height, self.max_grid_width, -1
        )
        freqs = torch.cat([freqs_w, freqs_h], dim=-1).reshape(
            self.max_grid_height * self.max_grid_width, -1
        )
        if self.use_cls_token:
            freqs = torch.cat([torch.zeros(1, freqs.shape[-1]), freqs], dim=0)
        freqs = freqs[None, None, ...]
        return freqs

    def forward(self, q: torch.Tensor, k: torch.Tensor, grid_hw: tuple[int, int]):
        if grid_hw[0] != self.max_grid_height or grid_hw[1] != self.max_grid_width:
            rows = torch.arange(grid_hw[0], device=q.device).view(-1, 1)
            cols = torch.arange(grid_hw[1], device=q.device).view(1, -1)
            positions = (rows * self.max_grid_width + cols).reshape(-1).to(torch.long)
            if self.use_cls_token:
                positions = torch.cat(
                    [torch.zeros(1, device=q.device), positions + 1], dim=0
                )
                positions = positions.to(torch.long)
            freqs = self.freqs_cache.index_select(2, positions)
        else:
            freqs = self.freqs_cache
        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)
        return q, k


class PerceptionEncoderLayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PerceptionEncoderMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        act_layer: Callable[[], nn.Module],
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()
        self.fc1 = ColumnParallelLinear(
            input_dim,
            hidden_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
            disable_tp=use_data_parallel,
        )
        self.activation = act_layer
        self.fc2 = RowParallelLinear(
            hidden_dim,
            input_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
            disable_tp=use_data_parallel,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x = self.activation(x)
        x, _ = self.fc2(x)
        return x


class PerceptionEncoderVisionAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_grid_height: int,
        max_grid_width: int,
        use_cls_token: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.total_num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        use_data_parallel = is_vit_use_data_parallel()
        tp_size = 1 if use_data_parallel else get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_size == 0, (
            "embed_dim must be divisible by num_heads"
        )
        self.num_heads = self.total_num_heads // tp_size

        self.qkv_proj = QKVParallelLinear(
            embed_dim,
            self.head_dim,
            self.total_num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            disable_tp=use_data_parallel,
        )
        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
            disable_tp=use_data_parallel,
        )
        self.attn = MMEncoderAttention(
            self.num_heads,
            self.head_dim,
            self.scale,
            prefix=f"{prefix}.attn",
        )
        self.rope = PerceptionEncoderRope2D(
            dim=self.head_dim,
            max_grid_height=max_grid_height,
            max_grid_width=max_grid_width,
            use_cls_token=use_cls_token,
        )

    def forward(self, x: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.chunk(chunks=3, dim=-1)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k = self.rope(q, k, grid_hw=grid_hw)
        q = q.permute(0, 2, 1, 3).reshape(bsz, seq_len, self.num_heads * self.head_dim)
        k = k.permute(0, 2, 1, 3).reshape(bsz, seq_len, self.num_heads * self.head_dim)

        attn_output = self.attn(q, k, v)
        attn_output, _ = self.out_proj(attn_output)
        return attn_output


class PerceptionEncoderVisionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        max_grid_height: int,
        max_grid_width: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        use_cls_token: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.attn = PerceptionEncoderVisionAttention(
            d_model,
            n_head,
            max_grid_height=max_grid_height,
            max_grid_width=max_grid_width,
            use_cls_token=use_cls_token,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.ls_1 = (
            PerceptionEncoderLayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )
        self.ls_2 = (
            PerceptionEncoderLayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )
        self.ln_1 = norm_layer(d_model)
        self.ln_2 = norm_layer(d_model)
        hidden_dim = int(d_model * mlp_ratio)
        self.mlp = PerceptionEncoderMLP(
            d_model,
            hidden_dim,
            act_layer,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(self, x: torch.Tensor, grid_hw: tuple[int, int]):
        x = x + self.ls_1(self.attn(self.ln_1(x), grid_hw=grid_hw))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class PerceptionEncoderVisionTransformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        max_grid_height: int,
        max_grid_width: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        use_cls_token: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                PerceptionEncoderVisionBlock(
                    d_model=width,
                    n_head=heads,
                    max_grid_height=max_grid_height,
                    max_grid_width=max_grid_width,
                    mlp_ratio=mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    use_cls_token=use_cls_token,
                    quant_config=quant_config,
                    prefix=f"{prefix}.resblocks.{i}",
                )
                for i in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, grid_hw: tuple[int, int]):
        for block in self.resblocks:
            x = block(x, grid_hw=grid_hw)
        return x


class PerceptionEncoder(nn.Module):
    def __init__(
        self,
        config,
        act_layer: Callable,
        norm_layer: Callable = _DEFAULT_NORM_LAYER,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.patch_size = config.patch_size

        self.output_dim = config.output_dim or config.width
        self.heads = config.heads
        self.width = config.width
        self.layers = config.layers

        self.use_abs_posemb = config.use_abs_posemb
        self.use_cls_token = config.use_cls_token
        self.use_rope2d = config.use_rope2d
        if not self.use_rope2d:
            raise ValueError("use_rope2d must be True")
        self.image_size = config.image_size

        self.conv1 = Conv2dLayer(
            in_channels=3,
            out_channels=config.width,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )

        self.ln_pre = norm_layer(config.width) if config.use_ln_pre else nn.Identity()
        self.ln_post = norm_layer(self.width) if config.use_ln_post else nn.Identity()

        self.transformer = PerceptionEncoderVisionTransformer(
            config.width,
            config.layers,
            config.heads,
            max_grid_height=self.image_size // self.patch_size,
            max_grid_width=self.image_size // self.patch_size,
            mlp_ratio=config.mlp_ratio,
            ls_init_value=config.ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            use_cls_token=self.use_cls_token,
            quant_config=quant_config,
            prefix=f"{prefix}.transformer",
        )

        self.vit_downsampler1 = Conv2dLayer(
            config.width, config.width * 2, kernel_size=3, stride=2, padding=1
        )
        self.vit_downsampler2 = Conv2dLayer(
            config.width * 2, config.width * 4, kernel_size=3, stride=2, padding=1
        )

        if self.use_cls_token:
            self.class_embedding = nn.Parameter(
                (self.width**-0.5) * torch.randn(self.width)
            )

        if self.use_abs_posemb:
            self.posemb_grid_size = self.image_size // self.patch_size
            self.positional_embedding = nn.Parameter(
                (self.width**-0.5)
                * torch.randn(
                    int(self.use_cls_token) + self.posemb_grid_size**2,
                    self.width,
                )
            )

    def sample_abs_posemb(self, grid_h: int, grid_w: int):
        if self.posemb_grid_size == grid_h and self.posemb_grid_size == grid_w:
            return self.positional_embedding[None, ...]

        pos_embed = self.positional_embedding
        if self.use_cls_token:
            cls_token_embed, pos_embed = pos_embed[:1], pos_embed[1:]

        pos_embed = (
            pos_embed.reshape(1, self.posemb_grid_size, self.posemb_grid_size, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        pos_embed = F.interpolate(
            pos_embed, size=(grid_h, grid_w), mode="bilinear", align_corners=False
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, self.width)

        if self.use_cls_token:
            pos_embed = torch.cat([cls_token_embed, pos_embed], dim=0)

        return pos_embed[None, ...]

    def forward_features(self, x: torch.Tensor):
        batch, _, h, w = x.shape
        grid_h, grid_w = h // self.patch_size, w // self.patch_size

        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).reshape(batch, -1, self.width)

        if self.use_cls_token:
            x = torch.cat(
                [self.class_embedding.view(1, 1, -1).expand(batch, -1, -1), x], dim=1
            )

        if self.use_abs_posemb:
            x = x + self.sample_abs_posemb(grid_h, grid_w)

        x = self.ln_pre(x)
        x = self.transformer(x, grid_hw=(grid_h, grid_w))
        x = self.ln_post(x)

        if self.use_cls_token:
            x = x[:, 1:, :]

        return x

    def forward(self, x: torch.Tensor):
        x = self.forward_features(x)
        B, P, C = x.shape
        T = int(P**0.5)
        x = x.transpose(2, 1).contiguous()
        x = x.view(B, C, T, T)

        x = self.vit_downsampler1(x)
        x = self.vit_downsampler2(x)

        B, C, T, T = x.shape
        return x.view(B, -1, T * T).transpose(1, 2)


class StepVLForConditionalGeneration(Step3VLForConditionalGeneration):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        },
        orig_to_new_substr={
            ".attn.in_proj_weight": ".attn.qkv_proj.weight",
            ".attn.in_proj_bias": ".attn.qkv_proj.bias",
            ".mlp.c_fc": ".mlp.fc1",
            ".mlp.c_proj": ".mlp.fc2",
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super(Step3VLForConditionalGeneration, self).__init__()

        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_model = PerceptionEncoder(
                config.vision_config,
                get_act_fn(config.vision_config.hidden_act),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "vision_model"),
            )
            self.vit_large_projector = ColumnParallelLinear(
                config.vision_config.width * 4,
                config.text_config.hidden_size,
                bias=config.projector_bias,
                gather_output=True,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "vit_large_projector"),
                disable_tp=self.use_data_parallel,
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _get_vision_model_output(
        self, input_tensor: torch.Tensor | None
    ) -> torch.Tensor | None:
        if input_tensor is None:
            return None
        if self.use_data_parallel:
            return run_dp_sharded_vision_model(input_tensor, self.vision_model)
        return self.vision_model(input_tensor)

    def _process_image_features(self, image_features: torch.Tensor) -> torch.Tensor:
        image_features, _ = self.vit_large_projector(image_features)
        return image_features
