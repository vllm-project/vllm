# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py
"""Inference-only Deepseek-OCR model compatible with HuggingFace weights."""

import copy
import math
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
    merge_multimodal_embeddings,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargs,
    NestedTensors,
)
from vllm.multimodal.parse import (
    ImageEmbeddingItems,
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.deepseek_vl2 import (
    DeepseekVLV2Config,
)
from vllm.transformers_utils.processors.deepseek_ocr import (
    DeepseekOCRProcessor,
    count_tiles,
)
from vllm.transformers_utils.tokenizer import cached_tokenizer_from_config

# The image token id may be various
_IMAGE_TOKEN = "<image>"
BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
PRINT_NUM_VIS_TOKENS = False
PROMPT = "<image>\n<|grounding|>Convert the document to markdown."


class MlpProjector(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        if cfg.projector_type == "identity":
            modules = nn.Identity()

        elif cfg.projector_type == "linear":
            modules = nn.Linear(cfg.input_dim, cfg.n_embed)

        elif cfg.projector_type == "mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            modules = [nn.Linear(cfg.input_dim, cfg.n_embed)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
            modules = nn.Sequential(*modules)

        elif cfg.projector_type == "normlayer_downsample_mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            mlp_ratio = cfg.get("mlp_ratio", 1)
            modules = [
                nn.LayerNorm(
                    cfg.input_dim * cfg.downsample_ratio * cfg.downsample_ratio
                ),
                nn.Linear(
                    cfg.input_dim * cfg.downsample_ratio * cfg.downsample_ratio,
                    cfg.n_embed * mlp_ratio,
                ),
            ]
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(
                    nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed * mlp_ratio)
                )
            modules.append(nn.GELU())
            modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed))
            modules = nn.Sequential(*modules)

        elif cfg.projector_type == "downsample_mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            mlp_ratio = cfg.get("mlp_ratio", 1)
            modules = [
                nn.Linear(
                    cfg.input_dim * cfg.downsample_ratio * cfg.downsample_ratio,
                    cfg.n_embed * mlp_ratio,
                )
            ]
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(
                    nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed * mlp_ratio)
                )
            modules.append(nn.GELU())
            modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed))
            modules = nn.Sequential(*modules)

        elif cfg.projector_type == "low_high_hybrid_split_mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            self.high_up_proj = nn.Linear(cfg.input_dim, cfg.n_embed // 2)
            self.low_up_proj = nn.Linear(cfg.input_dim, cfg.n_embed // 2)

            modules = []
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
            modules = nn.Sequential(*modules)

        elif cfg.projector_type == "hybrid_split_feature_mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            channel_div = cfg.get("channel_div", 0.5)
            self.high_up_proj = nn.Linear(
                cfg.input_dim[0], int(cfg.n_embed * channel_div)
            )
            self.low_up_proj = nn.Linear(
                cfg.input_dim[1], cfg.n_embed - int(cfg.n_embed * channel_div)
            )

            modules = []
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
            modules = nn.Sequential(*modules)

        elif cfg.projector_type == "low_high_split_mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            modules = []
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed // 2, cfg.n_embed // 2))
            modules = nn.Sequential(*modules)
            self.high_layers = nn.Sequential(*modules)
            self.low_layers = copy.deepcopy(modules)

        else:
            raise ValueError(f"Unknown projector type: {cfg.projector_type}")

        if cfg.get("token_pooling", False):
            self.token_pooling_layer = nn.Linear(cfg.input_dim * 4, cfg.input_dim)

        if cfg.get("conv_fusion_high_low_features", False):
            self.fusion_layer = nn.Linear(cfg.input_dim, cfg.input_dim)
        self.layers = modules

    def forward(self, x):
        if self.cfg.get("token_pooling", False):
            batch_size, wxh, channels = x.shape
            w = h = int(wxh**0.5)
            x = x.view(batch_size, w, h, channels)
            x = x.permute(0, 3, 1, 2)
            # import ipdb; ipdb.set_trace()
            patches = x.unfold(2, 2, 2).unfold(3, 2, 2)
            batch_size, channels, h_patches, w_patches, _, _ = patches.size()
            # 在通道维度上拼接
            patches = patches.contiguous().view(
                batch_size, channels, h_patches * w_patches, -1
            )

            # 通过线性层
            patches = patches.permute(0, 2, 1, 3).contiguous()
            patches = patches.view(batch_size, h_patches * w_patches, channels * 4)

            x = self.token_pooling_layer(patches)

        if self.cfg.get("conv_fusion_high_low_features", False):
            x = self.fusion_layer(x[:, 0]) + x[:, 1]

        if self.cfg.projector_type == "low_high_hybrid_split_mlp_gelu":
            high_x, low_x = x[0], x[1]
            high_x = self.high_up_proj(high_x)
            low_x = self.low_up_proj(low_x)
            x = torch.concat([high_x, low_x], dim=-1)

        if self.cfg.projector_type == "hybrid_split_feature_mlp_gelu":
            high_x = x[..., : self.cfg.input_dim[0]]
            low_x = x[..., self.cfg.input_dim[0] :]
            high_x = self.high_up_proj(high_x)
            low_x = self.low_up_proj(low_x)
            x = torch.concat([high_x, low_x], dim=-1)

        if self.cfg.projector_type == "low_high_split_mlp_gelu":
            high_x, low_x = x[0], x[1]
            high_x = self.high_layers(high_x)
            low_x = self.low_layers(low_x)
            x = torch.concat([high_x, low_x], dim=-1)
            return x

        if (
            self.cfg.projector_type == "downsample_mlp_gelu"
            or self.cfg.projector_type == "normlayer_downsample_mlp_gelu"
        ):
            bs, hw, input_dim = x.shape
            h = w = int((hw) ** 0.5)

            """compute padding"""
            if h % self.cfg.downsample_ratio:
                pad = self.cfg.downsample_ratio - h % self.cfg.downsample_ratio
            else:
                pad = 0
            x = x.reshape(bs, h, w, input_dim)
            if pad > 0:
                x = F.pad(x, (0, 0, 0, pad, 0, pad), "constant", 0)

            """4 to 1 concat"""
            x = x.permute(0, 3, 1, 2)  # B, C, H, W
            x = F.unfold(
                x,
                kernel_size=self.cfg.downsample_ratio,
                stride=self.cfg.downsample_ratio,
                padding=0,
            )  # B, C*4, HW // 4
            x = x.permute(0, 2, 1)

        return self.layers(x)

    @staticmethod
    def get_flops_per_sample(cfg):
        if cfg.projector_type == "linear":
            fwd = 2 * cfg.input_dim * cfg.n_embed

        elif "mlp_gelu" in cfg.projector_type:
            mlp_depth = cfg.get("depth", 1)
            downsample_ratio = cfg.get("downsample_ratio", 1)
            input_dim = (
                sum(cfg.input_dim) if isinstance(cfg.input_dim, list) else cfg.input_dim
            )
            input_dim = input_dim * downsample_ratio * downsample_ratio
            fwd = (
                2 * input_dim * cfg.n_embed
                + (mlp_depth - 1) * 2 * cfg.n_embed * cfg.n_embed
            )
        else:
            fwd = 0

        return fwd * 3


class LayerNormfp32(torch.nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: M
    # return: M, C

    # print(tgt_size)
    # print(abs_pos.shape)
    # exit()
    dim = abs_pos.size(-1)
    # print(dim)
    abs_pos_new = abs_pos.squeeze(0)
    cls_token, old_pos_embed = abs_pos_new[:1], abs_pos_new[1:]

    src_size = int(math.sqrt(abs_pos_new.shape[0] - 1))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        old_pos_embed = (
            old_pos_embed.view(1, src_size, src_size, dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        old_pos_embed = old_pos_embed.to(torch.float32)
        new_pos_embed = F.interpolate(
            old_pos_embed,
            size=(tgt_size, tgt_size),
            mode="bicubic",
            antialias=True,
            align_corners=False,
        ).to(dtype)
        new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
        new_pos_embed = new_pos_embed.view(tgt_size * tgt_size, dim)
        vision_pos_embed = torch.cat([cls_token, new_pos_embed], dim=0)
        vision_pos_embed = vision_pos_embed.view(1, tgt_size * tgt_size + 1, dim)
        return vision_pos_embed
    else:
        return abs_pos


@torch.jit.script
def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, hidden_size=1024, image_size=224, patch_size=14, num_channels=3):
        super().__init__()
        self.embed_dim = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size

        self.class_embedding = torch.nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = torch.nn.Conv2d(
            in_channels=num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = torch.nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids", torch.arange(self.num_positions).expand((1, -1))
        )

    def forward(self, pixel_values, patch_embeds):
        batch_size = pixel_values.shape[0]
        # patch_embeds = self.patch_embedding(
        #     pixel_values
        # )  # shape = [*, width, grid, grid]

        if patch_embeds is not None:
            patch_embeds = patch_embeds
            # print(patch_embeds.shape)
        else:
            patch_embeds = self.patch_embedding(pixel_values)
            # print(111111)
        # shape = [*, width, grid, grid]
        # patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        # x = torch.cat([cls_token, x], dim=1)
        embeddings = embeddings + get_abs_pos(
            self.position_embedding(self.position_ids), embeddings.size(1)
        )
        # embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class NoTPFeedForward(nn.Module):
    def __init__(
        self,
        cfg,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.fc1 = torch.nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = torch.nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x):
        output = self.fc2(quick_gelu(self.fc1(x)))
        return output


class NoTPAttention(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.n_local_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.max_seq_len = cfg.seq_length
        self.use_flash_attention = cfg.use_flash_attn

        self.qkv_proj = torch.nn.Linear(cfg.hidden_size, cfg.hidden_size * 3, bias=True)
        self.out_proj = torch.nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)

        self.attn_drop = cfg.attention_dropout

    def forward(
        self,
        x: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape
        xqkv = self.qkv_proj(x)
        xqkv = xqkv.view(bsz, seqlen, 3, self.num_heads, self.head_dim)

        if self.use_flash_attention:
            from flash_attn import flash_attn_qkvpacked_func

            output = flash_attn_qkvpacked_func(xqkv)
            output = output.view(bsz, seqlen, -1)
        else:
            xq, xk, xv = torch.split(xqkv, 1, dim=2)
            xq = xq.squeeze(2)
            xk = xk.squeeze(2)
            xv = xv.squeeze(2)
            # xq, xk, xv = xqkv[:, :, 0, ...], xqkv[:, :, 1, ...], xqkv[:, :, 2, ...]

            # （B, num_head, S, head_size)
            xq = xq.permute(0, 2, 1, 3)
            xk = xk.permute(0, 2, 1, 3)
            xv = xv.permute(0, 2, 1, 3)
            output = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=None
            )
            output = output.permute(0, 2, 1, 3).reshape(bsz, seqlen, -1)
        output = self.out_proj(output)
        return output


class NoTPTransformerBlock(nn.Module):
    def __init__(self, cfg, layer_id: int, multiple_of=256):
        super().__init__()

        self.n_heads = cfg.num_attention_heads
        self.dim = cfg.hidden_size
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.self_attn = NoTPAttention(cfg)
        self.mlp = NoTPFeedForward(
            cfg, dim=cfg.hidden_size, hidden_dim=cfg.ffn_hidden_size
        )
        self.layer_id = layer_id
        self.layer_norm1 = torch.nn.LayerNorm(
            cfg.hidden_size, eps=cfg.layernorm_epsilon
        )
        self.layer_norm2 = torch.nn.LayerNorm(
            cfg.hidden_size, eps=cfg.layernorm_epsilon
        )

    def forward(self, x: torch.Tensor):
        residual = self.self_attn.forward(self.layer_norm1(x))
        h = x + residual
        out = h + self.mlp.forward(self.layer_norm2(h))
        return out


class NoTPTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        # self.recompute_list = self.cfg.get("recompute_list", [])
        self.num_layers = cfg.num_layers  # _get_num_layers(cfg)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.num_layers):
            self.layers.append(
                NoTPTransformerBlock(
                    cfg,
                    layer_id + 1,
                )
            )

    def forward(
        self,
        hidden_states,
    ):
        for lid, layer in enumerate(self.layers):
            # if lid in self.recompute_list:
            #     def custom(layer_id):
            #         def custom_forward(*args, **kwargs):
            #             x_ = self.layers[layer_id](*args, **kwargs)
            #             return x_

            #         return custom_forward

            #     assert hidden_states.requires_grad == True, logger.warning(
            #         "When using recalculation, the input must have grad fn"
            #     )
            #     hidden_states = tensor_parallel.checkpoint(
            #         custom(lid),
            #         False,
            #         hidden_states.contiguous()
            #     )
            # else:
            hidden_states = layer(hidden_states)

        return hidden_states


class VitModel(nn.Module):
    def __init__(self, cfg, freeze_embed=False, freeze_pre_norm=False) -> None:
        super().__init__()

        self.embeddings = CLIPVisionEmbeddings(
            hidden_size=cfg.hidden_size,
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
        )

        if freeze_embed:
            for name, param in self.embeddings.named_parameters():
                param.requires_grad = False

        self.transformer = NoTPTransformer(cfg=cfg)

        if cfg.get("fp32norm", False):
            self.pre_layrnorm = LayerNormfp32(
                cfg.hidden_size,
                eps=cfg.get("pre_layernorm_epsilon", 1e-5),
            )
        else:
            self.pre_layrnorm = torch.nn.LayerNorm(
                cfg.hidden_size,
                eps=cfg.get("pre_layernorm_epsilon", 1e-5),
            )

        # self.pre_layrnorm = RMSNorm(
        #     cfg.hidden_size,
        #     eps=cfg.get("pre_layernorm_epsilon", 1e-5),
        #     sequence_parallel=False,
        #     use_fp32=True,
        #     use_optimus=True,
        # )

        if freeze_pre_norm:
            for name, param in self.pre_layrnorm.named_parameters():
                param.requires_grad = False

        for p in self.parameters():
            p.micro_dp = True

    def set_input_tensor(self, input_tensor):
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        self.transformer.set_input_tensor(input_tensor[0])

    def __str__(self) -> str:
        return "open_clip"

    def forward(self, x, patch_embeds):
        x = self.embeddings(x, patch_embeds)
        hidden_states = self.pre_layrnorm(x)

        # hidden_states, dis = local_dp_scatter(hidden_states)
        output = self.transformer(hidden_states)

        # output = local_dp_reduce(output, dis)

        return output


vit_model_cfg = dict(
    num_layers=24,
    hidden_size=1024,
    num_heads=16,
    num_attention_heads=16,
    ffn_hidden_size=4096,
    seq_length=256,
    max_position_embeddings=256,
    use_flash_attn=False,
    understand_projector_stride=2,
    hidden_dropout=0.0,
    attention_dropout=0.0,
    no_persist_layer_norm=False,
    layernorm_epsilon=1e-5,
    pre_layernorm_epsilon=1e-5,
    image_size=224,
    patch_size=14,
    recompute_list=[],
)


def build_clip_l():
    return VitModel(
        cfg=vit_model_cfg,
        freeze_embed=False,
        freeze_pre_norm=False,
    )


def get_abs_pos(abs_pos, tgt_size):
    dtype = abs_pos.dtype

    src_size = abs_pos.size(1)

    if src_size != tgt_size:
        old_pos_embed = abs_pos.permute(0, 3, 1, 2)
        old_pos_embed = old_pos_embed.to(torch.float32)
        new_pos_embed = F.interpolate(
            old_pos_embed,
            size=(tgt_size, tgt_size),
            mode="bicubic",
            antialias=True,
            align_corners=False,
        ).to(dtype)
        new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
        return new_pos_embed
    else:
        return abs_pos


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        act_layer: type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, img_size // patch_size, img_size // patch_size, embed_dim
                )
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

        self.net_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.net_3 = nn.Conv2d(
            512, 1024, kernel_size=3, stride=2, padding=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            # x = x + self.pos_embed
            x = x + get_abs_pos(self.pos_embed, x.size(1))

        for blk in self.blocks:
            x = blk(x)

        neck_output = self.neck(x.permute(0, 3, 1, 2))
        conv2_output = self.net_2(neck_output)
        # print(f"conv2_output shape: {conv2_output.shape}")
        conv3_output = self.net_3(conv2_output)

        return conv3_output


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        act_layer: type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(
            embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer
        )

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, (
                "Input size must be provided if using relative positional encoding."
            )
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = (
            self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        rel_h, rel_w = None, None
        if self.use_rel_pos:
            rel_h, rel_w = add_decomposed_rel_pos(
                q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )

        q = q.view(B, self.num_heads, H * W, -1)
        k = k.view(B, self.num_heads, H * W, -1)
        v = v.view(B, self.num_heads, H * W, -1)

        if self.use_rel_pos:
            rel_h = rel_h.view(
                B, self.num_heads, rel_h.size(1), rel_h.size(2), rel_h.size(3)
            )
            rel_w = rel_w.view(
                B, self.num_heads, rel_w.size(1), rel_w.size(2), rel_w.size(3)
            )
            attn_bias = (rel_h + rel_w).view(
                B, self.num_heads, rel_h.size(2), rel_h.size(3) * rel_w.size(4)
            )
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias
            )
            # x = _attention_rel_h_rel_w(q, k, v, rel_h, rel_w)
        else:
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            # qkv = torch.stack([q, k, v], dim=1).transpose(1, 3).reshape(B, H * W, 3, self.num_heads, -1)
            # x = flash_attn_qkvpacked_func(qkv, dropout_p=0.0, causal=False).transpose(1, 2)

        x = (
            x.view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )

        x = self.proj(x)

        return x


def window_partition(
    x: torch.Tensor, window_size: int
) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: tuple[int, int],
    hw: tuple[int, int],
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (tuple): padded height and width (Hp, Wp).
        hw (tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        dtype = rel_pos.dtype
        rel_pos = rel_pos.to(torch.float32)
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        ).to(dtype)
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(
        k_size / q_size, 1.0
    )
    k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(
        q_size / k_size, 1.0
    )
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: tuple[int, int],
    k_size: tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
    rel_h = rel_h.unsqueeze(-1)
    rel_w = rel_w.unsqueeze(-2)
    rel_h = rel_h.reshape(B, q_h * q_w, k_h, 1)
    rel_w = rel_w.reshape(B, q_h * q_w, 1, k_w)

    return rel_h, rel_w


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: tuple[int, int] = (16, 16),
        stride: tuple[int, int] = (16, 16),
        padding: tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (tuple): kernel size of the projection layer.
            stride (tuple): stride of the projection layer.
            padding (tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    image_encoder = ImageEncoderViT(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
    )

    if checkpoint is not None:
        # with open(checkpoint, "rb") as f:
        state_dict = torch.load(checkpoint)
        # print(state_dict.keys())
        # for key in state_dict:
        # image_encoder.load_state_dict({k[14:]: v for k, v in state_dict.items() if 'image_encoder' in k}, strict=False)
        # ocr-anyting
        # image_encoder.load_state_dict(state_dict, strict=True)
        # tob
        image_encoder.load_state_dict(
            {k[30:]: v for k, v in state_dict.items() if "vision_tower_high" in k},
            strict=True,
        )
        print(checkpoint)
    return image_encoder


class DeepseekOCRProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(DeepseekVLV2Config)

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(DeepseekOCRProcessor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_num_image_tokens(
        self, *, image_width: int, image_height: int, cropping: bool = True
    ) -> int:
        hf_processor = self.get_hf_processor()

        # image_size = hf_processor.image_size
        # patch_size = hf_processor.patch_size
        # downsample_ratio = hf_processor.downsample_ratio

        image_size = IMAGE_SIZE
        base_size = BASE_SIZE
        patch_size = 16
        downsample_ratio = 4

        if CROP_MODE:
            if image_width <= 640 and image_height <= 640:
                crop_ratio = [1, 1]
            else:
                # images_crop_raw, crop_ratio = hf_processor.dynamic_preprocess(image)

                # find the closest aspect ratio to the target
                crop_ratio = count_tiles(
                    image_width, image_height, image_size=IMAGE_SIZE
                )

                # print('===========')
                # print('crop_ratio ', crop_ratio)
                # print('============')

            num_width_tiles, num_height_tiles = crop_ratio
        else:
            num_width_tiles = num_height_tiles = 1

        h = w = math.ceil((base_size // patch_size) / downsample_ratio)

        h2 = w2 = math.ceil((image_size // patch_size) / downsample_ratio)

        global_views_tokens = h * (w + 1)
        if num_width_tiles > 1 or num_height_tiles > 1:
            local_views_tokens = (num_height_tiles * h2) * (num_width_tiles * w2 + 1)
        else:
            local_views_tokens = 0

        return global_views_tokens + local_views_tokens + 1

    def get_image_size_with_most_features(self) -> ImageSize:
        if IMAGE_SIZE == 1024 and BASE_SIZE == 1280:
            return ImageSize(width=1024 * 2, height=1024 * 2)
        return ImageSize(width=640 * 2, height=640 * 2)


class DeepseekOCRDummyInputsBuilder(BaseDummyInputsBuilder[DeepseekOCRProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        image_token = processor.image_token

        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        max_image_size = self.info.get_image_size_with_most_features()

        if "<image>" in PROMPT:
            return {
                "image": DeepseekOCRProcessor().tokenize_with_images(
                    images=self._get_dummy_images(
                        width=max_image_size.width,
                        height=max_image_size.height,
                        num_images=num_images,
                    ),
                    bos=True,
                    eos=True,
                    cropping=CROP_MODE,
                )
            }
        else:
            return {"image": []}


class DeepseekOCRMultiModalProcessor(
    BaseMultiModalProcessor[DeepseekOCRProcessingInfo]
):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # print(mm_data)
        if mm_data:
            processed_outputs = self.info.ctx.call_hf_processor(
                self.info.get_hf_processor(**mm_kwargs),
                dict(prompt=prompt, **mm_data),
                mm_kwargs,
            )

        else:
            tokenizer = self.info.get_tokenizer()
            processed_outputs = tokenizer(
                prompt, add_special_tokens=True, return_tensors="pt"
            )

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            images_spatial_crop=MultiModalFieldConfig.batched("image"),
            # image_embeds=MultiModalFieldConfig.batched("image2"),
            images_crop=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        image_token_id = hf_processor.image_token_id
        assert isinstance(image_token_id, int)

        def get_replacement_deepseek_vl2(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )

            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                width = images[0][-1][0][0]
                height = images[0][-1][0][1]

                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=width,
                    image_height=height,
                    # flag = True,
                    cropping=CROP_MODE,
                )
            return [image_token_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement_deepseek_vl2,
            )
        ]

    def _cached_apply_hf_processor(
        self,
        prompt: Union[str, list[int]],
        mm_data_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> tuple[list[int], MultiModalKwargs, bool]:
        # The processor logic is different for len(images) <= 2 vs > 2
        # Since the processing cache assumes that the processor output is
        # invariant of how many images are passed per prompt, we only
        # perform caching for the most common case
        if mm_data_items.get_count("image", strict=False) > 2:
            # This code path corresponds to the cache being disabled
            return self._apply_hf_processor_main(
                prompt=prompt,
                mm_items=mm_data_items,
                hf_processor_mm_kwargs=hf_processor_mm_kwargs,
                enable_hf_prompt_update=True,
            )

        return super()._cached_apply_hf_processor(
            prompt=prompt,
            mm_data_items=mm_data_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
        )


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekOCRMultiModalProcessor,
    info=DeepseekOCRProcessingInfo,
    dummy_inputs=DeepseekOCRDummyInputsBuilder,
)
class DeepseekOCRForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language.": "language_model.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: DeepseekVLV2Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        # config.model_type ='deepseek_vl_v2'

        self.config = config
        self.multimodal_config = multimodal_config

        self.vision_config = config.vision_config
        self.projector_config = config.projector_config
        self.text_config = config.text_config

        model_config = vllm_config.model_config
        tokenizer = cached_tokenizer_from_config(model_config)
        self.image_token_id = tokenizer.vocab[_IMAGE_TOKEN]

        self.sam_model = build_sam_vit_b()
        self.vision_model = build_clip_l()

        n_embed = 1280
        self.projector = MlpProjector(
            dict(projector_type="linear", input_dim=2048, n_embed=n_embed)
        )
        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos

        # self.sam_model = torch.compile(self.sam_model, mode="reduce-overhead")
        # self.vision_model = torch.compile(self.vision_model, mode="reduce-overhead")
        # self.projector = torch.compile(self.projector, mode="max-autotune")

        # special token for image token sequence format
        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        if self.tile_tag == "2D":
            # <|view_separator|>, <|\n|>
            self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
            self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)
        else:
            raise ValueError(
                f"Only 2D tile_tag is supported currently, got: {self.tile_tag}"
            )

        if self.text_config.topk_method == "noaux_tc":
            architectures = ["DeepseekV3ForCausalLM"]
        elif not self.text_config.use_mla:
            architectures = ["DeepseekForCausalLM"]
        else:
            architectures = ["DeepseekV2ForCausalLM"]

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=self.text_config,
            prefix=maybe_prefix(prefix, "language"),
            architectures=architectures,
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_image_input(self, **kwargs: object):
        pixel_values = kwargs.pop("pixel_values", None)
        images_spatial_crop = kwargs.pop("images_spatial_crop", None)
        images_crop = kwargs.pop("images_crop", None)

        if pixel_values is None or torch.sum(pixel_values).item() == 0:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError(
                    f"Incorrect type of pixel values. Got type: {type(pixel_values)}"
                )

            if not isinstance(images_spatial_crop, (torch.Tensor, list)):
                raise ValueError(
                    "Incorrect type of image sizes. "
                    f"Got type: {type(images_spatial_crop)}"
                )

            if not isinstance(images_crop, (torch.Tensor, list)):
                raise ValueError(
                    f"Incorrect type of image crop. Got type: {type(images_crop)}"
                )

            return [pixel_values, images_crop, images_spatial_crop]

        raise AssertionError("This line should be unreachable.")

    def _pixel_values_to_embedding(
        self,
        pixel_values: torch.Tensor,
        images_crop: torch.Tensor,
        images_spatial_crop: torch.Tensor,
    ) -> NestedTensors:
        # Pixel_values (global view): [n_image, batch_size, 3, height, width]
        # images_spatial_crop: [n_image, batch_size, [num_tiles_w, num_tiles_h]]
        # images_crop (local view): [n_image, batch_size, num_pathes, 3, h, w]
        # split the pixel and image_crop, all batch_size = 1

        images_in_this_batch = []

        # print(type(images_crop))

        # print(pixel_values.shape)

        with torch.no_grad():
            for jdx in range(images_spatial_crop.size(0)):
                # with torch.set_grad_enabled(False):
                patches = images_crop[jdx][0].to(torch.bfloat16)  # batch_size = 1
                image_ori = pixel_values[jdx]
                crop_shape = images_spatial_crop[jdx][0]

                if torch.sum(patches).item() != 0:  # if all values = 0, no crop
                    # P, C, H, W = patches.shape
                    # crop_flag = 1
                    local_features_1 = self.sam_model(patches)
                    # TODO del patches
                    # torch.compiler.cudagraph_mark_step_begin()
                    local_features_2 = self.vision_model(patches, local_features_1)

                    local_features = torch.cat(
                        (
                            local_features_2[:, 1:],
                            local_features_1.flatten(2).permute(0, 2, 1),
                        ),
                        dim=-1,
                    )
                    local_features = self.projector(local_features)

                    global_features_1 = self.sam_model(image_ori)
                    global_features_2 = self.vision_model(image_ori, global_features_1)
                    global_features = torch.cat(
                        (
                            global_features_2[:, 1:],
                            global_features_1.flatten(2).permute(0, 2, 1),
                        ),
                        dim=-1,
                    )
                    global_features = self.projector(global_features)

                    if PRINT_NUM_VIS_TOKENS:
                        print("=====================")
                        print("BASE: ", global_features.shape)
                        print("PATCHES: ", local_features.shape)
                        print("=====================")

                    _, hw, n_dim = global_features.shape
                    h = w = int(hw**0.5)

                    _2, hw2, n_dim2 = local_features.shape
                    h2 = w2 = int(hw2**0.5)

                    width_crop_num, height_crop_num = crop_shape[0], crop_shape[1]

                    global_features = global_features.view(h, w, n_dim)

                    global_features = torch.cat(
                        [
                            global_features,
                            self.image_newline[None, None, :].expand(h, 1, n_dim),
                        ],
                        dim=1,
                    )

                    global_features = global_features.view(-1, n_dim)

                    local_features = (
                        local_features.view(
                            height_crop_num, width_crop_num, h2, w2, n_dim2
                        )
                        .permute(0, 2, 1, 3, 4)
                        .reshape(height_crop_num * h2, width_crop_num * w2, n_dim2)
                    )
                    local_features = torch.cat(
                        [
                            local_features,
                            self.image_newline[None, None, :].expand(
                                height_crop_num * h2, 1, n_dim2
                            ),
                        ],
                        dim=1,
                    )
                    local_features = local_features.view(-1, n_dim2)

                    global_local_features = torch.cat(
                        [local_features, global_features, self.view_seperator[None, :]],
                        dim=0,
                    )

                else:
                    global_features_1 = self.sam_model(image_ori)
                    global_features_2 = self.vision_model(image_ori, global_features_1)
                    global_features = torch.cat(
                        (
                            global_features_2[:, 1:],
                            global_features_1.flatten(2).permute(0, 2, 1),
                        ),
                        dim=-1,
                    )
                    global_features = self.projector(global_features)

                    if PRINT_NUM_VIS_TOKENS:
                        print("=====================")
                        print("BASE: ", global_features.shape)
                        print("NO PATCHES")
                        print("=====================")

                    _, hw, n_dim = global_features.shape
                    h = w = int(hw**0.5)

                    global_features = global_features.view(h, w, n_dim)

                    global_features = torch.cat(
                        [
                            global_features,
                            self.image_newline[None, None, :].expand(h, 1, n_dim),
                        ],
                        dim=1,
                    )

                    global_features = global_features.view(-1, n_dim)

                    global_local_features = torch.cat(
                        [global_features, self.view_seperator[None, :]], dim=0
                    )

                images_in_this_batch.append(global_local_features)

        return images_in_this_batch

    def _process_image_input(self, image_input) -> torch.Tensor:
        # image_input: [pixel_values, images_crop, images_spatial_crop]

        pixel_values = image_input[0].to(torch.bfloat16)
        # print(image_input[1][0].shape)
        # print(type(image_input[1]))
        # exit()

        # images_crop = image_input[1].to(torch.bfloat16)
        images_crop = image_input[1]
        # images_crop = image_input[1]
        images_spatial_crop = image_input[2].to(dtype=torch.long)

        # local_start = time.time()
        vision_features = self._pixel_values_to_embedding(
            pixel_values=pixel_values,
            images_crop=images_crop,
            images_spatial_crop=images_spatial_crop,
        )

        # local_total_time = time.time() - local_start

        # print('encoder_time: ', local_total_time)
        # exit()
        return vision_features

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(
        self, **kwargs: object
    ) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)

        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings, self.image_token_id
            )
            # print(len(multimodal_embeddings))
            # print(input_ids.shape)
            # print(type(inputs_embeds))
            # print(inputs_embeds.shape)

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ):
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids, vision_embeddings)
            input_ids = None

        hidden_states = self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states, sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        processed_weights = []

        for name, tensor in weights:
            if (
                "sam_model" in name
                or "vision_model" in name
                or "projector" in name
                or "image_newline" in name
                or "view_seperator" in name
            ):
                new_name = name.replace("model.", "", 1)
            else:
                new_name = "language." + name

            processed_weights.append((new_name, tensor))

        loader = AutoWeightsLoader(self)
        autoloaded_weights = loader.load_weights(
            processed_weights, mapper=self.hf_to_vllm_mapper
        )

        return autoloaded_weights
