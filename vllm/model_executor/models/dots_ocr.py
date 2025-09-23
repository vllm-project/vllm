# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Mapping
from typing import Literal, Optional, TypedDict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen2_vl import Qwen2VLProcessor

from vllm.attention.layer import check_upstream_fa_availability
from vllm.config import VllmConfig
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.interfaces import (MultiModalEmbeddings,
                                                   SupportsMultiModal,
                                                   SupportsPP)
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.model_executor.models.qwen2_vl import (Qwen2VLDummyInputsBuilder,
                                                 Qwen2VLMultiModalProcessor,
                                                 Qwen2VLProcessingInfo)
from vllm.model_executor.models.utils import (AutoWeightsLoader, WeightsMapper,
                                              init_vllm_registered_model,
                                              maybe_prefix,
                                              merge_multimodal_embeddings)
from vllm.model_executor.models.vision import get_vit_attn_backend
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalDataDict
from vllm.platforms import _Backend
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.dotsocr import (DotsOCRConfig,
                                                     DotsVisionConfig)

IMAGE_TOKEN = "<|imgpad|>"


class DotsOCRImagePixelInputs(TypedDict):
    type: Literal["pixel_values", "image_grid_thw"]

    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor


class DotsOCRImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds", "image_grid_thw"]
    image_embeds: torch.Tensor
    """Supported types:
    - List[`torch.Tensor`]: A list of tensors holding all images' features.
        Each tensor holds an image's features.
    - `torch.Tensor`: A tensor holding all images' features
        (concatenation of all images' feature tensors).
    Tensor shape: `(num_image_features, hidden_size)`
    - `num_image_features` varies based on
        the number and resolution of the images.
    - `hidden_size` must match the hidden size of language model backbone.
    """

    image_grid_thw: torch.Tensor


DotsOCRImageInputs = Union[DotsOCRImagePixelInputs,
                           DotsOCRImageEmbeddingInputs]


class DotsOCRDummyInputsBuilder(Qwen2VLDummyInputsBuilder):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        return IMAGE_TOKEN * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        target_width, target_height = self.info.get_image_size_with_most_features(  # noqa: E501
        )

        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images),
        }


class DotsOCRProcessingInfo(Qwen2VLProcessingInfo):

    def get_hf_config(self) -> DotsOCRConfig:
        config = self.ctx.get_hf_config()
        if not config.__class__.__name__ == 'DotsOCRConfig':
            raise TypeError(f"Expected DotsOCRConfig, got {type(config)}")

        if hasattr(config, "vision_config") and isinstance(
                config.vision_config, dict):
            config.vision_config = DotsVisionConfig(**config.vision_config)

        return config

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        max_image_tokens = self.get_max_image_tokens()
        return {"image": max_image_tokens}

    def get_hf_processor(
        self,
        **kwargs: object,
    ) -> Qwen2VLProcessor:
        self.get_tokenizer(
        ).image_token = IMAGE_TOKEN  # Ensure image token is set
        processor = self.ctx.get_hf_processor(
            Qwen2VLProcessor,
            **kwargs,
        )
        processor.image_token = IMAGE_TOKEN
        processor.video_token = "<|video_pad|>"
        return processor


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(tensor: torch.Tensor,
                                freqs: torch.Tensor) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()

    cos = freqs.cos()
    sin = freqs.sin()

    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()

    output = (tensor * cos) + (rotate_half(tensor) * sin)

    output = output.to(orig_dtype)

    return output


class VisionRotaryEmbedding(nn.Module):

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta
                          **(torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen,
                           device=self.inv_freq.device,
                           dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class PatchMerger(nn.Module):

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        pre_norm="layernorm",
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.pre_norm = pre_norm
        if self.pre_norm == "layernorm":
            self.ln_q = LayerNorm(context_dim, eps=1e-6)
        elif self.pre_norm == "rmsnorm":
            self.ln_q = RMSNorm(context_dim, eps=1e-6)
        else:
            print("no norm in patch merger")

        self.mlp = nn.Sequential(
            ColumnParallelLinear(self.hidden_size,
                                 self.hidden_size,
                                 bias=True,
                                 return_bias=False,
                                 disable_tp=True),
            nn.GELU(),
            RowParallelLinear(self.hidden_size,
                              dim,
                              bias=True,
                              return_bias=False,
                              disable_tp=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        else:
            x = self.mlp(x.view(-1, self.hidden_size))
        return x


class DotsVisionAttention(nn.Module):

    def __init__(self,
                 config,
                 dim: int,
                 num_heads: int = 16,
                 bias: bool = True,
                 *,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "") -> None:
        super().__init__()
        from vllm.distributed import (parallel_state,
                                      tensor_model_parallel_all_gather)
        from vllm.distributed import utils as dist_utils

        self.embed_dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.num_heads_per_partition = dist_utils.divide(
            num_heads, self.tp_size)

        # qkv/proj follow Qwen2-VL style; bias controlled by arg
        self.qkv = QKVParallelLinear(hidden_size=dim,
                                     head_size=dim // num_heads,
                                     total_num_heads=num_heads,
                                     bias=bias,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.qkv")
        self.proj = RowParallelLinear(input_size=dim,
                                      output_size=dim,
                                      bias=bias,
                                      quant_config=quant_config,
                                      prefix=f"{prefix}.proj")
        self._all_gather = tensor_model_parallel_all_gather
        self._split_last = dist_utils.split_tensor_along_last_dim

        # Select attention backend
        self.attn_backend = get_vit_attn_backend(self.head_dim,
                                                 torch.get_default_dtype())
        self.use_upstream_fa = False
        if self.attn_backend != _Backend.FLASH_ATTN and \
                check_upstream_fa_availability(torch.get_default_dtype()):
            self.attn_backend = _Backend.FLASH_ATTN
            self.use_upstream_fa = True
        if self.attn_backend not in {
                _Backend.FLASH_ATTN, _Backend.TORCH_SDPA, _Backend.XFORMERS,
                _Backend.ROCM_AITER_FA
        }:
            raise RuntimeError(
                f"Unsupported vision attention backend: {self.attn_backend}")
        self.is_flash_attn_backend = self.attn_backend in {
            _Backend.FLASH_ATTN, _Backend.ROCM_AITER_FA
        }

    def _split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # qkv: [S, B, 3*dim]
        seq_len, bs, _ = qkv.shape
        if self.tp_size > 1:
            qkv = self._all_gather(qkv)
        q, k, v = qkv.chunk(3, dim=2)
        if self.tp_size > 1:
            q = self._split_last(q, num_partitions=self.tp_size)[self.tp_rank]
            k = self._split_last(k, num_partitions=self.tp_size)[self.tp_rank]
            v = self._split_last(v, num_partitions=self.tp_size)[self.tp_rank]
        new_shape = (seq_len, bs, self.num_heads_per_partition, self.head_dim)
        return (q.view(*new_shape), k.view(*new_shape), v.view(*new_shape))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        *,
        max_seqlen: Optional[int] = None,
        seqlens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        # [S, C] -> [S, B=1, C]
        x = hidden_states.unsqueeze(1)
        x, _ = self.qkv(x)
        q, k, v = self._split_qkv(x)
        bs = q.shape[1]
        # [S,B,H,D] -> [B,S,H,D]
        q = q.permute(1, 0, 2, 3).contiguous()
        k = k.permute(1, 0, 2, 3).contiguous()
        v = v.permute(1, 0, 2, 3).contiguous()

        if rotary_pos_emb is not None:
            qk_concat = torch.cat([q, k], dim=0)
            qk_rotated = apply_rotary_pos_emb_vision(qk_concat, rotary_pos_emb)
            q, k = torch.chunk(qk_rotated, 2, dim=0)

        if self.is_flash_attn_backend:
            if self.attn_backend == _Backend.ROCM_AITER_FA:
                from aiter import flash_attn_varlen_func
            else:
                if self.use_upstream_fa:
                    from flash_attn import flash_attn_varlen_func
                else:
                    from vllm.vllm_flash_attn import flash_attn_varlen_func
            q_ = q.reshape(bs * q.shape[1], q.shape[2], q.shape[3])
            k_ = k.reshape(bs * k.shape[1], k.shape[2], k.shape[3])
            v_ = v.reshape(bs * v.shape[1], v.shape[2], v.shape[3])
            output = flash_attn_varlen_func(q_,
                                            k_,
                                            v_,
                                            cu_seqlens_q=cu_seqlens,
                                            cu_seqlens_k=cu_seqlens,
                                            max_seqlen_q=max_seqlen,
                                            max_seqlen_k=max_seqlen,
                                            dropout_p=0.0,
                                            causal=False)
            context_layer = output.view(bs, -1, self.num_heads_per_partition,
                                        self.head_dim)
        elif self.attn_backend == _Backend.TORCH_SDPA:
            outputs = []
            for i in range(1, len(cu_seqlens)):
                s = int(cu_seqlens[i - 1])
                e = int(cu_seqlens[i])
                q_i = q[:, s:e].permute(0, 2, 1, 3)
                k_i = k[:, s:e].permute(0, 2, 1, 3)
                v_i = v[:, s:e].permute(0, 2, 1, 3)
                out_i = F.scaled_dot_product_attention(q_i,
                                                       k_i,
                                                       v_i,
                                                       dropout_p=0.0)
                out_i = out_i.permute(0, 2, 1, 3)
                outputs.append(out_i)
            context_layer = torch.cat(outputs, dim=1) if outputs else q[:, :0]
        elif self.attn_backend == _Backend.XFORMERS:
            from xformers import ops as xops
            from xformers.ops.fmha.attn_bias import BlockDiagonalMask
            attn_bias = BlockDiagonalMask.from_seqlens(q_seqlen=seqlens,
                                                       kv_seqlen=None,
                                                       device=q.device)
            context_layer = xops.memory_efficient_attention_forward(
                q, k, v, attn_bias=attn_bias, p=0, scale=None)
        else:
            raise RuntimeError("Unsupported attention backend")

        # [B,S,H,D] -> [S,B,H*D] -> [S, C]
        context_layer = context_layer.permute(1, 0, 2, 3).contiguous()
        context_layer = context_layer.view(context_layer.shape[0], bs, -1)
        out, _ = self.proj(context_layer)
        return out.squeeze(1)


class DotsSwiGLUFFN(nn.Module):

    def __init__(self,
                 config,
                 *,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        hidden_features = config.intermediate_size
        in_features = config.embed_dim
        bias = config.use_bias

        # Referenced aimv2.py AIMv2SwiGLUFFN
        self.fc13 = MergedColumnParallelLinear(in_features,
                                               [hidden_features] * 2,
                                               bias=bias,
                                               quant_config=quant_config,
                                               prefix=f"{prefix}.fc13",
                                               disable_tp=True)
        self.fc2 = RowParallelLinear(hidden_features,
                                     in_features,
                                     bias=bias,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.fc2",
                                     disable_tp=True)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc13(x)
        x = self.act_fn(x)
        x, _ = self.fc2(x)
        return x

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, w in weights:
            # Map fc1 -> fc13 (shard 0)
            if name.startswith("fc1."):
                tgt = name.replace("fc1.", "fc13.")
                if tgt in params:
                    params[tgt].weight_loader(params[tgt], w, 0)
                    loaded.add(tgt)
                continue
            # Map fc3 -> fc13 (shard 1)
            if name.startswith("fc3."):
                tgt = name.replace("fc3.", "fc13.")
                if tgt in params:
                    params[tgt].weight_loader(params[tgt], w, 1)
                    loaded.add(tgt)
                continue
            # Pass-through for fc2 and others
            if name in params:
                params[name].weight_loader(params[name], w)
                loaded.add(name)
        return loaded


class DotsPatchEmbed(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_channels = config.num_channels
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.embed_dim = config.embed_dim
        self.config = config
        self.proj = nn.Conv2d(
            config.num_channels,
            config.embed_dim,
            kernel_size=(config.patch_size, config.patch_size),
            stride=(config.patch_size, config.patch_size),
        )
        self.norm = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, grid_thw=None) -> torch.Tensor:
        x = x.view(-1, self.num_channels, self.temporal_patch_size,
                   self.patch_size, self.patch_size)[:, :, 0]
        x = self.proj(x).view(-1, self.embed_dim)
        x = self.norm(x)
        return x


class DotsViTPreprocessor(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.patch_h = config.patch_size
        self.patch_w = config.patch_size
        self.embed_dim = config.embed_dim
        self.config = config
        self.patchifier = DotsPatchEmbed(config)

    def forward(self, x: torch.Tensor, grid_thw=None) -> torch.Tensor:
        tokens = self.patchifier(x, grid_thw)
        return tokens


class DotsVisionBlock(nn.Module):

    def __init__(self,
                 config,
                 *,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()

        self.attn = DotsVisionAttention(
            config,
            config.embed_dim,
            num_heads=config.num_attention_heads,
            bias=config.use_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.norm1 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.mlp = DotsSwiGLUFFN(config,
                                 quant_config=quant_config,
                                 prefix=f"{prefix}.mlp")
        self.norm2 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

    def forward(self,
                hidden_states: torch.Tensor,
                *,
                cu_seqlens: torch.Tensor,
                rotary_pos_emb: torch.Tensor,
                max_seqlen: Optional[int] = None,
                seqlens: Optional[list[int]] = None) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            max_seqlen=max_seqlen,
            seqlens=seqlens,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class DotsVisionTransformer(PreTrainedModel):

    def __init__(
        self,
        config: DotsVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        require_post_norm: Optional[bool] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config)
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = DotsViTPreprocessor(config)

        head_dim = config.embed_dim // config.num_attention_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim, dtype=torch.get_default_dtype())
        if self.attn_backend != _Backend.FLASH_ATTN and \
                check_upstream_fa_availability(torch.get_default_dtype()):
            self.attn_backend = _Backend.FLASH_ATTN

        # Keep blocks for compatibility with other vision towers
        num_layers = (config.num_hidden_layers if num_hidden_layers_override
                      is None else num_hidden_layers_override)
        self.blocks = nn.ModuleList([
            DotsVisionBlock(config,
                            quant_config=quant_config,
                            prefix=f"{prefix}.blocks.{i}")
            for i in range(num_layers)
        ])
        if require_post_norm is None:
            require_post_norm = (len(self.blocks) == config.num_hidden_layers)
        if require_post_norm and self.config.post_norm:
            self.post_trunk_norm = RMSNorm(config.embed_dim,
                                           eps=config.rms_norm_eps)
        else:
            self.post_trunk_norm = None

        self.merger = PatchMerger(
            dim=config.hidden_size,
            context_dim=config.embed_dim,
            spatial_merge_size=config.spatial_merge_size,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.patchifier.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.patchifier.proj.weight.device

    def get_pos_ids_by_grid(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(
                torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

        return pos_ids

    def rot_pos_emb(self, grid_thw):
        pos_ids = self.get_pos_ids_by_grid(grid_thw)
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def compute_attn_mask_seqlen(
            self, cu_seqlens: torch.Tensor
    ) -> tuple[Optional[int], Optional[list[int]]]:
        max_seqlen, seqlens = None, None
        if self.attn_backend == _Backend.FLASH_ATTN:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        elif self.attn_backend == _Backend.XFORMERS:
            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        return max_seqlen, seqlens

    def forward(self, hidden_states: torch.Tensor,
                grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(self.dtype)
        hidden_states = self.patch_embed(hidden_states, grid_thw)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
                dim=0,
                dtype=grid_thw.dtype
                if torch.jit.is_tracing() else torch.int32,
            )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        max_seqlen, seqlens = self.compute_attn_mask_seqlen(cu_seqlens)
        for blk in self.blocks:
            hidden_states = blk(hidden_states,
                                cu_seqlens=cu_seqlens,
                                rotary_pos_emb=rotary_pos_emb,
                                max_seqlen=max_seqlen,
                                seqlens=seqlens)

        if self.post_trunk_norm is not None:
            hidden_states = self.post_trunk_norm(hidden_states)

        hidden_states = self.merger(hidden_states)
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2VLMultiModalProcessor,
    info=DotsOCRProcessingInfo,
    dummy_inputs=DotsOCRDummyInputsBuilder,
)
class DotsOCRForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            ".attn.qkv_proj.": ".attn.qkv.",
            ".attn.out_proj.": ".attn.proj.",
        },
        orig_to_new_prefix={
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        },
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return "<|img|><|imgpad|><|endofimg|>"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        self.config: DotsOCRConfig = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.multimodal_config = vllm_config.model_config.multimodal_config

        if isinstance(self.config.vision_config, dict):
            vision_config = DotsVisionConfig(**self.config.vision_config)
            self.config.vision_config = vision_config
        else:
            vision_config = self.config.vision_config

        self.vision_tower = DotsVisionTransformer(
            vision_config,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "vision_tower"),
        )
        self.language_model: Qwen2ForCausalLM = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=self.config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )

    def _validate_and_reshape_mm_tensor(self, mm_input: object,
                                        name: str) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. "
                             f"Got type: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            if mm_input.ndim == 2:
                return mm_input
            if mm_input.ndim != 3:
                raise ValueError(f"{name} should be 2D or batched 3D tensor. "
                                 f"Got ndim: {mm_input.ndim} "
                                 f"(shape={mm_input.shape})")
            return torch.concat(list(mm_input))
        else:
            return torch.concat(mm_input)

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[DotsOCRImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = self._validate_and_reshape_mm_tensor(
                pixel_values, "image pixel values")
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw")

            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return DotsOCRImagePixelInputs(type="pixel_values",
                                           pixel_values=pixel_values,
                                           image_grid_thw=image_grid_thw)

        if image_embeds is not None:
            image_embeds = self._validate_and_reshape_mm_tensor(
                image_embeds, "image embeds")
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw")

            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")
            return DotsOCRImageEmbeddingInputs(type="image_embeds",
                                               image_embeds=image_embeds,
                                               image_grid_thw=image_grid_thw)

    def _process_image_input(
            self, image_input: DotsOCRImageInputs) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(
                self.vision_tower.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(
                self.vision_tower.dtype)
            image_embeds = self.vision_tower(
                pixel_values, grid_thw)[:, :self.config.hidden_size]

        # Split concatenated embeddings for each image item.
        merge_size = self.vision_tower.spatial_merge_size
        sizes = (torch.tensor(grid_thw_list, dtype=torch.long).prod(-1) //
                 (merge_size * merge_size)).tolist()

        return image_embeds.split(sizes)

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []
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
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                self.config.image_token_id,
            )

        return inputs_embeds

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            inputs_embeds = None
        elif inputs_embeds is None and kwargs.get("pixel_values") is not None:
            image_input = self._parse_and_validate_image_input(**kwargs)
            if image_input is None:
                inputs_embeds = None
            else:
                assert input_ids is not None
                inputs_embeds = self.get_multimodal_embeddings(
                    input_ids,
                    image_input=image_input,
                )
                input_ids = None

        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
