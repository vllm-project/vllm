# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Mapping
from functools import cached_property
from typing import Literal, Optional, TypedDict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen2 import Qwen2ForCausalLM
from transformers.models.qwen2_vl import Qwen2VLProcessor

from vllm.config import VllmConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLMultiModalProcessor,
    Qwen2_5_VLProcessingInfo,
)
from vllm.model_executor.models.qwen2_vl import Qwen2VLDummyInputsBuilder
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
    merge_multimodal_embeddings,
)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalDataDict
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.dotsocr import (
    DotsOCRConfig,
    DotsVisionConfig,
)
from vllm.attention.layer import MultiHeadAttention


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


DotsOCRImageInputs = Union[DotsOCRImagePixelInputs, DotsOCRImageEmbeddingInputs]


class DotsOCRDummyInputsBuilder(Qwen2VLDummyInputsBuilder):
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        target_width, target_height = (
            self.info.get_image_size_with_most_features(  # noqa: E501
            )
        )

        return {
            "image": self._get_dummy_images(
                width=target_width, height=target_height, num_images=num_images
            ),
        }


class DotsOCRProcessingInfo(Qwen2_5_VLProcessingInfo):
    def get_hf_config(self) -> DotsOCRConfig:
        config = self.ctx.get_hf_config()
        if not config.__class__.__name__ == "DotsOCRConfig":
            raise TypeError(f"Expected DotsOCRConfig, got {type(config)}")

        if hasattr(config, "vision_config") and isinstance(
            config.vision_config, dict
        ):
            config.vision_config = DotsVisionConfig(**config.vision_config)

        return config

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None, "video": 0}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        max_image_tokens = self.get_max_image_tokens()
        return {"image": max_image_tokens, "video": 0}

    def get_hf_processor(
        self,
        *,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        size: Optional[dict[str, int]] = None,
        **kwargs: object,
    ) -> Qwen2VLProcessor:
        self.get_tokenizer().image_token = (
            "<|imgpad|>"  # Ensure image token is set
        )
        processor = self.ctx.get_hf_processor(
            Qwen2VLProcessor,
            **kwargs,
        )
        processor.image_token = "<|imgpad|>"
        processor.video_token = "<|video_pad|>"
        return processor


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    tensor: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
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
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class PatchMerger(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        pre_norm="layernorm",
        init_merger_std=None,
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
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

        if init_merger_std is not None:
            nn.init.normal_(self.mlp[0].weight, mean=0.0, std=init_merger_std)
            nn.init.zeros_(self.mlp[0].bias)
            nn.init.normal_(self.mlp[2].weight, mean=0.0, std=init_merger_std)
            nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        else:
            x = self.mlp(x.view(-1, self.hidden_size))
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class DotsSwiGLUFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_features = config.intermediate_size
        in_features = config.embed_dim
        bias = config.use_bias

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        self.fc3 = nn.Linear(in_features, hidden_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.fc1(x)) * self.fc3(x)
        x = self.fc2(x)
        return x


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


class VisionSdpaAttention(nn.Module):
    def __init__(
        self, config, dim: int, num_heads: int = 16, bias=True
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 使用 vLLM 的 MultiHeadAttention，它会自动选择最优的后端
        self.attn = MultiHeadAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            scale=self.scale,
        )
        
        # 保留原有的线性层
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        
        # 计算 QKV
        qkv = self.qkv(hidden_states)
        q, k, v = (
            qkv.reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )

        # 应用旋转位置编码
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        # 重新排列维度以匹配 MultiHeadAttention 的期望输入格式
        # MultiHeadAttention 期望 (batch_size, seq_len, hidden_size)
        q = q.reshape(1, seq_length, -1)  # (1, seq_len, hidden_size)
        k = k.reshape(1, seq_length, -1)  # (1, seq_len, hidden_size)  
        v = v.reshape(1, seq_length, -1)  # (1, seq_len, hidden_size)

        # 使用 vLLM 的 MultiHeadAttention，它会自动处理内存优化
        attn_output = self.attn(q, k, v)
        
        # 重新排列回原始格式
        attn_output = attn_output.squeeze(0)  # (seq_len, hidden_size)
        
        # 应用输出投影
        attn_output = self.proj(attn_output)
        return attn_output


class DotsVisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn = VisionSdpaAttention(
            config,
            config.embed_dim,
            num_heads=config.num_attention_heads,
            bias=config.use_bias,
        )
        self.norm1 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.mlp = DotsSwiGLUFFN(config)
        self.norm2 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

    def forward(
        self, hidden_states, cu_seqlens, rotary_pos_emb
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class DotsVisionTransformer(PreTrainedModel):
    def __init__(self, config: DotsVisionConfig) -> None:
        super().__init__(config)
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = DotsViTPreprocessor(config)
        self._init_weights(self.patch_embed.patchifier.proj)

        head_dim = config.embed_dim // config.num_attention_heads

        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        _num_hidden_layers = config.num_hidden_layers
        self.blocks = nn.ModuleList(
            [DotsVisionBlock(config) for _ in range(_num_hidden_layers)]
        )

        if self.config.post_norm:
            self.post_trunk_norm = RMSNorm(
                config.embed_dim, eps=config.rms_norm_eps
            )

        self.merger = PatchMerger(
            dim=config.hidden_size,
            context_dim=config.embed_dim,
            spatial_merge_size=config.spatial_merge_size,
            init_merger_std=self.config.init_merger_std,
        )

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.blocks[0].mlp.fc2.weight.device

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
                torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1)
            )

        return pos_ids

    def rot_pos_emb(self, grid_thw):
        pos_ids = self.get_pos_ids_by_grid(grid_thw)
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, bf16=True
    ) -> torch.Tensor:
        if bf16:
            hidden_states = hidden_states.bfloat16()
        hidden_states = self.patch_embed(hidden_states, grid_thw)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
            )

        if self.config.post_norm:
            hidden_states = self.post_trunk_norm(hidden_states)

        hidden_states = self.merger(hidden_states)
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5_VLMultiModalProcessor,
    info=DotsOCRProcessingInfo,
    dummy_inputs=DotsOCRDummyInputsBuilder,
)
class DotsOCRForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        }
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

        self.vision_tower = DotsVisionTransformer(vision_config)
        self.language_model: Qwen2ForCausalLM = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=self.config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return get_sampler()

    def _validate_and_reshape_mm_tensor(
        self, mm_input: object, name: str
    ) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(
                f"Incorrect type of {name}. Got type: {type(mm_input)}"
            )
        if isinstance(mm_input, torch.Tensor):
            if mm_input.ndim == 2:
                return mm_input
            if mm_input.ndim != 3:
                raise ValueError(
                    f"{name} should be 2D or batched 3D tensor. "
                    f"Got ndim: {mm_input.ndim} "
                    f"(shape={mm_input.shape})"
                )
            return torch.concat(list(mm_input))
        else:
            return torch.concat(mm_input)

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Optional[DotsOCRImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = self._validate_and_reshape_mm_tensor(
                pixel_values, "image pixel values"
            )
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw"
            )

            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError(
                    "Incorrect type of image pixel values. "
                    f"Got type: {type(pixel_values)}"
                )

            return DotsOCRImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        if image_embeds is not None:
            image_embeds = self._validate_and_reshape_mm_tensor(
                image_embeds, "image embeds"
            )
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw"
            )

            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError(
                    "Incorrect type of image embeddings. "
                    f"Got type: {type(image_embeds)}"
                )
            return DotsOCRImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
            )

    # TODO: Use vLLM internal utils to support DP sharding
    def vision_forward(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ):
        from vllm.distributed import (
            get_tensor_model_parallel_group,
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        assert self.vision_tower is not None

        tp_rank = get_tensor_model_parallel_rank()
        tp = get_tensor_model_parallel_world_size()

        image_grid_thw_chunk = image_grid_thw.chunk(tp)
        image_sizes_consum = torch.tensor(
            [i.prod(-1).sum() for i in image_grid_thw_chunk]
        ).cumsum(dim=0)
        merge_size_square = self.vision_tower.config.spatial_merge_size**2
        image_embedding = torch.zeros(
            (
                pixel_values.shape[0] // merge_size_square,
                self.vision_tower.config.hidden_size,
            ),
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )

        if tp_rank < len(image_sizes_consum):
            idx_start = (
                0 if tp_rank == 0 else image_sizes_consum[tp_rank - 1].item()
            )
            idx_end = image_sizes_consum[tp_rank].item()
            pixel_values_part = pixel_values[idx_start:idx_end]
            image_grid_thw_part = image_grid_thw_chunk[tp_rank]
            image_embedding_part = self.vision_tower(
                pixel_values_part, image_grid_thw_part
            )
            image_embedding[
                idx_start // merge_size_square : idx_end // merge_size_square
            ] = image_embedding_part

        group = get_tensor_model_parallel_group().device_group
        torch.distributed.all_reduce(image_embedding, group=group)
        return image_embedding

    def _process_image_input(
        self, image_input: DotsOCRImageInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(
                self.vision_tower.dtype
            )
        else:
            pixel_values = image_input["pixel_values"].type(
                self.vision_tower.dtype
            )
            image_embeds = self.vision_forward(pixel_values, grid_thw)[
                :, : self.config.hidden_size
            ]

        # Split concatenated embeddings for each image item.
        merge_size = self.vision_tower.spatial_merge_size
        sizes = (
            torch.tensor(grid_thw_list, dtype=torch.long).prod(-1)
            // (merge_size * merge_size)
        ).tolist()

        return image_embeds.split(sizes)

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(
        self, **kwargs: object
    ) -> Optional[MultiModalEmbeddings]:
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
        elif inputs_embeds is None and (
            kwargs.get("pixel_values") is not None
            or kwargs.get("image_embeds") is not None
        ):
            image_input = self._parse_and_validate_image_input(**kwargs)
            if image_input is None:
                inputs_embeds = None
            else:
                assert input_ids is not None
                vision_embeddings = self._process_image_input(image_input)
                inputs_embeds = self.get_input_embeddings(
                    input_ids,
                    multimodal_embeddings=vision_embeddings,
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
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(
            hidden_states, sampling_metadata
        )

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
