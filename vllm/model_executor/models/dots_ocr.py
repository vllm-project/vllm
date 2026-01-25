# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Mapping
from typing import Annotated, Literal, TypeAlias

import torch
import torch.nn as nn
from torch.nn import LayerNorm
from transformers.models.qwen2_vl import Qwen2VLProcessor

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import utils as dist_utils
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from vllm.model_executor.layers.conv import Conv2dLayer
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding.common import (
    ApplyRotaryEmb,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.model_executor.models.qwen2_vl import (
    Qwen2VisionAttention,
    Qwen2VLDummyInputsBuilder,
    Qwen2VLMultiModalProcessor,
    Qwen2VLProcessingInfo,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.model_executor.models.vision import get_vit_attn_backend
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalDataDict
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.dotsocr import DotsOCRConfig, DotsVisionConfig
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .vision import is_vit_use_data_parallel, run_dp_sharded_mrope_vision_model

IMAGE_TOKEN = "<|imgpad|>"


class DotsOCRImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - np: The total number of patches over each image over each prompt in
              the batch
        - ni: Number of images
        - cps: Number of channels * patch_size * patch_size
    """

    type: Literal["pixel_values"]

    pixel_values: Annotated[torch.Tensor, TensorShape("np", "cps")]
    image_grid_thw: Annotated[torch.Tensor, TensorShape("ni", 3)]


class DotsOCRImageEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - nf: Number of image features
        - hs: Hidden size
        - ni: Number of images
    """

    type: Literal["image_embeds"]

    image_embeds: Annotated[torch.Tensor, TensorShape("nf", "hs")]
    image_grid_thw: Annotated[torch.Tensor, TensorShape("ni", 3)]


DotsOCRImageInputs: TypeAlias = DotsOCRImagePixelInputs | DotsOCRImageEmbeddingInputs


class DotsOCRDummyInputsBuilder(Qwen2VLDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        return IMAGE_TOKEN * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        target_width, target_height = self.info.get_image_size_with_most_features(  # noqa: E501
        )

        image_overrides = mm_options.get("image") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            ),
        }


class DotsOCRProcessingInfo(Qwen2VLProcessingInfo):
    def get_hf_config(self) -> DotsOCRConfig:
        config = self.ctx.get_hf_config()
        if not config.__class__.__name__ == "DotsOCRConfig":
            raise TypeError(f"Expected DotsOCRConfig, got {type(config)}")

        if hasattr(config, "vision_config") and isinstance(config.vision_config, dict):
            config.vision_config = DotsVisionConfig(**config.vision_config)

        return config

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
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
        self.get_tokenizer().image_token = IMAGE_TOKEN  # Ensure image token is set
        processor = self.ctx.get_hf_processor(
            Qwen2VLProcessor,
            **kwargs,
        )
        processor.image_token = IMAGE_TOKEN
        processor.video_token = "<|video_pad|>"
        return processor


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
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
        prefix: str = "",
    ) -> None:
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.pre_norm = pre_norm
        if self.pre_norm == "layernorm":
            self.ln_q = LayerNorm(context_dim, eps=1e-6)
        elif self.pre_norm == "rmsnorm":
            self.ln_q = RMSNorm(context_dim, eps=1e-6)

        self.mlp = nn.Sequential(
            ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                bias=True,
                return_bias=False,
                prefix=f"{prefix}.0",
                disable_tp=use_data_parallel,
            ),
            nn.GELU(),
            RowParallelLinear(
                self.hidden_size,
                dim,
                bias=True,
                return_bias=False,
                prefix=f"{prefix}.2",
                disable_tp=use_data_parallel,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        else:
            x = self.mlp(x.view(-1, self.hidden_size))
        return x


class DotsVisionAttention(nn.Module):
    def __init__(
        self,
        config,
        dim: int,
        num_heads: int = 16,
        bias: bool = True,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()

        self.embed_dim = dim
        self.tp_size = (
            1 if use_data_parallel else get_tensor_model_parallel_world_size()
        )
        self.tp_rank = 0 if use_data_parallel else get_tensor_model_parallel_rank()
        self.hidden_size_per_attention_head = dist_utils.divide(dim, num_heads)
        self.num_attention_heads_per_partition = dist_utils.divide(
            num_heads, self.tp_size
        )
        # qkv/proj follow Qwen2-VL style; bias controlled by arg
        self.qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=self.hidden_size_per_attention_head,
            total_num_heads=num_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
            disable_tp=use_data_parallel,
        )
        self.proj = RowParallelLinear(
            input_size=dim,
            output_size=dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
            disable_tp=use_data_parallel,
        )

        self.attn = MMEncoderAttention(
            num_heads=self.num_attention_heads_per_partition,
            head_size=self.hidden_size_per_attention_head,
            scale=self.hidden_size_per_attention_head**-0.5,
            prefix=f"{prefix}.attn",
        )

        self.apply_rotary_emb = ApplyRotaryEmb(
            enforce_enable=True,
            enable_fp32_compute=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        *,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # [S, C] -> [S, B=1, C]
        x = hidden_states.unsqueeze(1)
        x, _ = self.qkv(x)
        q, k, v = Qwen2VisionAttention.split_qkv(self, x)
        bs = q.shape[1]
        # [S,B,H,D] -> [B,S,H,D]
        q = q.permute(1, 0, 2, 3).contiguous()
        k = k.permute(1, 0, 2, 3).contiguous()
        v = v.permute(1, 0, 2, 3).contiguous()

        if rotary_pos_emb is not None:
            qk_concat = torch.cat([q, k], dim=0)
            qk_rotated = self.apply_rotary_emb(
                qk_concat,
                rotary_pos_emb.cos(),
                rotary_pos_emb.sin(),
            )
            q, k = torch.chunk(qk_rotated, 2, dim=0)

        context_layer = self.attn(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        # [B,S,H,D] -> [S,B,H*D] -> [S, C]
        context_layer = context_layer.permute(1, 0, 2, 3).contiguous()
        context_layer = context_layer.view(context_layer.shape[0], bs, -1)
        out, _ = self.proj(context_layer)
        return out.squeeze(1)


class DotsSwiGLUFFN(nn.Module):
    def __init__(
        self,
        config,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        hidden_features = config.intermediate_size
        in_features = config.embed_dim
        bias = config.use_bias

        use_data_parallel = is_vit_use_data_parallel()
        # Referenced aimv2.py AIMv2SwiGLUFFN
        self.fc13 = MergedColumnParallelLinear(
            in_features,
            [hidden_features] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc13",
            disable_tp=use_data_parallel,
        )
        self.fc2 = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
            disable_tp=use_data_parallel,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc13(x)
        x = self.act_fn(x)
        x, _ = self.fc2(x)
        return x

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("fc13", "fc1", 0),
            ("fc13", "fc3", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class DotsPatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_channels = config.num_channels
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.embed_dim = config.embed_dim
        self.config = config
        self.proj = Conv2dLayer(
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


class DotsVisionBlock(nn.Module):
    def __init__(
        self,
        config,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
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
        self.mlp = DotsSwiGLUFFN(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.norm2 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            max_seqlen=max_seqlen,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class DotsVisionTransformer(nn.Module):
    def __init__(
        self,
        config: DotsVisionConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        num_hidden_layers_override: int | None = None,
        require_post_norm: bool | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = DotsViTPreprocessor(config)

        head_dim = config.embed_dim // config.num_attention_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim,
            dtype=torch.get_default_dtype(),
        )
        self.out_hidden_size = config.hidden_size
        # Keep blocks for compatibility with other vision towers
        num_layers = (
            config.num_hidden_layers
            if num_hidden_layers_override is None
            else num_hidden_layers_override
        )
        self.blocks = nn.ModuleList(
            [
                DotsVisionBlock(
                    config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{i}",
                )
                for i in range(num_layers)
            ]
        )
        if require_post_norm is None:
            require_post_norm = len(self.blocks) == config.num_hidden_layers
        if require_post_norm and self.config.post_norm:
            self.post_trunk_norm = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
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

    def get_pos_ids_by_grid(self, grid_thw: list[list[int]]) -> list[torch.Tensor]:
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
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

        return pos_ids

    def rot_pos_emb(self, grid_thw: list[list[int]]) -> torch.Tensor:
        pos_ids = self.get_pos_ids_by_grid(grid_thw)
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = max(max(h, w) for _, h, w in grid_thw)
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def compute_attn_mask_seqlen(self, cu_seqlens: torch.Tensor) -> int | None:
        max_seqlen = None
        if (
            self.attn_backend == AttentionBackendEnum.FLASH_ATTN
            or self.attn_backend == AttentionBackendEnum.ROCM_AITER_FA
        ):
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        return max_seqlen

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: list[list[int]]
    ) -> torch.Tensor:
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # Convert grid_thw to tensor (always expecting list format now)
        grid_thw = torch.tensor(grid_thw, device=hidden_states.device, dtype=torch.long)
        hidden_states = hidden_states.to(self.dtype)
        hidden_states = self.patch_embed(hidden_states, grid_thw)

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])

        max_seqlen = self.compute_attn_mask_seqlen(cu_seqlens)
        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                max_seqlen=max_seqlen,
            )

        if self.post_trunk_norm is not None:
            hidden_states = self.post_trunk_norm(hidden_states)

        hidden_states = self.merger(hidden_states)
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2VLMultiModalProcessor,
    info=DotsOCRProcessingInfo,
    dummy_inputs=DotsOCRDummyInputsBuilder,
)
class DotsOCRForCausalLM(nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA):
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

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
        ".attn.qkv": [".attn.qkv"],
        "fc13": ["fc1", "fc3"],
    }
    supports_encoder_tp_data = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|img|><|imgpad|><|endofimg|>"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        self.config: DotsOCRConfig = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        if isinstance(self.config.vision_config, dict):
            vision_config = DotsVisionConfig(**self.config.vision_config)
            self.config.vision_config = vision_config
        else:
            vision_config = self.config.vision_config

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_tower = DotsVisionTransformer(
                vision_config,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model: Qwen2ForCausalLM = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=self.config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Qwen2ForCausalLM"],
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> DotsOCRImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            return DotsOCRImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        if image_embeds is not None:
            return DotsOCRImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
            )

    def _process_image_input(
        self, image_input: DotsOCRImageInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.vision_tower.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(self.vision_tower.dtype)

            if self.use_data_parallel:
                return run_dp_sharded_mrope_vision_model(
                    self.vision_tower,
                    pixel_values,
                    grid_thw_list,
                    rope_type="rope_3d",
                )
            else:
                image_embeds = self.vision_tower(pixel_values, grid_thw_list)[
                    :, : self.config.hidden_size
                ]

        # Split concatenated embeddings for each image item.
        merge_size = self.vision_tower.spatial_merge_size
        sizes = (
            torch.tensor(grid_thw_list, dtype=torch.long).prod(-1)
            // (merge_size * merge_size)
        ).tolist()

        return image_embeds.split(sizes)

    def get_num_mm_encoder_tokens(self, num_image_tokens: int) -> int:
        merge_size = self.vision_tower.spatial_merge_size
        return num_image_tokens * (merge_size**2)

    def get_num_mm_connector_tokens(self, num_vision_tokens: int) -> int:
        merge_size = self.vision_tower.spatial_merge_size
        return num_vision_tokens // (merge_size**2)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

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
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="vision_tower.merger",
            tower_model="vision_tower.",
        )
