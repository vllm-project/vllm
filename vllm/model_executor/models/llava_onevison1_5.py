# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Inference-only
LLAVA ONEVISION 1.5 model compatible with HuggingFace weights.
"""
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Any, Callable, Literal, Optional, TypedDict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature
from transformers.models.llava_onevision1_5.configuration_llavaonevision1_5 import (  # noqa: E501
    LlavaOnevision1_5Config, RiceConfig)
from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor
from transformers.models.qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from vllm.config import VllmConfig
from vllm.distributed import parallel_state
from vllm.distributed import utils as dist_utils
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.activation import QuickGELU
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (ImageItem, ModalityData,
                                    MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs)
from vllm.multimodal.parse import (DictEmbeddingItems, ImageSize,
                                   ModalityDataItems, MultiModalDataItems,
                                   MultiModalDataParser)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.platforms import _Backend
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import uses_mrope

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .utils import (AutoWeightsLoader, WeightsMapper,
                    init_vllm_registered_model, maybe_prefix,
                    merge_multimodal_embeddings)
from .vision import get_vit_attn_backend

logger = init_logger(__name__)


def _create_field_factory(
    spatial_merge_size: int
) -> Callable[
    [Mapping[str, torch.Tensor]],
        Mapping[str, MultiModalFieldConfig],
]:

    def _field_config(hf_inputs: Mapping[str, torch.Tensor]):
        image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
        image_pixel_grid_sizes = image_grid_thw.prod(-1)
        image_embed_grid_sizes = (image_pixel_grid_sizes //
                                  spatial_merge_size // spatial_merge_size)
        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", image_pixel_grid_sizes),
            image_embeds=MultiModalFieldConfig.flat_from_sizes(
                "image", image_embed_grid_sizes),
            image_grid_thw=MultiModalFieldConfig.batched("image"),
        )

    return _field_config


class LlavaOnevision1_5_ImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor


class LlavaOnevision1_5_ImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    image_embeds: torch.Tensor
    image_grid_thw: torch.Tensor


LlavaOnevision1_5_ImageInputs = Union[LlavaOnevision1_5_ImagePixelInputs,
                                      LlavaOnevision1_5_ImageEmbeddingInputs]


class LlavaOnevision1_5_VisionRotaryEmbedding(nn.Module):

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


class LlavaOnevision1_5_VisionPatchEmbed(nn.Module):

    def __init__(self,
                 patch_size: int = 14,
                 in_channels: int = 3,
                 embed_dim: int = 1152) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels,
                              embed_dim,
                              kernel_size=(patch_size, patch_size),
                              stride=(patch_size, patch_size),
                              bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.in_channels, self.patch_size, self.patch_size)
        x = self.proj(x).view(-1, self.embed_dim)
        return x


class LlavaOnevision1_5_VisionMLP(nn.Module):

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 bias: bool = False,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = QuickGELU,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "",
                 use_data_parallel: bool = False) -> None:
        if quant_config is not None:
            raise ValueError(
                "LlavaOnevision1_5 is not support quantization for now")
        super().__init__()
        self.act_fn = act_fn()
        mlp_up_proj = (ReplicatedLinear
                       if use_data_parallel else ColumnParallelLinear)
        mlp_down_proj = (ReplicatedLinear
                         if use_data_parallel else RowParallelLinear)
        self.fc1 = mlp_up_proj(
            in_features,
            hidden_features,
            bias=bias,
            quant_config=quant_config,
        )
        self.fc2 = mlp_down_proj(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
        )

    def forward(self, x: torch.tensor) -> torch.Tensor:
        x1, _ = self.fc1(x)
        x2 = self.act_fn(x1)
        x3, _ = self.fc2(x2)
        return x3


class LlavaOnevision1_5_VisionAttn(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        if quant_config is not None:
            raise ValueError(
                "LlavaOnevision1_5 is not support quantization for now")
        self.tp_size = (1 if use_data_parallel else
                        parallel_state.get_tensor_model_parallel_world_size())
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.num_heads = num_heads
        self.hidden_size_per_attn_head = dist_utils.divide(
            projection_size, num_heads)
        self.num_attn_heads_per_partition = dist_utils.divide(
            num_heads, self.tp_size)
        if use_data_parallel:
            self.qkv = ReplicatedLinear(embed_dim,
                                        self.hidden_size_per_attn_head * 3 *
                                        num_heads,
                                        bias=True,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.qkv")
        else:
            self.qkv = QKVParallelLinear(
                hidden_size=embed_dim,
                head_size=self.hidden_size_per_attn_head,
                total_num_heads=num_heads,
                total_num_kv_heads=num_heads,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv")
        if norm_layer is None:
            norm_layer = partial(RMSNorm, eps=1e-6)
        _proj = (ReplicatedLinear if use_data_parallel else RowParallelLinear)
        self.proj = _proj(input_size=projection_size,
                          output_size=embed_dim,
                          quant_config=quant_config,
                          prefix=f"{prefix}.proj")
        self.attn_backend: _Backend = get_vit_attn_backend(support_fa=True)
        if self.attn_backend not in {_Backend.FLASH_ATTN}:
            raise ValueError(
                f"LlavaOnevision1_5 doesn't support {self.attn_backend}.")
        self.is_flash_attn_backend = self.attn_backend == _Backend.FLASH_ATTN

    def _all_gather_tensor(self, local_tensor, hidden_size: int,
                           tp_size: int) -> torch.Tensor:
        import torch.distributed as dist
        gathered_tensors = [
            torch.zeros_like(local_tensor) for _ in range(tp_size)
        ]
        dist.all_gather(gathered_tensors,
                        local_tensor,
                        group=parallel_state.get_tp_group().device_group)

        gathered_tensors_split = [
            torch.split(tensor, hidden_size // tp_size, -1)
            for tensor in gathered_tensors
        ]
        ordered_tensors = [
            tensor for pair in zip(*gathered_tensors_split) for tensor in pair
        ]
        result_tensor = torch.cat(ordered_tensors, dim=-1)
        return result_tensor

    def _split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor]:
        seq_len, _ = qkv.shape
        if self.tp_size > 1:
            qkv = self._all_gather_tensor(qkv, self.qkv.hidden_size,
                                          self.tp_size)
        qkv = qkv.reshape(qkv.shape[0], 1, -1)
        q, k, v = qkv.chunk(3, dim=2)
        if self.tp_size > 1:
            splitter = partial(dist_utils.split_tensor_along_last_dim,
                               num_partitions=self.tp_size)
            q = splitter(q)[self.tp_rank]
            k = splitter(k)[self.tp_rank]
            v = splitter(v)[self.tp_rank]
        new_shape = (seq_len, self.num_attn_heads_per_partition,
                     self.hidden_size_per_attn_head)
        q, k, v = (x.view(*new_shape) for x in (q, k, v))
        return q, k, v

    def _rotate_half(self, x) -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_embed(self,
                                t: torch.Tensor,
                                freqs: torch.Tensor,
                                cu_seqlens=None) -> torch.Tensor:
        origin_dtype = t.dtype
        t = t.float()
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(-2).float()
        sin = emb.sin().unsqueeze(-2).float()
        t = (t * cos) + (self._rotate_half(t) * sin)
        return t.to(origin_dtype)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        x, _ = self.qkv(x)  # [s, b, c]
        q, k, v = self._split_qkv(x)
        seq_len = q.shape[0]
        if rotary_pos_emb is not None:
            q = self._apply_rotary_pos_embed(q, rotary_pos_emb)
            k = self._apply_rotary_pos_embed(k, rotary_pos_emb)
        if self.attn_backend == _Backend.FLASH_ATTN:
            from flash_attn import flash_attn_varlen_func
            output = flash_attn_varlen_func(q,
                                            k,
                                            v,
                                            cu_seqlens_q=cu_seqlens,
                                            cu_seqlens_k=cu_seqlens,
                                            max_seqlen_q=max_seqlen,
                                            max_seqlen_k=max_seqlen)
        else:
            raise ValueError(
                f"LlavaOnevision1_5 doesn't support {self.attn_backend}.")
        output, _ = self.proj(output.reshape(seq_len, -1))
        return output


class LlavaOnevision1_5_VisionTowerBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = QuickGELU,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        if quant_config is not None:
            raise ValueError(
                "LlavaOnevision1_5 is not support quantization for now")
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = LlavaOnevision1_5_VisionAttn(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            use_data_parallel=use_data_parallel)
        self.mlp = LlavaOnevision1_5_VisionMLP(
            dim,
            mlp_hidden_dim,
            act_fn=act_fn,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            use_data_parallel=use_data_parallel)

    def forward(self,
                x: torch.Tensor,
                cu_seqlens: torch.Tensor,
                rotary_pos_emb: torch.Tensor,
                max_seqlen: Optional[int] = None) -> torch.Tensor:
        x_after_attn = self.attn(self.norm1(x),
                                 cu_seqlens=cu_seqlens,
                                 rotary_pos_emb=rotary_pos_emb,
                                 max_seqlen=max_seqlen)
        x += x_after_attn
        x_mlp = self.mlp(self.norm2(x))
        return x + x_mlp


class LlavaOnevision1_5_PatchMerger(nn.Module):

    def __init__(self,
                 d_model: int,
                 context_dim: int,
                 norm_layer: Optional[Callable[[int], nn.Module]] = None,
                 spatial_merge_size: int = 2,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "",
                 use_data_parallel: bool = False) -> None:
        super().__init__()
        if quant_config is not None:
            raise ValueError(
                "LlavaOnevision1_5 is not support quantization for now")
        self.hidden_size = context_dim * (spatial_merge_size**2)
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)
        self.ln_q = norm_layer(context_dim)
        cls_fc1 = (ReplicatedLinear
                   if use_data_parallel else ColumnParallelLinear)
        cls_fc2 = (ReplicatedLinear
                   if use_data_parallel else RowParallelLinear)
        self.mlp = nn.ModuleList([
            cls_fc1(self.hidden_size,
                    self.hidden_size,
                    bias=True,
                    quant_config=quant_config,
                    prefix=f"{prefix}.mlp.0"),
            nn.GELU(),
            cls_fc2(self.hidden_size,
                    d_model,
                    bias=True,
                    quant_config=quant_config,
                    prefix=f"{prefix}.mlp.2")
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x)
        x = x.view(-1, self.hidden_size)
        mlp_fc1, mlp_act, mlp_fc2 = self.mlp
        x, _ = mlp_fc1(x)
        x = mlp_act(x)
        out, _ = mlp_fc2(x)
        return out


class LlavaOnevision1_5_VisionTower(nn.Module):

    def __init__(
        self,
        vision_config: RiceConfig,
        norm_eps: float = 1e-5,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        if quant_config is not None:
            raise ValueError(
                "LlavaOnevision1_5 is not support quantization for now")
        patch_size = vision_config.patch_size
        act_fn = vision_config.hidden_act
        spatial_merge_size = vision_config.spatial_merge_size
        in_channels = vision_config.in_channels
        hidden_size = vision_config.hidden_size
        text_hidden_size = vision_config.text_hidden_size
        embed_dim = vision_config.embed_dim
        depth = vision_config.depth
        num_heads = vision_config.num_heads
        mlp_hidden_dim = vision_config.intermediate_size
        self.spatial_merge_size = spatial_merge_size
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.use_data_parallel = use_data_parallel
        self.act_fn_map = {"gelu": QuickGELU, "torch_gelu": F.gelu}
        if act_fn.lower() not in self.act_fn_map:
            raise ValueError(
                f"LlavaOnevision1_5 Unsupported activation: {act_fn}.")
        self.patch_embed = LlavaOnevision1_5_VisionPatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        head_dim = embed_dim // num_heads
        self.rotary_pos_emb = LlavaOnevision1_5_VisionRotaryEmbedding(
            head_dim // 2)
        self.blocks = nn.ModuleList([
            LlavaOnevision1_5_VisionTowerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                norm_layer=norm_layer,
                quant_config=quant_config,
                prefix=f"{prefix}.blocks.{layer_idx}",
                use_data_parallel=use_data_parallel,
                act_fn=self.act_fn_map[act_fn]) for layer_idx in range(depth)
        ])
        self.merger = LlavaOnevision1_5_PatchMerger(
            d_model=text_hidden_size,
            context_dim=embed_dim,
            norm_layer=norm_layer,
            quant_config=quant_config,
            prefix=f"{prefix}.merger",
            use_data_parallel=use_data_parallel)
        self.attn_backend: _Backend = get_vit_attn_backend(support_fa=True)
        scale = hidden_size**-0.5
        self.class_embedding = nn.Parameter(scale * torch.rand(hidden_size))
        self.class_pos_emb = nn.Parameter(torch.randn(1, head_dim // 2))
        self.pre_layernorm = nn.LayerNorm(hidden_size, norm_eps)

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def _rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            pos_ids.append(
                torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def compute_attn_mask_seqlen(
            self, cu_seqlens: torch.Tensor
    ) -> tuple[Optional[int], Optional[list[int]]]:
        max_seqlen = None
        if self.attn_backend == _Backend.FLASH_ATTN:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        return max_seqlen

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)
        rotary_pos_emb = self._rot_pos_emb(grid_thw)
        img_feats = x.shape[0]
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                             grid_thw[:, 0]).cumsum(
                                                 dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)
        cu = cu_seqlens.to(torch.long)
        num_segments = cu.numel() - 1
        cls_token = self.class_embedding.to(x.dtype).unsqueeze(0)

        total_patches = cu[-1].item()
        new_total = total_patches + num_segments
        D = x.size(-1)
        new_x = x.new_empty((new_total, D))
        new_rotary_pos_emb = rotary_pos_emb.new_empty(
            (new_total, rotary_pos_emb.shape[-1]))
        write_index = 0
        new_cu = [0]
        for i in range(1, num_segments + 1):
            seg_start = cu[i - 1].item()
            seg_end = cu[i].item()
            seg_len = seg_end - seg_start
            new_x[write_index] = cls_token
            new_rotary_pos_emb[write_index] = self.class_pos_emb
            new_x[write_index + 1:write_index + 1 +
                  seg_len] = x[seg_start:seg_end]
            new_rotary_pos_emb[write_index + 1:write_index + 1 +
                               seg_len] = rotary_pos_emb[seg_start:seg_end]
            write_index += 1 + seg_len
            new_cu.append(write_index)
        x = new_x
        cu_seqlens = torch.tensor(new_cu, device=x.device, dtype=torch.int32)
        rotary_pos_emb = new_rotary_pos_emb
        x = self.pre_layernorm(x)

        max_seqlen = self.compute_attn_mask_seqlen(cu_seqlens)
        for blk in self.blocks:
            x = blk(x,
                    cu_seqlens=cu_seqlens,
                    rotary_pos_emb=rotary_pos_emb,
                    max_seqlen=max_seqlen)
        new_x = x.new_empty((img_feats, D))
        for i in range(1, num_segments + 1):
            seg_start = cu[i - 1].item()
            seg_end = cu[i].item()
            new_x[seg_start:seg_end] = x[seg_start + 1:seg_end + 1]
        x = new_x
        return self.merger(x)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class LlavaOnevision1_5_ProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(LlavaOnevision1_5Config)

    def get_hf_processor(self, **kwargs: object) -> Qwen2_5_VLProcessor:
        return self.ctx.get_hf_processor(
            Qwen2_5_VLProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )

    def get_image_processor(self, **kwargs: object) -> Qwen2VLImageProcessor:
        return self.get_hf_processor(**kwargs).image_processor

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        max_image_tokens = self.get_max_image_tokens()
        return {"image": max_image_tokens}

    def _get_vision_info(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int = 1,
        do_resize: bool = True,
        image_processor: Optional[Qwen2VLImageProcessor],
    ) -> tuple[ImageSize, int]:
        if image_processor is None:
            image_processor = self.get_image_processor()

        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        patch_size = vision_config.patch_size
        merge_size = vision_config.spatial_merge_size
        temporal_patch_size = vision_config.temporal_patch_size

        if do_resize:
            resized_height, resized_width = smart_resize(
                height=image_height,
                width=image_width,
                factor=patch_size * merge_size,
                min_pixels=image_processor.min_pixels,
                max_pixels=image_processor.max_pixels,
            )
            preprocessed_size = ImageSize(width=resized_width,
                                          height=resized_height)
        else:
            preprocessed_size = ImageSize(width=image_width,
                                          height=image_height)

        # NOTE: Frames are padded to be divisible by `temporal_patch_size`
        # https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L294
        padded_num_frames = num_frames + num_frames % temporal_patch_size

        grid_t = max(padded_num_frames // temporal_patch_size, 1)
        grid_h = preprocessed_size.height // patch_size
        grid_w = preprocessed_size.width // patch_size
        num_patches = grid_t * grid_h * grid_w
        num_vision_tokens = num_patches // (merge_size**2)
        return preprocessed_size, num_vision_tokens

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        image_processor: Optional[Qwen2VLImageProcessor],
    ) -> int:
        _, num_image_tokens = self._get_vision_info(
            image_width=image_width,
            image_height=image_height,
            image_processor=image_processor,
        )
        return num_image_tokens

    def get_image_size_with_most_features(self) -> ImageSize:
        max_image_size, _ = self._get_vision_info(
            image_width=1800,
            image_height=1800,
            image_processor=None,
        )
        return max_image_size

    def get_max_image_tokens(self) -> int:
        target_width, target_height = self.get_image_size_with_most_features()
        return self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
            image_processor=None,
        )

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}


class LlavaOnevision1_5_DummyInputsBuilder(
        BaseDummyInputsBuilder[LlavaOnevision1_5_ProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        hf_processor = self.info.get_hf_processor()
        image_token: str = hf_processor.image_token
        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        target_width, target_height = \
            self.info.get_image_size_with_most_features()
        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images)
        }


class LlavaOnevision1_5_MultiModalDataParser(MultiModalDataParser):

    def __init__(self, spatial_merge_size: int, *args, **kwargs):
        self._spatial_merge_size = spatial_merge_size
        super().__init__(*args, **kwargs)

    def _parse_image_data(
        self,
        data: Union[dict[str, torch.Tensor], ModalityData[ImageItem]],
    ) -> Optional[ModalityDataItems[Any, Any]]:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="image",
                required_fields={"image_embeds", "image_grid_thw"},
                fields_factory=_create_field_factory(self._spatial_merge_size),
            )
        return super()._parse_image_data(data)


class LlavaOnevision1_5_MultiModalProcessor(
        BaseMultiModalProcessor[LlavaOnevision1_5_ProcessingInfo]):

    def _get_data_parser(self) -> MultiModalDataParser:
        return LlavaOnevision1_5_MultiModalDataParser(
            self.info.get_hf_config().vision_config.spatial_merge_size)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(
            **hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        placeholder = {
            "image": vocab[hf_processor.image_token],
        }
        merge_length = image_processor.merge_size**2

        def get_replacement(item_idx: int, modality: str):
            out_item = out_mm_kwargs[modality][item_idx]
            grid_thw = out_item[f"{modality}_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            num_tokens = int(grid_thw.prod()) // merge_length
            return [placeholder[modality]] * num_tokens

        return [
            PromptReplacement(
                modality=modality,
                target=[placeholder[modality]],
                replacement=partial(get_replacement, modality=modality),
            ) for modality in ("image", )
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _create_field_factory(
            self.info.get_hf_config().vision_config.spatial_merge_size)(
                hf_inputs)


@MULTIMODAL_REGISTRY.register_processor(
    LlavaOnevision1_5_MultiModalProcessor,
    info=LlavaOnevision1_5_ProcessingInfo,
    dummy_inputs=LlavaOnevision1_5_DummyInputsBuilder)
class LlavaOnevision1_5_ForConditionalGeneration(nn.Module, SupportsMultiModal,
                                                 SupportsPP):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # mapping for new names in checkpoint saved after transformers v4.52
            "model.language_model.": "language_model.model.",
            "model.visual.": "visual.",
            # mapping for original checkpoint
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        })

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        raise ValueError("Only image or video modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: LlavaOnevision1_5Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        if multimodal_config.get_limit_per_prompt("image"):
            self.visual = LlavaOnevision1_5_VisionTower(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=self._maybe_ignore_quant_config(quant_config),
                prefix=maybe_prefix(prefix, "visual"),
            )
        else:
            self.visual = None
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen3ForCausalLM"],
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    def _maybe_ignore_quant_config(self, quant_config: QuantizationConfig):
        return None

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
            self, **kwargs: object) -> Optional[LlavaOnevision1_5_ImageInputs]:
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
            return LlavaOnevision1_5_ImagePixelInputs(
                type="pixel_values",
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
            return LlavaOnevision1_5_ImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw)

    def _process_image_input(
        self, image_input: LlavaOnevision1_5_ImageInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"]
        else:
            pixel_values = image_input["pixel_values"]
            image_embeds = self.visual(pixel_values, grid_thw=grid_thw)
        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size
        return image_embeds.split(sizes.tolist())

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}
        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if input_key in ("pixel_values",
                             "image_embeds") and "images" not in modalities:
                modalities["images"] = self._parse_and_validate_image_input(
                    **kwargs)
        return modalities

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(self,
                                  **kwargs: object) -> MultiModalEmbeddings:
        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return []
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()
        for modality in modalities:
            if modality == "images":
                image_input = modalities["images"]
                vision_embeddings = self._process_image_input(image_input)
                multimodal_embeddings += vision_embeddings
        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None \
            and len(multimodal_embeddings) != 0:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.config.image_token_id)
        return inputs_embeds

    def get_input_embeddings_v0(
        self,
        input_ids: torch.Tensor,
        image_input: Optional[LlavaOnevision1_5_ImagePixelInputs] = None
    ) -> torch.Tensor:
        inputs_embeds = self.get_input_embeddings(input_ids)
        if image_input is not None:
            image_embeds = self._process_image_input(image_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                image_embeds,
                placeholder_token_id=self.config.image_token_id,
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner from
        # `get_multimodal_embeddings` and `get_input_embeddings`, this
        # condition is only for v0 compatibility.
        elif inputs_embeds is None:
            image_input = self._parse_and_validate_image_input(**kwargs)
            if image_input is None:
                inputs_embeds = None
            else:
                if uses_mrope(self.config):
                    assert positions.ndim == 2 and positions.size(0) == 3, (
                        "multimodal section rotary embedding requires "
                        f"(3, seq_len) positions, but got {positions.size()}")
                inputs_embeds = self.get_input_embeddings_v0(
                    input_ids, image_input=image_input)
                input_ids = None

        hidden_states = self.language_model.model(
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
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        skip_prefixes = []
        if self.visual is None:
            skip_prefixes.extend(["visual."])
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="visual.merger.",
            tower_model="visual.",
        )
