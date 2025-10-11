# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3-Omni-Moe model (thinker part)."""

from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Any, Callable, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeThinkerConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
)
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)
from transformers.models.whisper import WhisperFeatureExtractor

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen2_audio import (
    Qwen2AudioInputs,
    Qwen2AudioProcessingInfo,
)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargs, NestedTensors
from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    PlaceholderFeaturesInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.platforms import _Backend
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.tokenizer import decode_tokens

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .qwen2_5_omni_thinker import Qwen2_5OmniConditionalGenerationMixin
from .qwen2_5_omni_thinker import (
    Qwen2_5OmniThinkerDummyInputsBuilder as Qwen3OmniMoeThinkerDummyInputsBuilder,
)
from .qwen2_5_omni_thinker import (
    Qwen2_5OmniThinkerMultiModalProcessor,
    Qwen2_5OmniThinkerProcessingInfo,
)
from .qwen2_5_vl import (
    Qwen2_5_VisionAttention,
    Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5_VLProcessingInfo,
)
from .qwen3_moe import Qwen3MoeForCausalLM, Qwen3MoeModel
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
    merge_multimodal_embeddings,
)
from .vision import get_vit_attn_backend

try:
    import flash_attn
except (ImportError, ModuleNotFoundError):
    flash_attn = None

logger = init_logger(__name__)

class Qwen3_VisionPatchEmbed(nn.Module):

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(in_channels,
                              hidden_size,
                              kernel_size=kernel_size,
                              stride=kernel_size,
                              bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L, C = x.shape
        x = x.view(L, -1, self.temporal_patch_size, self.patch_size,
                   self.patch_size)
        x = self.proj(x).view(L, self.hidden_size)
        return x

class Qwen3_VisionMLP(nn.Module):

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 bias: bool = False,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.linear_fc1 = ColumnParallelLinear(in_features,
                                              hidden_features,
                                              bias=bias,
                                              quant_config=quant_config,
                                              return_bias=False,
                                              prefix=f"{prefix}.linear_fc1")
        self.linear_fc2 = RowParallelLinear(hidden_features,
                                           in_features,
                                           bias=bias,
                                           quant_config=quant_config,
                                           return_bias=False,
                                           prefix=f"{prefix}.linear_fc2")
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor):
        mlp_output = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return mlp_output


class Qwen3_VisionBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Qwen2_5_VisionAttention(embed_dim=dim,
                                            num_heads=num_heads,
                                            projection_size=dim,
                                            quant_config=quant_config,
                                            prefix=f"{prefix}.attn")
        self.mlp = Qwen3_VisionMLP(dim,
                                     mlp_hidden_dim,
                                     act_fn=act_fn,
                                     bias=True,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.mlp")

    def forward(
            self,
            x: torch.Tensor,
            cu_seqlens: torch.Tensor,
            rotary_pos_emb: torch.Tensor,
            max_seqlen: Optional[int] = None,  # Only used for Flash Attention
            seqlens: Optional[list[int]] = None,  # Only used for xFormers
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x),
                          cu_seqlens=cu_seqlens,
                          rotary_pos_emb=rotary_pos_emb,
                          max_seqlen=max_seqlen,
                          seqlens=seqlens)

        x = x + self.mlp(self.norm2(x))
        return x


class Qwen3_VisionPatchMerger(nn.Module):

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)

        self.use_postshuffle_norm = use_postshuffle_norm
        if self.use_postshuffle_norm:
            context_dim = self.hidden_size

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.ln_q = norm_layer(self.hidden_size if use_postshuffle_norm else context_dim)
        self.mlp = nn.ModuleList([
            ColumnParallelLinear(self.hidden_size,
                                 self.hidden_size,
                                 bias=True,
                                 quant_config=quant_config,
                                 prefix=f"{prefix}.mlp.0"),
            nn.GELU(),
            RowParallelLinear(self.hidden_size,
                              d_model,
                              bias=True,
                              quant_config=quant_config,
                              prefix=f"{prefix}.mlp.2"),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.ln_q(x.view(-1, self.hidden_size))
        else:
            x = self.ln_q(x).view(-1, self.hidden_size)

        mlp_fc1, mlp_act, mlp_fc2 = self.mlp
        x_parallel, _ = mlp_fc1(x)
        x_parallel = mlp_act(x_parallel)
        out, _ = mlp_fc2(x_parallel)
        return out


class Qwen3_VisionTransformer(nn.Module):

    def __init__(
        self,
        vision_config,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.image_size = vision_config.image_size
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.apply_vit_abs_pos_embed = vision_config.apply_vit_abs_pos_embed
        self.deepstack_visual_indexes = vision_config.deepstack_visual_indexes

        self.patch_embed = Qwen3_VisionPatchEmbed(
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            in_channels=vision_config.in_channels,
            hidden_size=self.hidden_size,
        )

        # vit pos embeding, TODO: spatial_patch_size vs patch_size
        if self.apply_vit_abs_pos_embed:
            self.pos_embed = nn.Embedding((self.image_size // self.patch_size) ** 2, self.hidden_size)
        else:
            self.pos_embed = nn.Parameter(torch.empty([1, (self.image_size // self.patch_size) ** 2, self.hidden_size]))

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([
            Qwen3_VisionBlock(
                dim=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_dim=vision_config.intermediate_size,
                act_fn=_ACTIVATION_REGISTRY[vision_config.hidden_act],
                norm_layer=norm_layer,
                quant_config=quant_config,
                prefix=f"{prefix}.blocks.{layer_idx}")
            for layer_idx in range(vision_config.depth)
        ])
        self.merger = Qwen3_VisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=self.hidden_size,
            norm_layer=norm_layer,
            spatial_merge_size=self.spatial_merge_size,
            quant_config=quant_config,
            prefix=f"{prefix}.merger",
        )
        if self.deepstack_visual_indexes is not None:
            self.merger_list = nn.ModuleList([
                Qwen3_VisionPatchMerger(
                    d_model=vision_config.out_hidden_size,
                    context_dim=self.hidden_size,
                    spatial_merge_size=self.spatial_merge_size,
                    use_postshuffle_norm=True,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=f"{prefix}.merger_list.{layer_idx}"
                ) for layer_idx in range(len(self.deepstack_visual_indexes))
            ])
        
        self.attn_backend: _Backend = get_vit_attn_backend(support_fa=True)

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def rot_pos_emb(self, grid_thw):
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
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def fast_pos_embed_interpolate(self, grid_thw):
        num_grid_per_side = self.image_size // self.patch_size

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        # TODO: use torch instand of np
        for t, h, w in grid_thw:
            h_idxs = np.linspace(0, num_grid_per_side-1, h)
            w_idxs = np.linspace(0, num_grid_per_side-1, w)

            h_idxs_floor = h_idxs.astype(int)
            w_idxs_floor = w_idxs.astype(int)
            h_idxs_ceil = (h_idxs.astype(int) + 1).clip(max=num_grid_per_side-1)
            w_idxs_ceil = (w_idxs.astype(int) + 1).clip(max=num_grid_per_side-1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            idx_list[0].extend(((h_idxs_floor * num_grid_per_side)[None].T + w_idxs_floor[None]).flatten().tolist() * t)
            idx_list[1].extend(((h_idxs_floor * num_grid_per_side)[None].T + w_idxs_ceil[None]).flatten().tolist()  * t)
            idx_list[2].extend(((h_idxs_ceil  * num_grid_per_side)[None].T + w_idxs_floor[None]).flatten().tolist() * t)
            idx_list[3].extend(((h_idxs_ceil  * num_grid_per_side)[None].T + w_idxs_ceil[None]).flatten().tolist()  * t)

            weight_list[0].extend(((1-dh)[None].T * (1-dw)[None]).flatten().tolist() * t)
            weight_list[1].extend(((1-dh)[None].T *   dw[None]  ).flatten().tolist() * t)
            weight_list[2].extend((  dh[None].T   * (1-dw)[None]).flatten().tolist() * t)
            weight_list[3].extend((  dh[None].T   *   dw[None]  ).flatten().tolist() * t)

        device = self.pos_embed.weight.device
        dtype = self.pos_embed.weight.dtype

        p0 = self.pos_embed(torch.tensor(idx_list[0], dtype=torch.long, device=device)) * torch.tensor(weight_list[0], dtype=dtype, device=device)[:, None]
        p1 = self.pos_embed(torch.tensor(idx_list[1], dtype=torch.long, device=device)) * torch.tensor(weight_list[1], dtype=dtype, device=device)[:, None]
        p2 = self.pos_embed(torch.tensor(idx_list[2], dtype=torch.long, device=device)) * torch.tensor(weight_list[2], dtype=dtype, device=device)[:, None]
        p3 = self.pos_embed(torch.tensor(idx_list[3], dtype=torch.long, device=device)) * torch.tensor(weight_list[3], dtype=dtype, device=device)[:, None]

        patch_pos_embeds = p0 + p1 + p2 + p3
        patch_pos_embeds = patch_pos_embeds.split([t*h*w for t, h, w in grid_thw])
        patch_pos_embeds_permute = []
        m_size = self.spatial_merge_size
        for pos_embed, (t, h, w) in zip(patch_pos_embeds, grid_thw):
            pos_embed = pos_embed.view(t, h // m_size, m_size , w // m_size, m_size, -1).permute(0, 1, 3, 2, 4, 5).flatten(0, 4)
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds
    
    def compute_attn_mask_seqlen(
        self,
        cu_seqlens: torch.Tensor,
    ) -> tuple[Optional[int], Optional[list[int]]]:
        max_seqlen, seqlens = None, None
        if self.attn_backend == _Backend.FLASH_ATTN:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        elif self.attn_backend == _Backend.XFORMERS:
            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        return max_seqlen, seqlens

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: list[list[int]],
    ) -> torch.Tensor:
        hidden_states = x.to(device=self.device, dtype=self.dtype)
        hidden_states = self.patch_embed(hidden_states)

        if self.apply_vit_abs_pos_embed:
            pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
            hidden_states = hidden_states + pos_embeds
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        hidden_states = hidden_states.unsqueeze(1)
        rotary_pos_emb = rotary_pos_emb.to(hidden_states.device)
        max_seqlen, seqlens = self.compute_attn_mask_seqlen(cu_seqlens)

        hidden_states_list = []
        deepstack_visual_indexes = self.deepstack_visual_indexes

        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, 
                                cu_seqlens=cu_seqlens, 
                                rotary_pos_emb=rotary_pos_emb,
                                max_seqlen=max_seqlen,
                                seqlens=seqlens
                            )
            if deepstack_visual_indexes is not None and layer_num in deepstack_visual_indexes:
                hidden_states_list.append(hidden_states)

        hidden_states = self.merger(hidden_states)

        # processing deepstack
        if deepstack_visual_indexes is not None:
            processed_hidden_states_list = [hidden_states]
            for idx, x in enumerate(hidden_states_list):
                x = self.merger_list[idx](x)
                processed_hidden_states_list.append(x)
            # we cat the original visual features and deepstack features along the feature dim
            hidden_states = torch.cat(processed_hidden_states_list, dim=1) # [seq_len, hidden_size * (1 + depth_of_deepstack)]

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("attn.qkv.", "attn.q.", "q"),
            ("attn.qkv.", "attn.k.", "k"),
            ("attn.qkv.", "attn.v.", "v"),
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


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    })
class Qwen3MoeLLMModel(Qwen3MoeModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config,
                         prefix=prefix)

        self.deepstack_multiscale_layer_start = 1

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        # args for deepstack
        input_embeds_multiscale = None
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer_idx, layer in enumerate(self.layers[self.start_layer:self.end_layer]):
            layer_idx = layer_idx + self.start_layer

            # process deepstack
            if input_embeds_multiscale is not None and \
                        layer_idx in range(self.deepstack_multiscale_layer_start, self.deepstack_multiscale_layer_start+len(input_embeds_multiscale)):
                input_embeds_multiscale_this = input_embeds_multiscale[layer_idx - self.deepstack_multiscale_layer_start]
                hidden_states = hidden_states + input_embeds_multiscale_this

            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3MoeLLMForCausalLM(Qwen3MoeForCausalLM):
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3MoeForCausalLM, self).__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3MoeLLMModel(vllm_config=vllm_config,
                                   prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      quant_config=quant_config)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

class Qwen3OmniMoeThinkerProcessingInfo(
    Qwen2AudioProcessingInfo,
    Qwen2_5_VLProcessingInfo
):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3OmniMoeConfig).thinker_config

    def get_hf_processor(
        self,
        *,
        sampling_rate: Optional[int] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        size: Optional[dict[str, int]] = None,
        fps: Optional[Union[float, List[float]]] = None,
        **kwargs: object,
    ) -> Qwen3OmniMoeProcessor:
        if fps is not None:
            kwargs["fps"] = fps
        processor = self.ctx.get_hf_processor(
            Qwen3OmniMoeProcessor,
            image_processor=self.get_image_processor(
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                size=size,
            ),
            **kwargs,
        )
        if not hasattr(processor, "audio_token"):
            processor.audio_token = "<|audio_pad|>"
        if not hasattr(processor, "image_token"):
            processor.image_token = "<|image_pad|>"
        if not hasattr(processor, "video_token"):
            processor.video_token = "<|video_pad|>"
        return processor

    def get_feature_extractor(
        self,
        *,
        sampling_rate: Optional[int] = None,
        **kwargs: object,
    ):
        hf_processor = self.get_hf_processor(sampling_rate=sampling_rate)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": None, "image": None, "video": None}




class Qwen3OmniMoeThinkerMultiModalProcessor(
    Qwen2_5OmniThinkerMultiModalProcessor,
):
    
    def _get_feat_extract_output_lengths(
        self, 
        input_lengths: torch.Tensor
    )-> torch.Tensor:
        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        return feat_lengths, output_lengths

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        prompt_ids: list[int],
        mm_kwargs: MultiModalKwargs,
        is_update_applied: bool,
    ) -> tuple[list[int], str, Mapping[str, list[PlaceholderFeaturesInfo]]]:
        """
        Qwen3-Omni reimplements this function to handle `use_audio_in_video`.
        """
        unbound_prompt_updates = self._get_prompt_updates(
            mm_items,
            hf_processor_mm_kwargs,
            mm_kwargs,
        )
        mm_prompt_updates = self._bind_and_group_updates(
            unbound_prompt_updates)

        mm_item_counts = mm_items.get_all_counts()
        self._validate_mm_kwargs(mm_kwargs, mm_item_counts)

        use_audio_in_video = hf_processor_mm_kwargs.get(
            "use_audio_in_video", False)

        if use_audio_in_video:
            if "video" in mm_item_counts:
                assert "audio" in mm_item_counts
                mm_item_counts["audio"] -= mm_item_counts["video"]

        if is_update_applied:
            prompt_ids = self._get_raw_input_ids(prompt_ids, use_audio_in_video)

        (
            prompt_ids,
            prompt,
            mm_placeholders,
        ) = self._apply_prompt_updates(
            prompt_ids,
            mm_prompt_updates,
            mm_item_counts,
        )
        self._validate_mm_placeholders(
            mm_placeholders,
            mm_item_counts)

        tokenizer = self.info.get_tokenizer()
        prompt = decode_tokens(tokenizer, prompt_ids)

        if use_audio_in_video:
            mm_kwargs["use_audio_in_video"] = True

        return prompt_ids, prompt, mm_placeholders

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        image_processor = self.info.get_image_processor(
            **hf_processor_mm_kwargs)
        vocab = tokenizer.get_vocab()

        audio_token = processor.audio_token
        image_token = processor.image_token
        video_token = processor.video_token
        audio_token_id = vocab[audio_token]
        image_token_id = vocab[image_token]
        video_token_id = vocab[video_token]

        audio_feature_lengths = out_mm_kwargs.get("audio_feature_lengths")
        feature_attention_mask = out_mm_kwargs.get("feature_attention_mask")
        if audio_feature_lengths is None and feature_attention_mask is None:
            audio_output_lengths = []
        elif audio_feature_lengths is not None:
            _, audio_output_lens = self._get_feat_extract_output_lengths(
                audio_feature_lengths)
            audio_output_lengths = audio_output_lens.tolist()
        elif feature_attention_mask is not None:
            assert isinstance(feature_attention_mask, torch.Tensor)
            _, audio_output_lens = self._get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1))
            audio_output_lengths = audio_output_lens.tolist()

        # number of audios read from video.
        audio_in_video_item_idx = 0
        audio_item_idx = 0

        def get_replacement_qwen2_audio(item_idx: int):
            nonlocal audio_item_idx
            item_idx += audio_in_video_item_idx

            audio_item_idx += 1

            num_features = audio_output_lengths[item_idx]
            if num_features == 0:
                audios = mm_items.get_items("audio", AudioProcessorItems)
                audio = audios.get(item_idx)
                raise ValueError(
                    f"The audio {audio} (len={len(audio)}) is too short "
                    "to be represented inside the model")

            return [audio_token_id] * num_features

        def get_replacement_qwen2_vision(item_idx: int, modality: str):
            grid_thw = out_mm_kwargs[f"{modality}_grid_thw"][item_idx]
            assert isinstance(grid_thw, torch.Tensor)
            merge_length = image_processor.merge_size**2

            token_id = image_token_id if modality == "image" else video_token_id
            return [token_id] * (int(grid_thw.prod()) // merge_length)

        use_audio_in_video = hf_processor_mm_kwargs.get(
            "use_audio_in_video", False)
        thinker_config = self.info.get_hf_config()

        def get_replacement_qwen2_use_audio_in_video(item_idx: int):
            nonlocal audio_in_video_item_idx
            audio_num_features = audio_output_lengths[audio_item_idx +
                                                      item_idx]
            video_grid_thw = out_mm_kwargs["video_grid_thw"][item_idx]

            audio_in_video_item_idx += 1

            second_per_grid_ts = hf_processor_mm_kwargs.get(
                "second_per_grid_ts", None)
            if second_per_grid_ts:
                video_second_per_grid_t = second_per_grid_ts[item_idx]
            else:
                video_second_per_grid_t = 1.0

            return MRotaryEmbedding.omni3_get_updates_use_audio_in_video(
                thinker_config=thinker_config,
                audio_len=audio_num_features,
                video_grid_thw=video_grid_thw,
                video_second_per_grid_t=video_second_per_grid_t,
            )

        video_replacement_fn = (
            get_replacement_qwen2_use_audio_in_video if use_audio_in_video else
            partial(get_replacement_qwen2_vision, modality="video"))

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_qwen2_audio,
            ),
            PromptReplacement(
                modality="image",
                target=image_token,
                replacement=partial(get_replacement_qwen2_vision,
                                    modality="image"),
            ),
            PromptReplacement(
                modality="video",
                target=video_token,
                replacement=video_replacement_fn,
            ),
        ]

    def _validate_mm_placeholders(
        self,
        mm_placeholders: Mapping[str, list[PlaceholderFeaturesInfo]],
        mm_item_counts: Mapping[str, int],
    ) -> None:
        BaseMultiModalProcessor[Qwen2_5OmniThinkerProcessingInfo]._validate_mm_placeholders(self, mm_placeholders, mm_item_counts)

    def _get_raw_input_ids(
        self,
        token_ids: list[int],
        use_audio_in_video: bool = False,
    ) -> list[int]:
        
        tokenizer = self.info.get_tokenizer()
        vision_bos_token = tokenizer.encode(tokenizer.vision_bos_token)[0]
        vision_eos_token = tokenizer.encode(tokenizer.vision_eos_token)[0]
        audio_bos_token = tokenizer.encode(tokenizer.audio_bos_token)[0]
        audio_eos_token = tokenizer.encode(tokenizer.audio_eos_token)[0]
        audio_token = tokenizer.encode("<|audio_pad|>")[0]
        image_token = tokenizer.encode("<|image_pad|>")[0]
        video_token = tokenizer.encode("<|video_pad|>")[0]

        result = token_ids[:]
        if use_audio_in_video:
            while True:
                start = None
                for i in range(len(result) - 1):
                    if result[i: i + 2] == [vision_bos_token, audio_bos_token]:
                        start = i
                        break
                if start is not None:
                    end = None
                    for i in range(start + 2, len(result) - 1):
                        if result[i: i + 2] == [audio_eos_token, vision_eos_token]:
                            end = i
                            break
                    if end is not None:
                        result = result[:start] + [vision_bos_token, video_token, vision_eos_token] + result[end + 2:]
                else:
                    break

        for mm_token in [audio_token, image_token, video_token]:
            compressed = []
            for x in result:
                if x != mm_token or (not compressed or compressed[-1] != mm_token):
                    compressed.append(x)
            result = compressed
        
        return result


class Qwen3OmniMoeConditionalGenerationMixin(
    Qwen2_5OmniConditionalGenerationMixin
):
    
    def _validate_and_reshape_mm_tensor(self,
                                        mm_input: object,
                                        name: str,
                                        dim: int = 0) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. "
                             f"Got type: {type(mm_input)}")
        if name == "feature_attention_mask":
            dim = -1
        if isinstance(mm_input, torch.Tensor):
            return torch.concat(list(mm_input), dim=dim)
        else:
            if isinstance(mm_input[0], list):
                return torch.concat([torch.concat(mm_input[i], dim=dim) for i in range(len(mm_input))], dim=dim)
            else:
                return torch.concat(mm_input, dim=dim)
    
    def _get_feat_extract_output_lengths(
        self, 
        input_lengths: torch.Tensor
    )-> torch.Tensor:
        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        return output_lengths, output_lengths

    def _process_audio_input(
        self,
        audio_input: Qwen2AudioInputs,
        audio_hashes: list[str] = None,
        cached_audio_features: torch.Tensor = None,
    ) -> torch.Tensor:

        input_features = audio_input["input_features"]
        audio_feature_lengths = audio_input["audio_feature_lengths"]

        if input_features.ndim == 3:
            assert input_features.shape[0] == 1
            input_features = input_features.squeeze(0)

        if not isinstance(audio_feature_lengths, torch.Tensor):
            audio_feature_lengths = torch.cat(audio_feature_lengths) 
        if audio_feature_lengths.ndim == 2:
            audio_feature_lengths = audio_feature_lengths.reshape(
                -1)

        audio_feat_lengths, audio_output_lengths = (
            self._get_feat_extract_output_lengths(
                audio_feature_lengths))

        audio_outputs = self.audio_tower(
            input_features.to(self.audio_tower.dtype),
            feature_lens=audio_feature_lengths,
            aftercnn_lens=audio_feat_lengths,
        )
        audio_features = audio_outputs.last_hidden_state
        return audio_features.split(audio_output_lengths.tolist())


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3OmniMoeThinkerMultiModalProcessor,
    info=Qwen3OmniMoeThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniMoeThinkerDummyInputsBuilder,
)
class Qwen3OmniMoeThinkerForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    Qwen3OmniMoeConditionalGenerationMixin,
):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "thinker.lm_head.": "language_model.lm_head.",
            "thinker.model.": "language_model.model.",
            "thinker.": "",
        }
    )
    
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        if modality.startswith("audio"):
            return f"<|audio_start|><|audio_pad|><|audio_end|>"

        raise ValueError("Only image, video or audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        thinker_config: Qwen3OmniMoeThinkerConfig = (
            vllm_config.model_config.hf_config.thinker_config
        )
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = thinker_config
        self.multimodal_config = multimodal_config

        # force "use_flash_attention_2=True" to audio tower to align
        # the results.
        if flash_attn is not None:
            audio_config = thinker_config.audio_config
            audio_config._attn_implementation_autoset = True
            audio_config._attn_implementation = "flash_attention_2"
        else:
            logger.warning(
                "flash_attn is not available, the model may not yield the "
                "exactly same result as the transformers implementation "
                "in the audio tower part."
            )

        self.audio_tower = Qwen3OmniMoeAudioEncoder(thinker_config.audio_config)

        self.visual = Qwen3_VisionTransformer(
            vision_config=thinker_config.vision_config,
            norm_eps=getattr(thinker_config.text_config, "rms_norm_eps", 1e-6),
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "visual"),
        )
        self.quant_config = quant_config

        self.language_model = Qwen3MoeLLMForCausalLM(
            vllm_config=vllm_config.with_hf_config(thinker_config.text_config,
                                                   architectures=["Qwen3MoeForCausalLM"]),
            prefix=maybe_prefix(prefix, "language_model")
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        self.input_embeds_multiscale = None

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if (
                input_key in ("pixel_values", "image_embeds")
                and "image" not in mm_input_by_modality
            ):
                mm_input_by_modality["image"] = (
                    self._parse_and_validate_image_input(**kwargs)
                )
            if (
                input_key in ("pixel_values_videos", "video_embeds")
                and "video" not in mm_input_by_modality
            ):
                mm_input_by_modality["video"] = (
                    self._parse_and_validate_video_input(**kwargs)
                )
            if (
                input_key in ("input_audio_features")
                and "audio" not in mm_input_by_modality
            ):
                mm_input_by_modality["audio"] = (
                    self._parse_and_validate_audio_input(**kwargs)
                )
        return mm_input_by_modality

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(
        self, **kwargs: object
    ) -> Optional[MultiModalEmbeddings]:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(
            **kwargs
        )
        if not mm_input_by_modality:
            return []

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                vision_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings += vision_embeddings
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                multimodal_embeddings += video_embeddings
            if modality == "audio":
                audio_embeddings = self._process_audio_input(multimodal_input)
                multimodal_embeddings += audio_embeddings
        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        input_embeds_multiscale = None
        if multimodal_embeddings is not None and len(multimodal_embeddings) != 0:
            # TODO (ywang96): support overlapping modalitiy embeddings so that
            # `use_audio_in_video` will work on V1.
            # split the feat dim to obtain multi-scale visual feature
            if self.visual.deepstack_visual_indexes is not None:
                multiscale_len = len(self.visual.deepstack_visual_indexes)
                multimodal_embeddings_multiscale = []
                for index, embeddings in enumerate(multimodal_embeddings):
                    if embeddings.shape[-1] != self.config.text_config.hidden_size:
                        visual_dim = embeddings.shape[-1] // (multiscale_len + 1)
                        main_dim, multi_dim = visual_dim, visual_dim * multiscale_len
                        embeddings_main, embeddings_multiscale = torch.split(
                            embeddings, [main_dim, multi_dim], dim=-1
                        )
                        multimodal_embeddings[index] = embeddings_main
                        multimodal_embeddings_multiscale.append(embeddings_multiscale)
                if len(multimodal_embeddings_multiscale) > 0:
                    input_embeds_multiscale = inputs_embeds.new_zeros(inputs_embeds.size(0), multiscale_len * inputs_embeds.size(1))
                    input_embeds_multiscale = merge_multimodal_embeddings(
                        input_ids,
                        input_embeds_multiscale,
                        multimodal_embeddings_multiscale,
                        placeholder_token_id=[self.config.image_token_id, self.config.video_token_id],
                    )
                    input_embeds_multiscale = input_embeds_multiscale.view(inputs_embeds.shape[0], multiscale_len, visual_dim).permute(1,0,2).contiguous()

            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                [
                    self.config.image_token_id,
                    self.config.video_token_id,
                    self.config.audio_token_id,
                ],
            )
        
        return inputs_embeds, input_embeds_multiscale

    def get_multimodal_embeddings_v0(
        self, **kwargs: object
    ) -> Optional[NestedTensors]:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        image_input = self._parse_and_validate_image_input(**kwargs)
        video_input = self._parse_and_validate_video_input(**kwargs)

        if audio_input is None and image_input is None and video_input is None:
            return None

        multimodal_embeddings: List[Tuple[NestedTensors, str]] = []

        if audio_input is not None:
            audio_embeds = self._process_audio_input(audio_input)
            multimodal_embeddings.append((audio_embeds, "audio"))
        if image_input is not None:
            image_embeds = self._process_image_input(image_input)
            multimodal_embeddings.append((image_embeds, "image"))
        if video_input is not None:
            video_embeds = self._process_video_input(video_input)
            multimodal_embeddings.append((video_embeds, "video"))
        return multimodal_embeddings

    def get_input_embeddings_v0(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds, None

        use_deepstack = self.visual.deepstack_visual_indexes is not None
        if use_deepstack:
            multiscale_len = len(self.visual.deepstack_visual_indexes)
            visual_dim = inputs_embeds.shape[-1]
            input_embeds_multiscale = None
            for embeddings, modality in multimodal_embeddings:
                if modality in ["image", "video"]:
                    input_embeds_multiscale = torch.zeros_like(inputs_embeds).unsqueeze(1).repeat(1, len(self.visual.deepstack_visual_indexes), 1).flatten(1)
                    break

        for embeddings, modality in multimodal_embeddings:
            if modality == "audio":
                placeholder_token_id = self.config.audio_token_id
            if modality == "image":
                placeholder_token_id = self.config.image_token_id
            if modality == "video":
                placeholder_token_id = self.config.video_token_id
            if use_deepstack and modality in ["image", "video"]:
                embeddings = torch.cat(embeddings)
                embeddings, embeddings_multiscale = embeddings.split([visual_dim, visual_dim * multiscale_len], dim=-1)
                input_embeds_multiscale = merge_multimodal_embeddings(
                    input_ids,
                    input_embeds_multiscale,
                    embeddings_multiscale,
                    placeholder_token_id=placeholder_token_id,
                )
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, embeddings, placeholder_token_id
            )

        if use_deepstack and input_embeds_multiscale is not None:
            input_embeds_multiscale = input_embeds_multiscale.view(inputs_embeds.shape[0], multiscale_len, visual_dim).permute(1,0,2).contiguous()
            return inputs_embeds, input_embeds_multiscale

        return inputs_embeds, None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        input_embeds_multiscale: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            multimodal_embeddings = self.get_multimodal_embeddings_v0(**kwargs)
            inputs_embeds, input_embeds_multiscale = self.get_input_embeddings_v0(
                input_ids, multimodal_embeddings
            )
            input_ids = None

        if input_embeds_multiscale is not None:
            assert input_embeds_multiscale.shape[-1] == inputs_embeds.shape[-1]
            
        hidden_states = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
            # args for deepstack
            input_embeds_multiscale=input_embeds_multiscale,
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

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> Set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["talker.", "code2wav."],
        )
        loaded_weights = loader.load_weights(
            weights, mapper=self.hf_to_vllm_mapper
        )

        return loaded_weights
