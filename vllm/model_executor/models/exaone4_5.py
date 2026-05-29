# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

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
"""Inference-only EXAONE-4.5 model compatible with HuggingFace weights."""

from collections.abc import Callable, Iterable
from functools import partial

import einops
import torch
import torch.nn as nn
from transformers.models.exaone4_5 import (
    Exaone4_5_Config,
    Exaone4_5_Processor,
)
from transformers.models.exaone4_5.configuration_exaone4_5 import Exaone4_5_VisionConfig

from vllm.compilation.decorators import (
    should_torch_compile_mm_encoder,
    support_torch_compile,
)
from vllm.config import VllmConfig
from vllm.distributed import parallel_state
from vllm.distributed import utils as dist_utils
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding.common import (
    ApplyRotaryEmb,
)
from vllm.model_executor.models.exaone4 import Exaone4GatedMLP as Exaone4_5_VisionMLP
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VisionTransformer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from .qwen2_vl import Qwen2VLDummyInputsBuilder as Exaone4_5_DummyInputsBuilder
from .qwen2_vl import Qwen2VLMultiModalProcessor as Exaone4_5_MultiModalProcessor
from .utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix

logger = init_logger(__name__)


# === Vision Encoder === #


class EXAONE4_5_VisionAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        projection_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        # Per attention head and per partition values.
        self.tp_size = (
            1
            if use_data_parallel
            else parallel_state.get_tensor_model_parallel_world_size()
        )
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads
        )
        self.num_attention_heads_per_partition = dist_utils.divide(
            num_heads, self.tp_size
        )

        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        self.num_heads = num_heads // self.tp_size
        self.num_kv_heads = max(1, num_kv_heads // self.tp_size)

        self.head_dim = embed_dim // num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.hidden_size_per_attention_head,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
            disable_tp=use_data_parallel,
        )

        self.proj = RowParallelLinear(
            input_size=projection_size,
            output_size=embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
            disable_tp=use_data_parallel,
        )

        self.attn = MMEncoderAttention(
            num_heads=self.num_attention_heads_per_partition,
            head_size=self.hidden_size_per_attention_head,
            num_kv_heads=self.num_kv_heads,
            scale=self.hidden_size_per_attention_head**-0.5,
            prefix=f"{prefix}.attn",
        )

        self.apply_rotary_emb = ApplyRotaryEmb(enforce_enable=True)

    def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # qkv: [s, b, (h + 2*hk) * d]
        s, b, _ = qkv.shape
        h = self.num_heads
        hk = self.num_kv_heads
        d = self.head_dim

        qkv = qkv.view(s, b, h + 2 * hk, d)

        q = qkv[:, :, :h, :]
        k = qkv[:, :, h : h + hk, :]
        v = qkv[:, :, h + hk :, :]

        # [s, b, h, d] -> [b, s, h, d]
        return (
            q.permute(1, 0, 2, 3).contiguous(),
            k.permute(1, 0, 2, 3).contiguous(),
            v.permute(1, 0, 2, 3).contiguous(),
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        # [s, b, c] --> [s, b, head * 3 * head_dim]
        x, _ = self.qkv(x)
        seq_len, batch_size, _ = x.shape

        q, k, v = self.split_qkv(x)
        q = self.apply_rotary_emb(
            q,
            rotary_pos_emb_cos,
            rotary_pos_emb_sin,
        )

        k = self.apply_rotary_emb(
            k,
            rotary_pos_emb_cos,
            rotary_pos_emb_sin,
        )

        context_layer = self.attn(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        context_layer = einops.rearrange(
            context_layer, "b s h d -> s b (h d)", b=batch_size
        ).contiguous()

        output, _ = self.proj(context_layer)
        return output


@support_torch_compile(
    dynamic_arg_dims={
        "x": 0,
        "cu_seqlens": 0,
        "rotary_pos_emb_cos": 0,
        "rotary_pos_emb_sin": 0,
    },
    enable_if=should_torch_compile_mm_encoder,
    is_encoder=True,
)
class Exaone4_5_VisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_hidden_dim: int,
        hidden_act: str = "silu",
        norm_layer: Callable[[int], nn.Module] | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = EXAONE4_5_VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            projection_size=dim,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            use_data_parallel=use_data_parallel,
        )
        self.mlp = Exaone4_5_VisionMLP(
            dim,
            mlp_hidden_dim,
            hidden_act=hidden_act,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            use_data_parallel=use_data_parallel,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,  # Only used for Flash Attention
        seqlens: list[int] | None = None,  # Only used for xFormers
    ) -> torch.Tensor:
        x_attn = self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            max_seqlen=max_seqlen,
        )
        x_fused_norm, residual = self.norm2(x, residual=x_attn)
        x = residual + self.mlp(x_fused_norm)
        return x


class EXAONE4_5_VisionTransformer(Qwen2_5_VisionTransformer):
    def __init__(
        self,
        vision_config: Exaone4_5_VisionConfig,
        norm_eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__(
            vision_config=vision_config,
            norm_eps=norm_eps,
            quant_config=quant_config,
            prefix=prefix,
        )
        depth = vision_config.depth
        self.num_kv_heads = vision_config.num_key_value_heads

        norm_layer = partial(RMSNorm, eps=norm_eps)

        self.blocks = nn.ModuleList(
            [
                Exaone4_5_VisionBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    mlp_hidden_dim=vision_config.intermediate_size,
                    hidden_act=vision_config.hidden_act,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                    use_data_parallel=use_data_parallel,
                )
                for layer_idx in range(depth)
            ]
        )


class Exaone4_5_ProcessingInfo(Qwen2VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Exaone4_5_Config)

    def get_hf_processor(self, **kwargs: object) -> Exaone4_5_Processor:
        return self.ctx.get_hf_processor(
            Exaone4_5_Processor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )


@MULTIMODAL_REGISTRY.register_processor(
    Exaone4_5_MultiModalProcessor,
    info=Exaone4_5_ProcessingInfo,
    dummy_inputs=Exaone4_5_DummyInputsBuilder,
)
class Exaone4_5_ForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)

        config: Exaone4_5_Config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.config = config
        self.multimodal_config = multimodal_config
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = EXAONE4_5_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "visual"),
                use_data_parallel=self.use_data_parallel,
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
                hf_config=config.get_text_config(),
                architectures=["Exaone4ForCausalLM"],
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["mtp."]),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<vision><|image_pad|></vision>"
        if modality.startswith("video"):
            return "<vision><|video_pad|></vision>"

        raise ValueError("Only image or video modality is supported")
