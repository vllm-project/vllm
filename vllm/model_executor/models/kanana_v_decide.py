# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kanana-2-v-decide: Qwen3-VL whose vision attention has a per-head sigmoid
output gate, served as an embedding model (last token, normalized).
Structured decisions read it with KananaVDecideStrategy."""

from functools import cached_property
from typing import TYPE_CHECKING

import einops
import torch
from transformers.models.qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen3_vl import Qwen3VLProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    Qwen3VLVideoProcessor,
)

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.offloader import get_offloader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .adapters import as_embedding_model
from .qwen2_5_vl import Qwen2_5_VisionAttention
from .qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from .utils import WeightsMapper, maybe_prefix
from .vision import is_vit_use_data_parallel


class KananaVDecideVisionAttention(Qwen2_5_VisionAttention):
    """Vision attention whose output is multiplied by sigmoid(gate). The
    checkpoint's qkv packs [query | gate] per head, then key and value."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            projection_size=embed_dim,
            quant_config=quant_config,
            prefix=prefix,
        )
        # Twice the query heads: each head's query, then its gate.
        self.qkv = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.hidden_size_per_attention_head,
            total_num_heads=2 * num_heads,
            total_num_kv_heads=num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
            disable_tp=is_vit_use_data_parallel(),
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        x, _ = self.qkv(x)
        seq_len, batch_size, _ = x.shape
        heads = self.num_attention_heads_per_partition
        size = heads * self.hidden_size_per_attention_head
        query_gate, k, v = x.split([2 * size, size, size], dim=-1)
        q, gate = einops.rearrange(
            query_gate, "s b (h two d) -> two b s h d", two=2, h=heads
        ).unbind(0)
        k = einops.rearrange(k, "s b (h d) -> b s h d", h=heads)
        v = einops.rearrange(v, "s b (h d) -> b s h d", h=heads)
        if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            qk = self.apply_rotary_emb(
                torch.cat([q, k]).contiguous(), rotary_pos_emb_cos, rotary_pos_emb_sin
            )
            q, k = qk.chunk(2)
        context = self.attn(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        context = context * torch.sigmoid(gate)
        context = einops.rearrange(context, "b s h d -> s b (h d)").contiguous()
        output, _ = self.proj(context)
        return output


def gate_vision_attention(
    visual: Qwen3_VisionTransformer,
    vision_config,
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> None:
    """Swap the gated attention into every block of a Qwen3-VL vision tower."""
    if not vision_config.gated_attention:
        raise ValueError("Kanana-2-v-decide checkpoints use gated vision attention")
    if visual.attn_backend == AttentionBackendEnum.FLASHINFER:
        # FlashInfer reads q, k and v as views of a three-part qkv; the
        # gated qkv has four parts.
        raise ValueError(
            "Kanana-2-v-decide does not support the FLASHINFER vision attention"
        )
    offloader = get_offloader()
    for i, block in enumerate(visual.blocks):
        block.attn = KananaVDecideVisionAttention(
            embed_dim=vision_config.hidden_size,
            num_heads=vision_config.num_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.blocks.{i}.attn",
        )
        if offloader.supports_tower_offload:
            # The tower went through the offloader when Qwen3-VL built it.
            offloader.wrap_modules(
                (m for m in [block.attn]), prefix=f"{prefix}.blocks.{i}.attn"
            )


class KananaVDecideProcessingInfo(Qwen3VLProcessingInfo):
    """The checkpoint names no processor class, so the Qwen3-VL processor is
    built from its preprocessor config."""

    @cached_property
    def _image_options(self) -> dict[str, object]:
        model_config = self.ctx.model_config
        options = (
            get_hf_file_to_dict(
                "preprocessor_config.json",
                model_config.model,
                revision=model_config.revision,
            )
            or {}
        )
        keys = (
            "min_pixels",
            "max_pixels",
            "patch_size",
            "merge_size",
            "temporal_patch_size",
            "image_mean",
            "image_std",
        )
        return {k: options[k] for k in keys if k in options}

    @cached_property
    def _default_processor(self) -> Qwen3VLProcessor:
        return self._build_processor()

    def _build_processor(self, **kwargs: object) -> Qwen3VLProcessor:
        tokenizer = self.get_tokenizer()
        return self.ctx.init_processor(
            Qwen3VLProcessor,
            image_processor=Qwen2VLImageProcessor(**self._image_options),
            video_processor=Qwen3VLVideoProcessor(),
            tokenizer=tokenizer,
            chat_template=getattr(tokenizer, "chat_template", None),
            **kwargs,
        )

    def get_hf_processor(self, **kwargs: object) -> Qwen3VLProcessor:
        return self._build_processor(**kwargs) if kwargs else self._default_processor

    def get_supported_mm_limits(self):
        return {"image": None}

    def get_supported_mm_processor_kwargs(self) -> dict[str, set[str]]:
        # Qwen3-VL's override adds video kwargs, and this model takes no video.
        return super(Qwen3VLProcessingInfo, self).get_supported_mm_processor_kwargs()


if TYPE_CHECKING:
    _KananaVDecideBase = Qwen3VLForConditionalGeneration
else:
    # An embedding model even when the runner is left to auto.
    _KananaVDecideBase = as_embedding_model(Qwen3VLForConditionalGeneration)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=KananaVDecideProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class KananaVDecideModel(_KananaVDecideBase):
    # The checkpoint's training-time similarity temperature is unused: the
    # config's decision_temperature scales the cosines.
    hf_to_vllm_mapper = Qwen3VLForConditionalGeneration.hf_to_vllm_mapper | (
        WeightsMapper(orig_to_new_prefix={"similarity.": None})
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = self.config
        if config.embedding_dimension != config.text_config.hidden_size:
            raise ValueError(
                "embedding_dimension must equal the text hidden size, "
                f"got {config.embedding_dimension}"
            )
        if isinstance(self.visual, Qwen3_VisionTransformer):
            gate_vision_attention(
                self.visual,
                config.vision_config,
                vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )
