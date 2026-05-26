# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch Ovis 2.6 Next model.

Wraps the existing :class:`Qwen3NextForCausalLM` backbone (hybrid Mamba2 /
GatedDeltaNet + Attention + MoE) with the same Ovis vision-language adapter
(SigLIP2-NaViT visual tower + visual tokenizer + visual token embedding)
that :class:`Ovis2_5` already implements for the Ovis 2.5 and Ovis 2.6 Moe
variants.

The implementation is structurally identical to :class:`Ovis2_5`; the only
deltas are:

1. ``__init__`` passes ``config.llm_config`` (the :class:`Qwen3NextConfig`
   under ``sub_configs``) to :func:`init_vllm_registered_model` rather than
   ``config.text_config``, which the Ovis 2.6 Next config classes do not
   define.
2. The class declares the :class:`IsHybrid` interface so vLLM's
   ``HybridAttentionMambaModelConfig.verify_and_update_config`` runs during
   config setup. That hook auto-derives ``cache_config.mamba_block_size``
   from the LLM's ``max_model_len`` and triggers the Mamba-only cache-config
   plumbing. The Mamba state classmethods below delegate to the existing
   :class:`Qwen3NextForCausalLM` implementation — the Ovis wrapper has no
   Mamba layers of its own; only the inner LLM does, so direct delegation
   is correct.

Safetensors weight layout matches the Moe sibling — ``llm.model.layers.*``
for the LLM, ``visual_tokenizer.vit.*`` for SigLIP2,
``visual_tokenizer.head.*`` for the visual head, and ``vte.weight`` for the
visual token embedding — so :class:`AutoWeightsLoader` (inherited via
``Ovis2_5.load_weights``) maps them to the right submodules without
modification.

Tested with ``AIDC-AI/Ovis2.6-80B-A3B``.
"""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import IsHybrid
from vllm.model_executor.models.ovis import VisualEmbedding
from vllm.model_executor.models.ovis2_5 import (
    IMAGE_PAD_TOKEN_ID,
    Ovis2_5,
    Ovis2_5DummyInputsBuilder,
    Ovis2_5MultiModalProcessor,
    Ovis2_5ProcessingInfo,
    VisualTokenizer,
)
from vllm.model_executor.models.qwen3_next import Qwen3NextForCausalLM
from vllm.model_executor.models.utils import (
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

if TYPE_CHECKING:
    from vllm.model_executor.layers.mamba.abstract import MambaStateCopyFunc


@MULTIMODAL_REGISTRY.register_processor(
    Ovis2_5MultiModalProcessor,
    info=Ovis2_5ProcessingInfo,
    dummy_inputs=Ovis2_5DummyInputsBuilder,
)
class Ovis2_6_Next(Ovis2_5, IsHybrid):
    """Ovis 2.6 with Qwen3-Next backbone (hybrid Mamba2/GatedDeltaNet +
    Attention + MoE).

    Inherits all vision processing, prompt-replacement and weight-loading
    logic from :class:`Ovis2_5`. ``__init__`` is overridden to pass
    ``config.llm_config`` (rather than the non-existent
    ``config.text_config``) to :func:`init_vllm_registered_model`; the
    Mamba state classmethods delegate to :class:`Qwen3NextForCausalLM`.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Replicates Ovis2_5.__init__ but passes config.llm_config in place
        # of config.text_config, because Ovis 2.6 Next declares its LLM
        # backbone under sub_configs.llm_config (Qwen3NextConfig).
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config: PretrainedConfig = config

        with self._mark_language_model(vllm_config):
            self.llm = init_vllm_registered_model(
                vllm_config=vllm_config.with_hf_config(config.llm_config),
                prefix=maybe_prefix(prefix, "llm"),
            )

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual_tokenizer = VisualTokenizer(
                config=config.vit_config,
                visual_vocab_size=config.visual_vocab_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual_tokenizer"),
            )
            self.vte = VisualEmbedding(
                config.visual_vocab_size,
                config.hidden_size,
            )

        self.image_pad_token_id: int = IMAGE_PAD_TOKEN_ID

        self.make_empty_intermediate_tensors = (
            self.get_language_model().make_empty_intermediate_tensors
        )

    # IsHybrid protocol: vLLM's HybridAttentionMambaModelConfig consults
    # these classmethods at config-setup time to derive Mamba cache-spec
    # parameters and populate cache_config.mamba_block_size. The Ovis
    # wrapper has no Mamba layers of its own — only the inner Qwen3-Next
    # backbone does — so we delegate to that class's existing impl.
    #
    # Qwen3NextForCausalLM's classmethods read
    # ``vllm_config.model_config.hf_text_config`` for the LLM-side
    # config (linear_num_key_heads, linear_conv_kernel_dim, etc.). For
    # Ovis 2.6 Next, that inner config lives at
    # ``hf_config.llm_config``. We substitute it explicitly via
    # ``vllm_config.with_hf_config(...)`` before delegating, mirroring
    # the pattern this class uses in ``__init__`` for
    # ``init_vllm_registered_model``.
    @classmethod
    def _vllm_config_with_llm(cls, vllm_config: VllmConfig) -> VllmConfig:
        return vllm_config.with_hf_config(vllm_config.model_config.hf_config.llm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return Qwen3NextForCausalLM.get_mamba_state_shape_from_config(
            cls._vllm_config_with_llm(vllm_config)
        )

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return Qwen3NextForCausalLM.get_mamba_state_dtype_from_config(
            cls._vllm_config_with_llm(vllm_config)
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple["MambaStateCopyFunc", "MambaStateCopyFunc"]:
        return Qwen3NextForCausalLM.get_mamba_state_copy_func()
