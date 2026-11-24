# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash text/vision composition using HYV4 processing and checkpoint layouts."""

from contextlib import nullcontext
from functools import cached_property

from vllm.model_executor.models.utils import WeightsMapper
from vllm.models.hy_v4.nvidia.mm_preprocess import (
    HYV4VLDummyInputsBuilder,
    HYV4VLMultiModalProcessor,
    HYV4VLProcessingInfo,
)
from vllm.models.hy_v4.nvidia.vl_model import HYV4VLForConditionalGeneration
from vllm.models.hy_v4_flash.configs import HYV4FlashVLConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.processor import get_processor_kwargs_keys
from vllm.transformers_utils.processors.hy_v4 import (
    HYV4VLProcessor,
    HYV4VLProcessorKwargs,
)

from .model import HYV4FlashForCausalLM, validate_platform


class HYV4FlashVLProcessingInfo(HYV4VLProcessingInfo):
    def get_hf_config(self) -> HYV4FlashVLConfig:
        return self.ctx.get_hf_config(HYV4FlashVLConfig)

    def _load_processor(self, kwargs) -> HYV4VLProcessor:
        dynamic_keys = get_processor_kwargs_keys(HYV4VLProcessorKwargs)
        options = {
            key: value
            for key, value in self.ctx.get_merged_mm_kwargs(kwargs).items()
            if key not in dynamic_keys
        }
        options["config"] = self.get_hf_config()
        options["tokenizer"] = self.ctx.tokenizer
        return HYV4VLProcessor.from_pretrained(
            self.ctx.model_config.model,
            revision=self.ctx.model_config.revision,
            **options,
        )

    @cached_property
    def _default_processor(self) -> HYV4VLProcessor:
        return self._load_processor({})

    def get_hf_processor(self, **kwargs) -> HYV4VLProcessor:
        return self._load_processor(kwargs) if kwargs else self._default_processor


@MULTIMODAL_REGISTRY.register_processor(
    HYV4VLMultiModalProcessor,
    info=HYV4FlashVLProcessingInfo,
    dummy_inputs=HYV4VLDummyInputsBuilder,
)
class HYV4FlashVLForConditionalGeneration(HYV4VLForConditionalGeneration):
    language_model_architecture = "HYV4FlashForCausalLM"
    requires_raw_input_tokens = True
    supports_pp = False  # type: ignore[assignment]
    packed_modules_mapping = {
        **HYV4FlashForCausalLM.packed_modules_mapping,
        "qkv": ["q_proj", "k_proj", "v_proj"],
    }
    hf_to_vllm_mapper = (
        HYV4VLForConditionalGeneration.hf_to_vllm_mapper
        | WeightsMapper(
            orig_to_new_prefix={
                "language_model.model.model.": "language_model.model.",
                "language_model.model.lm_head": "language_model.lm_head",
            }
        )
    )

    def __init__(self, *, vllm_config, prefix=""):
        validate_platform()
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.token_lookback_depth = self.language_model.token_lookback_depth
        engram_hash = self.language_model.model.engram_hash
        if engram_hash is not None:
            engram_hash.visual_token_ids = (
                self.config.image_token_id,
                self.config.video_token_id,
            )
        for name, value in vars(self.config).items():
            if name.startswith("dfly_"):
                setattr(self.language_model.config, name, value)

    @staticmethod
    def get_model_state_cls():
        return HYV4FlashForCausalLM.get_model_state_cls()

    def load_weights(self, weights):
        with (
            nullcontext()
            if self.model_config.get_multimodal_config().mm_encoder_only
            else self.language_model.weight_loading_session()
        ):
            loaded = super().load_weights(weights)
        required_visual = (
            {f"visual.{name}" for name, _ in self.visual.named_parameters()}
            if self.visual is not None
            else set()
        )
        missing = required_visual - loaded
        if missing:
            raise ValueError(f"Missing Flash vision weights: {sorted(missing)}")
        return loaded

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        lookback_token_ids=None,
        **kwargs,
    ):
        return self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
            lookback_token_ids=lookback_token_ids,
        )
