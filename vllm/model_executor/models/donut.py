# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal, Optional, Union

import torch
import torch.nn as nn
from transformers import BatchFeature, NougatProcessor

from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.bart import BartParallelLMHead, MBartDecoder
from vllm.model_executor.models.interfaces import (MultiModalEmbeddings,
                                                   SupportsMultiModal,
                                                   SupportsV0Only)
from vllm.model_executor.models.swin import SwinModel
from vllm.model_executor.models.utils import (AutoWeightsLoader,
                                              _flatten_embeddings, flatten_bn)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargsItems)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (BaseProcessingInfo,
                                        EncDecMultiModalProcessor,
                                        PromptIndexTargets, PromptInsertion,
                                        PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.utils.tensor_schema import TensorSchema, TensorShape


class MBartDecoderWrapper(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.decoder = MBartDecoder(config,
                                    cache_config,
                                    quant_config=quant_config,
                                    prefix=f"{prefix}.decoder")

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)


class DonutLanguageForConditionalGeneration(nn.Module, SupportsV0Only):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config

        self.config = config
        self.model = MBartDecoderWrapper(vllm_config=vllm_config,
                                         prefix=f"{prefix}.model")
        embed_scale = math.sqrt(
            config.d_model) if config.scale_embedding else 1.0

        self.vocab_size = config.vocab_size
        self.lm_head = BartParallelLMHead(self.vocab_size,
                                          config.d_model,
                                          embed_scale=embed_scale)

        self.logits_processor = LogitsProcessor(self.vocab_size,
                                                config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            input_ids
                torch.Tensor of *decoder* input token ids.
            positions
                torch.Tensor of *decoder* position indices.
        Returns:
            Output torch.Tensor
        """

        return self.model(decoder_input_ids=input_ids,
                          decoder_positions=positions,
                          encoder_hidden_states=inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
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
                if "final_logits_bias" in name:
                    continue
                # if self.config.tie_word_embeddings and "embed_tokens" in name:
                #     continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class DonutImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - b: Batch size
        - c: Number of channels (3)
        - h: Height
        - w: Width
    """
    type: Literal["pixel_values"]
    data: Annotated[torch.Tensor, TensorShape("b", 3, "h", "w")]


class DonutProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self):
        return self.ctx.get_hf_processor()

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": 1}

    def get_num_image_tokens(self) -> int:
        return 1


class DonutDummyInputsBuilder(BaseDummyInputsBuilder[DonutProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        target_width, target_height = self.info.get_hf_config(
        ).encoder.image_size

        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images)
        }


class DonutMultiModalProcessor(EncDecMultiModalProcessor[DonutProcessingInfo]):

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def create_encoder_prompt(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
    ) -> Union[str, list[int]]:
        return prompt

    def create_decoder_prompt(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
    ) -> Union[str, list[int]]:
        return prompt

    @property
    def pad_dummy_encoder_prompt(self) -> bool:
        return True

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        hf_processor = self.info.get_hf_processor()
        if mm_data:
            processed_outputs = super()._call_hf_processor(
                prompt, mm_data, mm_kwargs, tok_kwargs)
            if isinstance(hf_processor, NougatProcessor):
                processed_outputs["input_ids"] = processed_outputs["labels"]
        else:
            tokenizer = hf_processor.tokenizer
            processed_outputs = tokenizer(prompt,
                                          add_special_tokens=False,
                                          return_tensors="pt")
        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(pixel_values=MultiModalFieldConfig.batched("image"))

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor()
        tokenizer = hf_processor.tokenizer
        pad_token_id = tokenizer.pad_token_id
        num_image_tokens = self.info.get_num_image_tokens()
        image_tokens = [pad_token_id] * num_image_tokens

        return [
            PromptInsertion(
                modality="image",
                target=PromptIndexTargets.start(),
                insertion=image_tokens,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(DonutMultiModalProcessor,
                                        info=DonutProcessingInfo,
                                        dummy_inputs=DonutDummyInputsBuilder)
class DonutForConditionalGeneration(nn.Module, SupportsMultiModal,
                                    SupportsV0Only):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        processor_config = vllm_config.model_config.hf_image_processor_config

        self.config = config
        self.vision_config = config.encoder
        self.processor_config = processor_config
        self.encoder = SwinModel(config=config.encoder)

        self.decoder = DonutLanguageForConditionalGeneration(
            vllm_config=vllm_config.with_hf_config(config.decoder),
            prefix=f"{prefix}.decoder",
        )
        self.pad_token_id = config.pad_token_id

    def _parse_and_validate_image_input(self, **kwargs: object):
        pixel_values: Optional[Union[list[list[torch.Tensor]],
                                     list[torch.Tensor],
                                     torch.Tensor]] = kwargs.pop(
                                         "pixel_values", None)
        image_embeds: Optional[Union[list[list[torch.Tensor]],
                                     list[torch.Tensor],
                                     torch.Tensor]] = kwargs.pop(
                                         "image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None and image_embeds is not None:
            raise ValueError(
                "Both pixel values and image embeds are provided.")

        if pixel_values is not None:
            h, w = self.config.encoder.image_size
            return DonutImagePixelInputs(type="pixel_values",
                                         data=flatten_bn(pixel_values,
                                                         concat=True),
                                         resolve_bindings={
                                             "h": h,
                                             "w": w,
                                         })

        if image_embeds is not None:
            raise NotImplementedError

        raise AssertionError("This line should be unreachable.")

    def _process_image_input(
            self, image_input: DonutImagePixelInputs) -> torch.Tensor:
        assert image_input["type"] == "pixel_values"
        pixel_values = image_input["data"]
        dtype = next(self.encoder.parameters()).dtype
        pixel_values = pixel_values.to(dtype)
        return self.encoder(pixel_values)

    def get_language_model(self) -> torch.nn.Module:
        return self.decoder

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings,
    ) -> torch.Tensor:
        return _flatten_embeddings(multimodal_embeddings)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        encoder_input_ids: torch.Tensor,
        encoder_positions: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            input_ids
                torch.Tensor of *decoder* input token ids.
            positions
                torch.Tensor of *decoder* position indices.
            encoder_input_ids
                torch.Tensor of *encoder* input token ids.
            encoder_positions
                torch.Tensor of *encoder* position indices
        Returns:
            Output torch.Tensor
        """

        inputs_embeds = None
        if encoder_input_ids.numel() > 0:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(encoder_input_ids,
                                                      vision_embeddings)

        hidden_states = self.decoder(input_ids,
                                     positions,
                                     inputs_embeds=inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.decoder.compute_logits(hidden_states, sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
