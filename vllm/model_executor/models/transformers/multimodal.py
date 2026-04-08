# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The vLLM team.
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
"""Transformers modeling backend mixin for multi-modal models."""

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from transformers import AutoModel

from vllm.compilation.decorators import should_torch_compile_mm_encoder
from vllm.config.utils import getattr_iter
from vllm.inputs import MultiModalDataDict, MultiModalInput, mm_input
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMRoPE, SupportsMultiModal
from vllm.multimodal import MultiModalKwargsItems
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    PlaceholderRange,
)
from vllm.multimodal.parse import (
    ImageProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    TimingContext,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from transformers import BatchFeature, PreTrainedModel

    from vllm.config import VllmConfig
    from vllm.config.multimodal import BaseDummyOptions

logger = init_logger(__name__)


class MultiModalProcessingInfo(BaseProcessingInfo):
    def _is_audio_model(self) -> bool:
        processor = self.get_hf_processor()
        return hasattr(processor, "audio_token")

    def _get_audio_token_id(self) -> int:
        processor = self.get_hf_processor()
        if hasattr(processor, "audio_token_id"):
            return processor.audio_token_id
        config = self.get_hf_config()
        for attr in ("audio_token_id", "audio_token_index"):
            if hasattr(config, attr):
                return getattr(config, attr)
        if hasattr(processor, "audio_token"):
            tokenizer = self.get_tokenizer()
            vocab = tokenizer.get_vocab()
            if processor.audio_token in vocab:
                return vocab[processor.audio_token]
        raise ValueError("Cannot find audio_token_id on processor or model config")

    def _get_audio_sampling_rate(self) -> float:
        processor = self.get_hf_processor()
        for attr in ("audio_processor", "feature_extractor"):
            sub = getattr(processor, attr, None)
            if sub is not None and hasattr(sub, "sampling_rate"):
                return sub.sampling_rate
        return 16000.0

    def get_data_parser(self) -> MultiModalDataParser:
        if self._is_audio_model():
            return MultiModalDataParser(
                target_sr=self._get_audio_sampling_rate(),
                expected_hidden_size=self._get_expected_hidden_size(),
            )
        return super().get_data_parser()

    def get_supported_mm_limits(self):
        if self._is_audio_model():
            return {"audio": None}
        return {"image": None}

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        if self._is_audio_model():
            return {"audio": self.get_max_audio_tokens()}
        return {"image": self.get_max_image_tokens()}

    def get_max_audio_tokens(self) -> int:
        config = self.get_hf_config()
        for cfg_attr in ("audio_config", "encoder_config"):
            sub = getattr(config, cfg_attr, None)
            if sub is not None:
                for token_attr in ("max_source_positions", "max_position_embeddings"):
                    val = getattr(sub, token_attr, None)
                    if val is not None:
                        return int(val)
        # Voxtral's max_source_positions=3000 is the largest known value;
        # AudioFlamingo3/GLM-ASR use 1500. Granite Speech is variable.
        return 3000

    def get_max_image_tokens(self) -> int:
        width, height = self.get_max_image_size()
        processor = self.get_hf_processor()
        multimodal_config = self.ctx.model_config.multimodal_config
        mm_processor_kwargs = multimodal_config.mm_processor_kwargs or {}
        mm_tokens = processor._get_num_multimodal_tokens(
            image_sizes=([height, width],), **mm_processor_kwargs
        )
        image_tokens = mm_tokens["num_image_tokens"][0]
        return image_tokens

    def get_max_image_size(self):
        return 10_000, 10_000  # hardcode for arbitrary very large size


class MultiModalDummyInputsBuilder(BaseDummyInputsBuilder[MultiModalProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        if self.info._is_audio_model():
            num_audios = mm_counts.get("audio", 0)
            processor = self.info.get_hf_processor()
            audio_token = getattr(processor, "audio_token", "")
            return audio_token * num_audios

        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        if "gemma3" in processor.__class__.__name__.lower():
            image_token = processor.boi_token
        else:
            image_token = getattr(processor, "image_token", "")
        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, "BaseDummyOptions"],
    ) -> MultiModalDataDict:
        if self.info._is_audio_model():
            num_audios = mm_counts.get("audio", 0)
            audio_overrides = mm_options.get("audio")
            sampling_rate = self.info._get_audio_sampling_rate()
            processor = self.info.get_hf_processor()
            for attr in ("audio_processor", "feature_extractor"):
                sub = getattr(processor, attr, None)
                if sub is not None:
                    chunk_length = getattr(sub, "chunk_length", None)
                    if chunk_length is not None:
                        break
            else:
                chunk_length = 30
            audio_len = int(chunk_length * sampling_rate)
            return {
                "audio": self._get_dummy_audios(
                    length=audio_len,
                    num_audios=num_audios,
                    overrides=audio_overrides,
                ),
            }

        num_images = mm_counts.get("image", 0)

        target_width, target_height = self.info.get_max_image_size()

        image_overrides = mm_options.get("image")

        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            ),
        }


class MultiModalProcessor(BaseMultiModalProcessor[MultiModalProcessingInfo]):
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ):
        """
        Given the original multi-modal items for this modality
        and HF-processed data, output the updates to perform.

        The information returned by this method is used to update token inputs
        which bypass the HF processor. It is also used to update the output of
        HF processor if the HF process does not apply prompt updates to text
        inputs.

        Moreover, this information is critical to determine the token positions
        in order to construct  :class:`~vllm-multimodal.input.PlaceholderRange`
        for each multi-modal item.
        """
        return None

    def _get_mm_fields_config(
        self,
        hf_inputs: "BatchFeature",
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # HF Processors always return a mask but vLLM doesn't need it
        hf_inputs.pop("attention_mask", None)

        if self.info._is_audio_model():
            num_audio_tokens = hf_inputs.get("num_audio_tokens")
            mm_fields = {
                key: MultiModalFieldConfig.flat_from_sizes("audio", num_audio_tokens)
                for key in hf_inputs
            }
            mm_fields["num_audio_tokens"] = MultiModalFieldConfig.batched("audio")
            return mm_fields

        num_image_patches = hf_inputs.get("num_image_patches")
        mm_fields = {
            key: MultiModalFieldConfig.flat_from_sizes("image", num_image_patches)
            for key in hf_inputs
        }
        mm_fields["image_embeds"] = MultiModalFieldConfig.flat_from_sizes(
            "image", num_image_patches
        )

        # Keep these as batched, as they always have batch size as first dim
        mm_fields["image_grid_thw"] = MultiModalFieldConfig.batched("image")
        mm_fields["video_grid_thw"] = MultiModalFieldConfig.batched("image")
        mm_fields["num_image_patches"] = MultiModalFieldConfig.batched("image")
        return mm_fields

    def _get_hf_mm_data(
        self,
        mm_items: MultiModalDataItems,
    ) -> tuple[Mapping[str, object], Mapping[str, object]]:
        """
        In contrast to the base class, this method adds
        `return_mm_token_type_ids` for vision models and remaps the
        `audios` key to `audio` for audio models.
        """
        processor_data, passthrough_data = super()._get_hf_mm_data(mm_items)
        if self.info._is_audio_model():
            if "audios" in processor_data:
                processor_data["audio"] = processor_data.pop("audios")
        else:
            processor_data["return_mm_token_type_ids"] = True
        return processor_data, passthrough_data

    def _apply_audio(
        self,
        prompt_ids: list[int],
        processed_data: "BatchFeature",
        hf_processor_mm_kwargs: Mapping[str, object],
        mm_hashes: object,
    ) -> MultiModalInput:
        audio_token_id = self.info._get_audio_token_id()
        prompt_tensor = torch.tensor(prompt_ids)
        is_audio = prompt_tensor == audio_token_id

        mm_placeholders: dict[str, list[PlaceholderRange]] = {}
        if is_audio.any():
            padded = torch.cat([torch.tensor([False]), is_audio, torch.tensor([False])])
            transitions = padded.int().diff()
            starts = torch.where(transitions == 1)[0]
            ends = torch.where(transitions == -1)[0]
            lengths = ends - starts

            ranges = [
                PlaceholderRange(
                    offset=s.item(),
                    length=ln.item(),
                    is_embed=torch.ones(ln.item(), dtype=torch.bool),
                )
                for s, ln in zip(starts, lengths)
            ]
            mm_placeholders = {"audio": ranges}
            processed_data["num_audio_tokens"] = lengths

        mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
            processed_data,
            self._get_mm_fields_config(processed_data, hf_processor_mm_kwargs),
        )

        return mm_input(
            prompt_token_ids=prompt_ids,
            mm_kwargs=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
        )

    def _apply_vision(
        self,
        prompt_ids: list[int],
        processed_data: "BatchFeature",
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        mm_hashes: object,
    ) -> MultiModalInput:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        # For gemma3 we check `token_type_ids` as the key
        token_type_key = (
            "mm_token_type_ids"
            if "mm_token_type_ids" in processed_data
            else "token_type_ids"
        )
        mm_token_type_ids = processed_data.get(token_type_key)

        # We can infer vLLM style placeholder from token type ids, if we split
        # it for each input `mm_data`.
        mm_positions = torch.where(mm_token_type_ids == 1)[1]
        images = mm_items.get_items("image", ImageProcessorItems)
        image_sizes = []
        for item_idx in range(len(images)):
            image_size = images.get_image_size(item_idx)
            image_sizes.append((image_size.height, image_size.width))

        mm_tokens_per_modality = hf_processor._get_num_multimodal_tokens(
            image_sizes=image_sizes,
            **self.info.ctx.get_merged_mm_kwargs({}),
        )

        mm_placeholders: dict[str, list[PlaceholderRange]] = {}
        split_sizes = mm_tokens_per_modality["num_image_tokens"]
        if split_sizes:
            chunked_mm_positions = torch.split(mm_positions, split_sizes)
            mm_tokens = torch.tensor(prompt_ids)[mm_token_type_ids[0].bool()]
            chunked_mm_tokens = torch.split(mm_tokens, split_sizes)
            ranges = [
                PlaceholderRange(
                    offset=positions[0].item(),
                    length=positions.shape[0],
                    is_embed=(mm_tokens == hf_processor.image_token_id).bool(),
                )
                for positions, mm_tokens in zip(chunked_mm_positions, chunked_mm_tokens)
            ]
            mm_placeholders = {"image": ranges}

        processed_data["num_image_patches"] = torch.tensor(
            mm_tokens_per_modality["num_image_patches"]
        )
        mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
            processed_data,
            self._get_mm_fields_config(processed_data, hf_processor_mm_kwargs),
        )

        return mm_input(
            prompt_token_ids=prompt_ids,
            mm_kwargs=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
        )

    def apply(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> MultiModalInput:
        """
        Process multi-modal inputs to be used in vLLM.

        Apply HF Processor on prompt text and multi-modal data together,
        outputting token IDs and processed tensors.
        """
        prompt = inputs.prompt
        mm_items = inputs.mm_data_items
        hf_processor_mm_kwargs = inputs.hf_processor_mm_kwargs
        tokenization_kwargs = inputs.tokenization_kwargs

        with timing_ctx.record("apply_hf_processor"):
            hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
            if not isinstance(prompt, str):
                # the prompt is the tokenized ids which is not supported
                # by the hf_processor, which is why we would need to decode the ids
                # into string
                prompt = hf_processor.decode(prompt)

            # Bypass cached processor and always apply to the full set of mm inputs
            # NOTE: we can't just set caching=False because base class method
            # transforms outputs to `MultiModalKwargs` which is not going to
            # work for Transformers. The vision path has logic tied to
            # `mm_tokens_per_modality` in _apply_vision()
            prompt_ids, processed_data, _ = self._apply_hf_processor_text_mm(
                prompt_text=prompt,
                mm_items=mm_items,
                hf_processor_mm_kwargs=hf_processor_mm_kwargs,
                tokenization_kwargs=tokenization_kwargs,
            )

        # Use overrides if provided; fallback to data-dependent hashing.
        with timing_ctx.record("get_mm_hashes"):
            mm_hashes = inputs.get_mm_hashes(self.info.model_id)

        if self.info._is_audio_model():
            return self._apply_audio(
                prompt_ids,
                processed_data,
                hf_processor_mm_kwargs,
                mm_hashes,
            )

        return self._apply_vision(
            prompt_ids,
            processed_data,
            mm_items,
            hf_processor_mm_kwargs,
            mm_hashes,
        )


class MultiModalMixin(SupportsMultiModal, SupportsMRoPE):
    supports_multimodal_raw_input_only = True

    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        # Skip SupportsMRoPE.__init__ and call the next class in MRO
        super(SupportsMRoPE, self).__init__(vllm_config=vllm_config, prefix=prefix)

    def _get_encoder_cls(
        self, modality: str = "image", **kwargs: dict
    ) -> type["PreTrainedModel"]:
        """
        Get the encoder class from the model.

        Args:
            kwargs: The kwargs to create the model.

        Returns:
            The encoder class.
        """
        with torch.device("meta"):
            model: PreTrainedModel = AutoModel.from_config(**kwargs)
        encoder_cls = type(model.get_encoder(modality=modality))
        logger.debug("Identified encoder class as: %s", encoder_cls)
        if type(model) is encoder_cls:
            raise ValueError(
                "Unable to infer vision encoder class from the model. "
                "You must either: update the model so that "
                "https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.get_encoder"
                " can detect the vision encoder correctly, or remove "
                "'compile_mm_encoder'."
            )
        del model
        return encoder_cls

    def _decorate_for_torch_compile(self, **kwargs: dict):
        """
        Decorate the model's decoder and encoder classes to indicate to vLLM that they
        support torch compile if `can_enable_torch_compile` and
        `should_torch_compile_mm_encoder` are True respectively.

        Args:
            kwargs: The kwargs to create the model, which are needed to get the decoder
                and encoder classes.
        """
        super()._decorate_for_torch_compile(**kwargs)
        # Decorate the vision encoder model class to support torch compile if needed
        if self.compilation_config.compile_mm_encoder:
            self.check_version("5.0.0", "multimodal encoder compilation support")
            logger.warning_once(
                "Multimodal encoder compilation with the Transformers modeling backend "
                "is an experimental feature. It relies on:\n"
                "- The vision encoder being torch compilable.\n"
                "- All vision encoder tensor inputs must be type hinted as either "
                "`torch.Tensor` or `torch.FloatTensor`.\n"
                "- The 0-th dimension of all tensor inputs to the vision encoder being "
                "the dynamic dimension (i.e., sequence length or number of patches).\n"
                "Please report any issues you encounter to help us improve it."
            )
            self._decorate_cls_for_torch_compile(
                cls=self._get_encoder_cls(**kwargs),
                # TODO: properly infer dynamic_arg_dims based on the encoder's forward
                # method signature. Currently we assume dim 0 for all tensor inputs.
                dynamic_arg_dims=None,
                enable_if=should_torch_compile_mm_encoder,
                is_encoder=True,
            )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        # Gemma3 and PaliGemma needs `token_type_ids` to work correctly
        # Other models will not have `token_type_ids` in kwargs
        kwargs = {k: v for k, v in kwargs.items() if k == "token_type_ids"}
        # Positions shape handling for MRoPE models
        if self.model_config.uses_mrope:
            # [3, seq_len] -> [3, 1, seq_len]
            positions = positions[:, None]
        model_output = super().forward(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )
        return model_output

    def get_language_model(self) -> torch.nn.Module:
        """Transformers modeling backend multimodal classes do not contain a separate
        vLLM language model class. Therefore, in order to return a language model vLLM
        class, we use a wrapper to give `self` the same interface as a text model."""

        # Exclude self and object
        bases = self.__class__.mro()[1:-1]
        # Keep only classes defined in `vllm.model_executor.models.transformers`
        bases = [b for b in bases if ".transformers." in b.__module__]
        # Exclude MultiModalMixin itself
        bases = [b for b in bases if b is not MultiModalMixin]

        class LanguageModel(*bases):
            def __init__(self, multimodal_model):
                # Don't call super().__init__() to avoid re-initialization
                self.__dict__.update(multimodal_model.__dict__)

            model = getattr_iter(self.model, ("language_model", "text_model"), None)

        return LanguageModel(self)

    def _split_embeddings(
        self, embeddings: torch.Tensor, split_sizes: list[int]
    ) -> list[torch.Tensor]:
        total_expected = sum(split_sizes)

        # Flatten to 2D: [total_tokens, hidden_dim]
        if embeddings.ndim == 3:
            embeddings = embeddings.view(-1, embeddings.shape[-1])

        total_tokens = embeddings.shape[0]
        if total_tokens == total_expected:
            # Direct match: split_sizes are actual token counts
            token_split_sizes = split_sizes
        elif total_expected > 0 and total_tokens % total_expected == 0:
            # Uniform expansion: each item expands to N tokens
            tokens_per_item = total_tokens // total_expected
            token_split_sizes = [s * tokens_per_item for s in split_sizes]
        elif total_expected > 0:
            # Mismatch (profiling with dummy data) - pad/truncate
            if total_tokens == 0:
                raise ValueError(
                    "Encoder returned empty embeddings. "
                    f"Expected {total_expected} tokens from "
                    f"split_sizes={split_sizes}"
                )
            if total_tokens < total_expected:
                repeat_factor = (total_expected + total_tokens - 1) // total_tokens
                embeddings = embeddings.repeat(repeat_factor, 1)
            embeddings = embeddings[:total_expected]
            token_split_sizes = split_sizes
        else:
            return []

        return list(torch.split(embeddings, token_split_sizes, dim=0))

    def _embed_audio(self, **kwargs) -> list[torch.Tensor] | None:
        input_features: torch.Tensor | None = kwargs.pop("input_features", None)
        if input_features is None:
            input_features = kwargs.pop("input_values", None)
        if input_features is None:
            return None

        num_audio_tokens = kwargs.pop("num_audio_tokens")
        kwargs.pop("token_type_ids", None)
        kwargs.pop("mm_token_type_ids", None)

        if current_platform.is_rocm():
            with torch.nn.attention.sdpa_kernel(
                backends=[torch.nn.attention.SDPBackend.MATH]
            ):
                audio_output = self.model.get_audio_features(input_features, **kwargs)
        else:
            audio_output = self.model.get_audio_features(input_features, **kwargs)

        if isinstance(audio_output, tuple):
            audio_embeddings = audio_output[1]
        elif hasattr(audio_output, "pooler_output"):
            audio_embeddings = audio_output.pooler_output
        else:
            audio_embeddings = audio_output

        split_sizes = num_audio_tokens.flatten().tolist()
        return self._split_embeddings(audio_embeddings, split_sizes)

    def _embed_vision(self, **kwargs) -> list[torch.Tensor] | torch.Tensor | None:
        pixel_values: torch.Tensor | None = kwargs.pop("pixel_values", None)
        image_embeds: torch.Tensor | None = kwargs.pop("image_embeds", None)
        # Model might use `image_patches` instead of `pixel_values`
        if pixel_values is None:
            pixel_values = kwargs.pop("image_patches", None)

        if image_embeds is not None:
            return image_embeds

        if pixel_values is None:
            return None

        num_image_patches = kwargs.pop("num_image_patches")
        kwargs.pop("token_type_ids", None)  # used only in `forward`
        kwargs.pop("mm_token_type_ids", None)  # used only in `get_rope_index`

        # ROCm: Force math SDP backend for vision encoder to avoid accuracy issues
        # with flash_sdp and mem_efficient_sdp
        if current_platform.is_rocm():
            # TODO: [ROCm] Fix accuracy issues with flash backend
            logger.debug(
                "ROCm platform detected. Forcing math SDP backend "
                "for vision encoder. Currently ROCm platform has "
                "accuracy issues with `flash_sdp` and"
                "`mem_efficient_sdp` backends. See issue: "
                "https://github.com/vllm-project/vllm/issues/30167"
            )
            with torch.nn.attention.sdpa_kernel(
                backends=[torch.nn.attention.SDPBackend.MATH]
            ):
                vision_embeddings = self.model.get_image_features(
                    pixel_values, **kwargs
                )
        else:
            vision_embeddings = self.model.get_image_features(pixel_values, **kwargs)

        # Transformers `v5`, `self.get_image_features` returns a tuple
        # containing the features and optionally attentions/hidden_states
        # After v5 is settled, we can enable qwen3-vl with several outputs
        # from `self.get_image_features`
        if isinstance(vision_embeddings, tuple):
            vision_embeddings = vision_embeddings[0]
        elif isinstance(vision_embeddings, dict):
            vision_embeddings = vision_embeddings.pooler_output

        if isinstance(vision_embeddings, torch.Tensor):
            split_sizes = num_image_patches.flatten().tolist()
            return self._split_embeddings(vision_embeddings, split_sizes)

        return vision_embeddings

    def embed_multimodal(self, **kwargs):
        if "input_features" in kwargs or "input_values" in kwargs:
            return self._embed_audio(**kwargs)
        return self._embed_vision(**kwargs)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        kwargs = MultiModalFeatureSpec.gather_kwargs(
            mm_features,
            {
                "image_grid_thw",
                "video_grid_thw",
                "mm_token_type_ids",
                "second_per_grid_ts",
                "audio_feature_lengths",
                "use_audio_in_video",
            },
        )
        if any(
            v
            for k, v in kwargs.items()
            if k not in {"image_grid_thw", "mm_token_type_ids"}
        ):
            raise NotImplementedError(
                "Transformers modeling backend only supports images."
            )

        image_grid_thw = kwargs.get("image_grid_thw", [])
        video_grid_thw = kwargs.get("video_grid_thw", [])
        mm_token_type_ids = kwargs.get("mm_token_type_ids")

        image_grid_thw = (torch.stack if image_grid_thw else torch.tensor)(
            image_grid_thw
        )
        video_grid_thw = (torch.stack if video_grid_thw else torch.tensor)(
            video_grid_thw
        )

        # In v4 `get_rope_index` doesn't have wildcard `kwargs`, and
        # can't accept arbitrary args, even if its value is `None`
        kwargs = {}
        if not hasattr(self, "_get_rope_index_accepts_mm_token_type_ids"):
            import inspect

            sig = inspect.signature(self.model.get_rope_index)
            params = sig.parameters
            self._get_rope_index_accepts_mm_token_type_ids = (
                "mm_token_type_ids" in params
                or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            )
        if self._get_rope_index_accepts_mm_token_type_ids:
            if mm_token_type_ids:
                kwargs["mm_token_type_ids"] = torch.cat(mm_token_type_ids)
            else:
                shape = (1, len(input_tokens))
                kwargs["mm_token_type_ids"] = torch.zeros(*shape, dtype=torch.int)

        mrope_positions, mrope_position_delta = self.model.get_rope_index(
            input_ids=torch.tensor(input_tokens).unsqueeze(0),
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs,
        )

        mrope_positions = mrope_positions[:, 0]
        mrope_position_delta = mrope_position_delta[0].item()

        return mrope_positions, mrope_position_delta
