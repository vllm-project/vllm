# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://huggingface.co/nvidia/LocateAnything-3B/blob/main/modeling_locateanything.py  # noqa: E501
# and from vllm/model_executor/models/kimi_vl.py
from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from functools import cached_property
from typing import Annotated, Any, Literal

import torch
from torch import nn
from transformers import BatchFeature
from transformers.activations import GELUActivation

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.moonvit import (
    MoonVitPretrainedModel,
    get_num_image_tokens,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    NestedTensors,
)
from vllm.multimodal.parse import (
    ImageEmbeddingItems,
    ImageProcessorItems,
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sampling_params import SamplingParams
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.locate_anything import LocateAnythingConfig
from vllm.transformers_utils.configs.moonvit import MoonViTConfig
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from .vision import run_dp_sharded_mrope_vision_model


# For dummy input only
@dataclass
class MaxImageTokenMeta:
    width: int = 1024
    height: int = 1024


@dataclass(frozen=True)
class LocateAnythingTokenIds:
    box_start: int
    box_end: int
    ref_start: int
    ref_end: int
    coord_start: int
    coord_end: int
    none: int

    @classmethod
    def from_config(cls, config: LocateAnythingConfig) -> "LocateAnythingTokenIds":
        return cls(
            box_start=config.box_start_token_id,
            box_end=config.box_end_token_id,
            ref_start=config.ref_start_token_id,
            ref_end=config.ref_end_token_id,
            coord_start=config.coord_start_token_id,
            coord_end=config.coord_end_token_id,
            none=config.none_token_id,
        )


class LocateAnythingSlowLogitsProcessor:
    """Constrain AR decoding only while generating LocateAnything box blocks."""

    def __init__(self, token_ids: LocateAnythingTokenIds):
        self.token_ids = token_ids
        # Precompute the coordinate id range — it is ~1000 ids and would
        # otherwise be rebuilt on every decode step.
        self._coords = frozenset(range(token_ids.coord_start, token_ids.coord_end + 1))

    def __call__(
        self,
        past_tokens_ids: Sequence[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        allowed = self._allowed_token_ids(past_tokens_ids)
        if allowed is None:
            return logits

        # The allowed set is often the full coordinate range (~1000 ids), so
        # assign in a single vectorized index_put rather than looping in Python
        # (which dispatches one kernel per token, every decode step).
        vocab_size = logits.shape[-1]
        allowed_ids = torch.tensor(
            [t for t in allowed if 0 <= t < vocab_size],
            dtype=torch.long,
            device=logits.device,
        )
        mask = torch.full_like(logits, -float("inf"))
        mask[allowed_ids] = logits[allowed_ids]
        return mask

    def _allowed_token_ids(
        self,
        past_tokens_ids: Sequence[int],
    ) -> AbstractSet[int] | None:
        ids = self.token_ids
        last_start = self._rindex(past_tokens_ids, ids.box_start)
        if last_start is None:
            return None

        last_end = self._rindex(past_tokens_ids, ids.box_end)
        if last_end is not None and last_end > last_start:
            return None

        box_tokens = past_tokens_ids[last_start + 1 :]
        coords = self._coords
        if not box_tokens:
            return coords | {ids.none}

        if box_tokens == [ids.none]:
            return {ids.box_end}

        if any(token_id not in coords for token_id in box_tokens):
            return {ids.box_end}

        num_coords = len(box_tokens)
        if num_coords < 2:
            return coords
        if num_coords == 2:
            return coords | {ids.box_end}
        if num_coords < 4:
            return coords
        return {ids.box_end}

    @staticmethod
    def _rindex(tokens: Sequence[int], target: int) -> int | None:
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] == target:
                return i
        return None


class LocateAnythingSlowGrammarLogitsProcessor(AdapterLogitsProcessor):
    """vLLM adapter for LocateAnything's slow-mode box grammar.

    This processor constrains autoregressive decoding so that `<box_start>`
    blocks only emit valid coordinate / `none` / `<box_end>` token
    sequences. It is **not** enabled automatically when the model is loaded —
    vLLM only wires in custom logits processors that are passed explicitly
    (or registered via the `vllm.logits_processors` entry point). To enable
    the box grammar, pass this class to the engine and disable special-token
    skipping (required, otherwise `validate_params` raises).

    Offline (`LLM`):

    ```python
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.locate_anything import (
        LocateAnythingSlowGrammarLogitsProcessor,
    )

    llm = LLM(
        model="nvidia/LocateAnything-3B",
        trust_remote_code=True,
        logits_processors=[LocateAnythingSlowGrammarLogitsProcessor],
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        skip_special_tokens=False,
    )
    ```

    Online (`vllm serve`):

    ```bash
    vllm serve nvidia/LocateAnything-3B --trust-remote-code \\
        --logits-processors \\
    vllm.model_executor.models.locate_anything:LocateAnythingSlowGrammarLogitsProcessor
    ```

    Without this wiring the model still runs, but box blocks decode without
    grammar constraints and may emit malformed coordinate sequences.
    """

    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool
    ):
        super().__init__(vllm_config, device, is_pin_memory)
        config = vllm_config.model_config.hf_config
        self.token_ids = LocateAnythingTokenIds.from_config(config)

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        if sampling_params.skip_special_tokens:
            raise ValueError(
                "LocateAnything structured outputs require "
                "SamplingParams(skip_special_tokens=False)."
            )

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> LocateAnythingSlowLogitsProcessor:
        return LocateAnythingSlowLogitsProcessor(self.token_ids)


class LocateAnythingMultiModalProjector(nn.Module):
    """InternVL-style ``mlp1`` connector.

    Unlike Kimi-VL, the LayerNorm is applied *after* the 2x2 patch merge
    (i.e. over the 4608-dim feature), matching the HF ``mlp1`` Sequential:

        mlp1.0 = LayerNorm(1152 * 2 * 2)
        mlp1.1 = Linear(4608, 2048)
        mlp1.2 = GELU()
        mlp1.3 = Linear(2048, 2048)
    """

    def __init__(self, config: LocateAnythingConfig, prefix: str = ""):
        super().__init__()
        merge = config.vision_config.merge_kernel_size
        self.merged_size = config.vision_config.hidden_size * merge[0] * merge[1]
        text_hidden = config.text_config.hidden_size

        self.pre_norm = nn.LayerNorm(self.merged_size, eps=1e-5)
        self.linear_1 = ReplicatedLinear(
            self.merged_size,
            text_hidden,
            bias=True,
            prefix=maybe_prefix(prefix, "linear_1"),
        )
        self.act = GELUActivation()
        self.linear_2 = ReplicatedLinear(
            text_hidden,
            text_hidden,
            bias=True,
            prefix=maybe_prefix(prefix, "linear_2"),
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = image_features.view(-1, self.merged_size)
        hidden_states = self.pre_norm(hidden_states)
        hidden_states, _ = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


class LocateAnythingImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - np: Number of patches (total across all images)
        - ps: Patch size
        - ni: Number of images
    """

    type: Literal["pixel_values"] = "pixel_values"

    pixel_values: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("np", 3, "ps", "ps"),
    ]

    image_grid_hws: Annotated[torch.Tensor, TensorShape("ni", 2)]


# We only support pixel input for LocateAnything now
LocateAnythingImageInputs = LocateAnythingImagePixelInputs


class LocateAnythingProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(LocateAnythingConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        image_processor = self.get_hf_processor().image_processor
        return get_num_image_tokens(
            image_width=image_width,
            image_height=image_height,
            patch_size=image_processor.patch_size,
            merge_kernel_size=image_processor.merge_kernel_size,
            in_token_limit=image_processor.in_token_limit,
        )

    @property
    def image_token_id(self) -> int:
        return self.get_hf_config().image_token_index

    @cached_property
    def image_wrapper_token_ids(self) -> tuple[int, int]:
        """Resolve the static ``<img>`` / ``</img>`` wrapper token ids once.

        These are fixed added-tokens for the model, so look them up (and
        validate they exist) a single time instead of on every prompt-update
        pass.
        """
        tokenizer = self.get_hf_processor().tokenizer
        img_start_id = tokenizer.convert_tokens_to_ids("<img>")
        img_end_id = tokenizer.convert_tokens_to_ids("</img>")
        unk_id = tokenizer.unk_token_id
        if img_start_id == unk_id:
            raise ValueError(
                "Tokenizer does not have '<img>' token — "
                "expected it in added_tokens for LocateAnything"
            )
        if img_end_id == unk_id:
            raise ValueError(
                "Tokenizer does not have '</img>' token — "
                "expected it in added_tokens for LocateAnything"
            )
        return img_start_id, img_end_id


class LocateAnythingDummyInputsBuilder(
    BaseDummyInputsBuilder[LocateAnythingProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        # The HF LocateAnythingProcessor only computes ``pixel_values`` for
        # images that are referenced by a numbered ``<image-N>`` placeholder
        # (1-indexed) in the text; images without a matching placeholder are
        # silently dropped. The dummy text must therefore use the same
        # numbered placeholders the processor recognises.
        return "".join(f"<image-{i + 1}>" for i in range(num_images))

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        image_overrides = mm_options.get("image")

        return {
            "image": self._get_dummy_images(
                width=MaxImageTokenMeta.width,
                height=MaxImageTokenMeta.height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class LocateAnythingMultiModalProcessor(
    BaseMultiModalProcessor[LocateAnythingProcessingInfo]
):
    # Our _call_hf_processor does NOT expand <image-N> placeholders into
    # <img> + IMG_CONTEXT + </img> — it only tokenises raw text.  Tell the
    # framework to apply the replacements from _get_prompt_updates instead
    # of assuming the HF processor already did so.
    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_grid_hws = hf_inputs.get("image_grid_hws", torch.empty((0, 2)))
        image_grid_sizes = image_grid_hws.prod(-1)

        # pixel_values is merged as a single large tensor
        # image_grid_hws is shapes for each subtensor in pixel_values
        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", image_grid_sizes
            ),
            image_grid_hws=MultiModalFieldConfig.batched("image"),
        )

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Override to avoid calling the full HF LocateAnythingProcessor, which
        # binds placeholder expansion to regex-matching <image-N> in the text
        # *and* requiring the corresponding image in the same call (otherwise
        # IndexError in replace_media_placeholder).  vLLM's caching path calls
        # text-only and mm-only separately, so we split them here.
        hf_processor = self.info.get_hf_processor(**mm_kwargs)

        tokenized = hf_processor.tokenizer(prompt, **tok_kwargs)
        input_ids = tokenized["input_ids"]
        attn_mask = tokenized["attention_mask"]
        # Convert tensor to nested list if needed.
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
            attn_mask = attn_mask.tolist()
        # Wrap in a batch list — _apply_hf_processor_text_mm does
        # (prompt_ids,) = input_ids, so it expects [[tokens...]].
        if input_ids and not isinstance(input_ids[0], list):
            input_ids = [input_ids]
            attn_mask = [attn_mask]
        # Handle empty prompt (e.g. get_dummy_text({})) — the framework
        # expects at least one (possibly empty) sequence in the batch.
        if not input_ids:
            input_ids = [[]]
            attn_mask = [[]]
        result = BatchFeature(
            data=dict(input_ids=input_ids, attention_mask=attn_mask),
        )

        images = mm_data.get("images") if mm_data else None
        if images:
            # Inject return_tensors="pt" — the base class ctx.call_hf_processor
            # normally does this, but we bypass it here.
            img_kwargs = dict(
                return_tensors="pt", **(mm_kwargs.get("images_kwargs") or {})
            )
            image_outputs = hf_processor.image_processor(
                images=images,
                **img_kwargs,
            )
            # Convert numpy arrays to tensors — the image processor may
            # return image_grid_hws as a numpy array, but the framework's
            # TensorSchema validation expects tensors/lists.
            if "image_grid_hws" in image_outputs:
                gh = image_outputs["image_grid_hws"]
                if not isinstance(gh, torch.Tensor):
                    image_outputs["image_grid_hws"] = torch.tensor(gh)
            result.update(image_outputs)

        return result

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        image_token_id = self.info.image_token_id

        # The <img> / </img> wrapper token IDs are static; resolve them once
        # (cached on the ProcessingInfo) instead of re-looking-up per call.
        img_start_id, img_end_id = self.info.image_wrapper_token_ids

        def get_replacement(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )

            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                )

            img_ctx = [image_token_id] * num_image_tokens
            # Only the IMG_CONTEXT tokens (middle part) get embeddings;
            # the <img> and </img> wrappers are plain text.
            return PromptUpdateDetails.select_token_id(
                seq=[img_start_id] + img_ctx + [img_end_id],
                embed_token_id=image_token_id,
            )

        # target is a callable returning a per-item string so the framework
        # tokenises it and matches against the actual token sequence produced
        # by our overridden _call_hf_processor / _apply_hf_processor_text_only.
        def get_target(item_idx: int):
            return f"<image-{item_idx + 1}>"

        return [
            PromptReplacement(
                modality="image",
                target=get_target,
                replacement=get_replacement,
            ),
        ]


@MULTIMODAL_REGISTRY.register_processor(
    LocateAnythingMultiModalProcessor,
    info=LocateAnythingProcessingInfo,
    dummy_inputs=LocateAnythingDummyInputsBuilder,
)
class LocateAnythingForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    # Enable data-parallel vision encoding (same MoonViT as KimiVL).
    supports_encoder_tp_data = True

    # HF top-level submodule names -> vLLM naming
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "vision_model.": "vision_tower.",
            "mlp1.0.": "multi_modal_projector.pre_norm.",
            "mlp1.1.": "multi_modal_projector.linear_1.",
            "mlp1.3.": "multi_modal_projector.linear_2.",
        },
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            # The HF LocateAnythingProcessor recognises *numbered* placeholders
            # of the form ``<image-N>`` (1-indexed) in the input text. Only
            # when such a placeholder is present does it run the image
            # processor and emit ``pixel_values`` / ``image_grid_hws``; it then
            # expands ``<image-N>`` into ``<img>`` + num_image_tokens x
            # ``<IMG_CONTEXT>`` + ``</img>``. vLLM locates that repeated
            # ``<IMG_CONTEXT>`` run via ``_get_prompt_updates``. ``i`` is the
            # 1-indexed item number supplied by chat_utils.
            return f"<image-{i}>"

        raise ValueError("Only image modality is supported")

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        model_config = vllm_config.model_config
        config: LocateAnythingConfig = model_config.hf_config

        self.config = config
        assert isinstance(config.vision_config, MoonViTConfig)
        self.use_data_parallel = (
            model_config.multimodal_config.mm_encoder_tp_mode == "data"
        )

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_tower = MoonVitPretrainedModel(
                config.vision_config,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            self.multi_modal_projector = LocateAnythingMultiModalProjector(
                config=config,
                prefix=maybe_prefix(prefix, "multi_modal_projector"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Qwen2ForCausalLM"],
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> LocateAnythingImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_hws = kwargs.pop("image_grid_hws", None)

        if pixel_values is None:
            return None

        if image_grid_hws is None:
            raise ValueError("image_grid_hws is required when pixel_values is provided")

        return LocateAnythingImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            image_grid_hws=image_grid_hws,
        )

    @torch.inference_mode()
    def _process_image_pixels(
        self, inputs: LocateAnythingImagePixelInputs
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        pixel_values = inputs["pixel_values"]
        image_grid_hws = inputs["image_grid_hws"]
        # The image processor returns float32; cast to the model's dtype.
        pixel_values = pixel_values.to(dtype=next(self.vision_tower.parameters()).dtype)
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.vision_tower,
                pixel_values,
                image_grid_hws.tolist(),
                rope_type="rope_2d",
            )
        return self.vision_tower(pixel_values, image_grid_hws)

    def _process_image_input(
        self, image_input: LocateAnythingImageInputs
    ) -> tuple[torch.Tensor, ...]:
        assert image_input["type"] == "pixel_values"
        image_features = self._process_image_pixels(image_input)
        assert isinstance(image_features, (list, tuple))
        lengths = [x.shape[0] for x in image_features]
        return self.multi_modal_projector(torch.cat(image_features)).split(lengths)

    def embed_multimodal(self, **kwargs: object) -> NestedTensors | None:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        # The HF checkpoint ships ``language_model.lm_head.weight`` even though
        # ``text_config.tie_word_embeddings`` is True. With tied embeddings the
        # Qwen2 backbone reuses ``embed_tokens`` for the output projection and
        # has no separate ``lm_head`` parameter, so the checkpoint key has no
        # destination and loading would raise "unexpected key in source
        # state_dict: language_model.lm_head.weight". Skip it when tied.
        tied = getattr(self.config.text_config, "tie_word_embeddings", False)
        skip = ["language_model.lm_head."] if tied else None
        loader = AutoWeightsLoader(self, skip_prefixes=skip)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
