# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 vision variant.

Thin multimodal wrapper around the text-only ``DeepseekV41LLMForCausalLM``:

- ``vision`` ViT + ``aligner`` produce per-image embeddings for the IMAGE
  positions; three learned vectors (``image_start`` / ``image_newline`` /
  ``image_end``) fill the delimiter positions. Every image-span position
  carries ``image_token_id`` (129264) in ``input_ids``; the per-position
  roles come from the processor's ``types`` tensor (see
  ``common/mm_preprocess.py``, and ``common/vision.py`` for the tower).
- Merged embeddings enter the text model via ``inputs_embeds``, i.e. before
  its hyper-connection stream expansion. Raw ``input_ids`` still flow into
  the model so the MoE router can apply ``bias_vl`` to image tokens
  (``requires_raw_input_tokens``).
"""

from collections.abc import Iterable
from typing import Annotated

import torch
from torch import nn

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.model_executor.models.vision import is_vit_use_data_parallel
from vllm.models.deepseek_v4.common.vision import (
    DeepseekV4Aligner,
    DeepseekV4ViT,
    run_dp_sharded_vision_tower,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from ..common.mm_preprocess import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_PAD_ID,
    IMAGE_PLACEHOLDER,
    IMAGE_SENTINEL_BASE_ID,
    IMAGE_START,
    DeepseekV4VLDummyInputsBuilder,
    DeepseekV4VLMultiModalProcessor,
    DeepseekV4VLProcessingInfo,
)
from .model import (
    DeepseekV41LLMForCausalLM,
    _linear_scale_param_name,
    _make_deepseek_v4_weights_mapper,
)


class DeepseekV4VLImagePixelInputs(TensorSchema):
    """ViT patch inputs for one batched set of images.

    Dimensions:
        - np: Total ViT patches across images (sum of n_vit_h * n_vit_w)
        - c: Number of image channels (3)
        - p: ViT patch size
        - ni: Number of images
        - ns: Total image-span positions (sum of n_llm_h * (n_llm_w + 1) + 2)
    """

    patches: Annotated[
        torch.Tensor, TensorShape("np", 3, "p", "p", dynamic_dims={"np"})
    ]
    # [n_vit_h, n_vit_w] per image
    vit_grid: Annotated[torch.Tensor, TensorShape("ni", 2)]
    # [n_llm_h, n_llm_w] per image
    llm_grid: Annotated[torch.Tensor, TensorShape("ni", 2)]
    types: Annotated[torch.Tensor, TensorShape("ns", dynamic_dims={"ns"})]


def _make_deepseek_v4_vl_weights_mapper(
    expert_dtype: str, linear_scale_name: str
) -> WeightsMapper:
    """Text-checkpoint mapping rules re-rooted under ``language_model.``."""
    base = _make_deepseek_v4_weights_mapper(expert_dtype, linear_scale_name)
    orig_to_new_prefix: dict[str, str | None] = {
        "layers.": "language_model.model.layers.",
        "embed.": "language_model.model.embed.",
        "norm.": "language_model.model.norm.",
        "hc_head": "language_model.model.hc_head",
        "mtp.": "language_model.model.mtp.",
    }
    return WeightsMapper(
        orig_to_new_prefix=orig_to_new_prefix,
        orig_to_new_regex=base.orig_to_new_regex,
        orig_to_new_suffix={
            "head.weight": "language_model.lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
        orig_to_new_substr={
            ".shared_experts.w2": ".shared_experts.down_proj",
            # The MTP/DSpark draft heads are not supported for the vision
            # variant; drop their weights.
            "mtp.": None,
        },
    )


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekV4VLMultiModalProcessor,
    info=DeepseekV4VLProcessingInfo,
    dummy_inputs=DeepseekV4VLDummyInputsBuilder,
)
class DeepseekV41ForCausalLM(nn.Module, SupportsMultiModal, SupportsPP, SupportsEagle3):
    """Multimodal entry point for DeepSeek-V4.1 checkpoints with a vision tower.

    ``SupportsEagle3`` (aux hidden-state plumbing for MTP/DSpark drafters)
    delegates through ``language_model`` via the protocol defaults.
    """

    supports_encoder_tp_data = True

    # The MoE router needs raw token ids to detect image-span tokens
    # (all carrying image_token_id, see common/mm_preprocess.py) and apply
    # bias_vl.
    requires_raw_input_tokens = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return IMAGE_PLACEHOLDER
        raise ValueError(f"Unsupported modality: {modality!r}")

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__()
        model_config = vllm_config.model_config
        config = model_config.hf_config
        self.config = config
        self.multimodal_config = model_config.multimodal_config
        assert self.multimodal_config is not None

        # The tower is always built; _mark_tower_model stubs it out
        # (StageMissingLayer, weights skipped) when the image limit is 0.
        with self._mark_tower_model(vllm_config, {"image"}):
            self.use_data_parallel = is_vit_use_data_parallel(config.vision_n_heads)
            self.vision = DeepseekV4ViT(config)
            self.aligner = DeepseekV4Aligner(config)
            self.image_start = nn.Parameter(
                torch.empty(config.hidden_size, dtype=torch.float32)
            )
            self.image_end = nn.Parameter(
                torch.empty(config.hidden_size, dtype=torch.float32)
            )
            self.image_newline = nn.Parameter(
                torch.empty(config.hidden_size, dtype=torch.float32)
            )
            self.vision.to(dtype=model_config.dtype)
            self.aligner.to(dtype=model_config.dtype)

        with self._mark_language_model(vllm_config):
            self.language_model = DeepseekV41LLMForCausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )
        # The outer mapper (see load_weights) fully resolves HF names into
        # this wrapper's namespace before AutoWeightsLoader strips the
        # "language_model." prefix and delegates to the child's load_weights,
        # so the child's own mapper must be a no-op. Its suffix rules are not
        # idempotent (e.g. "lm_head.weight".endswith("head.weight") would
        # re-fire "head.weight" -> "lm_head.weight").
        self.language_model.hf_to_vllm_mapper = WeightsMapper()
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.language_model.make_empty_intermediate_tensors
        )

        expert_dtype = getattr(config, "expert_dtype", "fp4")
        self.hf_to_vllm_mapper = _make_deepseek_v4_vl_weights_mapper(
            expert_dtype, _linear_scale_param_name(vllm_config, expert_dtype)
        )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> DeepseekV4VLImagePixelInputs | None:
        patches = kwargs.pop("patches", None)
        if patches is None:
            return None
        return DeepseekV4VLImagePixelInputs(
            patches=patches,
            vit_grid=kwargs.pop("vit_grid"),
            llm_grid=kwargs.pop("llm_grid"),
            types=kwargs.pop("types"),
            resolve_bindings={"p": self.config.vision_patch_size},
        )

    def _encode_image(
        self,
        patches: torch.Tensor,
        n_vit_h: int,
        n_vit_w: int,
    ) -> torch.Tensor:
        # Aligner rows in reading order, one per IMAGE slot.
        return self.aligner(self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)

    def _build_image_span(
        self, image_embeds: torch.Tensor, types: torch.Tensor
    ) -> torch.Tensor:
        """Full image span: aligner rows at IMAGE slots, the learned
        delimiter vectors at IMAGE_START/IMAGE_NEW_LINE/IMAGE_END."""
        types = types.to(image_embeds.device)
        span = image_embeds.new_empty(types.numel(), image_embeds.shape[-1])
        dtype = image_embeds.dtype
        span[types == IMAGE_START] = self.image_start.to(dtype)
        span[types == IMAGE_END] = self.image_end.to(dtype)
        span[types == IMAGE_NEW_LINE] = self.image_newline.to(dtype)
        span[types == IMAGE] = image_embeds
        return span

    def _process_image_input(
        self,
        image_input: DeepseekV4VLImagePixelInputs,
    ) -> tuple[torch.Tensor, ...]:
        patches = image_input.patches.to(self.aligner.w1.weight.dtype)
        vit_grid = image_input.vit_grid.tolist()

        image_embeds_list: list[torch.Tensor]
        if self.use_data_parallel and get_tensor_model_parallel_world_size() > 1:
            # Data-parallel ViT: shard images across TP ranks and all-gather
            # the per-image embeddings (weights are replicated on every rank).
            image_embeds_list = run_dp_sharded_vision_tower(
                self.vision, self.aligner, patches, vit_grid
            )
        else:
            image_embeds_list = []
            vit_offset = 0
            for n_vit_h, n_vit_w in vit_grid:
                n_vit = n_vit_h * n_vit_w
                image_embeds_list.append(
                    self._encode_image(
                        patches[vit_offset : vit_offset + n_vit], n_vit_h, n_vit_w
                    )
                )
                vit_offset += n_vit

        embeds: list[torch.Tensor] = []
        span_offset = 0
        for image_embeds, (n_llm_h, n_llm_w) in zip(
            image_embeds_list, image_input.llm_grid.tolist(), strict=True
        ):
            span_len = n_llm_h * (n_llm_w + 1) + 2
            embeds.append(
                self._build_image_span(
                    image_embeds,
                    image_input.types[span_offset : span_offset + span_len],
                )
            )
            span_offset += span_len
        return tuple(embeds)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []
        return self._process_image_input(image_input)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.model_executor.models.utils import _merge_multimodal_embeddings

        # Compressor-alignment pads borrow a reserved id; embed them as the
        # plain image token (the checkpoint has no image_pad vector).
        # Branch-free: safe inside compiled/captured regions.
        input_ids = input_ids.masked_fill(
            input_ids == IMAGE_PAD_ID, IMAGE_SENTINEL_BASE_ID
        )
        inputs_embeds = self.language_model.embed_input_ids(input_ids)

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        assert is_multimodal is not None
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    @staticmethod
    def get_model_state_cls():
        from ..nvidia.model_state import DeepseekV41ModelState

        return DeepseekV41ModelState

    @property
    def token_lookback_depth(self) -> int:
        return self.language_model.token_lookback_depth

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids=lookback_token_ids,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.compute_logits_local(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.language_model.get_expert_mapping()

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        """Pre-hc_head residual stream buffer for the MTP/DSpark draft model."""
        return self.language_model.get_mtp_target_hidden_states()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Map HF names into this wrapper's namespace up front and sort, so
        # the "language_model." group reaches the child loader as one
        # contiguous block (AutoWeightsLoader delegates per contiguous group,
        # and the child's load_weights finalizes fused expert weights, which
        # must not run on a partially loaded model).
        mapped = sorted(self.hf_to_vllm_mapper.apply(weights), key=lambda x: x[0])
        loader = AutoWeightsLoader(self)
        loaded_params = loader.load_weights(mapped)
        # The child's load_weights already ran its post-load finalization.
        self._weights_finalized = True
        return loaded_params

    def process_weights_after_loading(self) -> None:
        # Model-level post-load hook (called by the loader after any load
        # format). Under DummyModelLoader the child's load_weights — and
        # hence its finalize step — is bypassed, so run it here instead.
        if getattr(self, "_weights_finalized", False):
            return
        self.language_model.process_weights_after_loading()
