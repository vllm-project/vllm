# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 vision variant (e.g. DeepSeek-V4-Flash-Vision-Exp).

Thin multimodal wrapper around the text-only ``DeepseekV4ForCausalLM``:

- ``vision`` ViT + ``aligner`` produce per-image embeddings for the IMAGE
  sentinel positions; four learned vectors (``image_start`` / ``image_pad`` /
  ``image_newline`` / ``image_end``) fill the remaining sentinel positions.
- Image placeholders (``<｜deepseek_image｜>``) are expanded by the processor
  in ``common/mm_preprocess.py`` into sentinel blocks borrowing reserved
  in-vocab tokens ``<|place_holder_mm_span_0431|>``..``_0435|>``
  (see ``common/vision.py`` for the tower itself).
- Merged embeddings enter the text model via ``inputs_embeds``, i.e. before
  its hyper-connection stream expansion. Raw ``input_ids`` still flow into
  the model so the MoE router can apply ``bias_vl`` to image tokens
  (``requires_raw_input_tokens``).
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from ..common.mm_preprocess import (
    IMAGE_PLACEHOLDER,
    IMAGE_SENTINEL_BASE_ID,
    DeepseekV4VLDummyInputsBuilder,
    DeepseekV4VLMultiModalProcessor,
    DeepseekV4VLProcessingInfo,
    image_sentinel_mask,
)
from ..common.vision import DeepseekV4Aligner, DeepseekV4ViT
from .model import _make_deepseek_v4_weights_mapper


def _make_deepseek_v4_vl_weights_mapper(
    expert_dtype: str, image_enabled: bool
) -> WeightsMapper:
    """Text-checkpoint mapping rules re-rooted under ``language_model.``."""
    base = _make_deepseek_v4_weights_mapper(expert_dtype)
    orig_to_new_prefix: dict[str, str | None] = {
        "layers.": "language_model.model.layers.",
        "embed.": "language_model.model.embed.",
        "norm.": "language_model.model.norm.",
        "hc_head": "language_model.model.hc_head",
        "mtp.": "language_model.model.mtp.",
    }
    if not image_enabled:
        orig_to_new_prefix.update({"vision.": None, "aligner.": None, "image_": None})
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
class DeepseekV4ForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsEagle3
):
    """Multimodal entry point for DeepSeek-V4 checkpoints with a vision tower.

    ``SupportsEagle3`` (aux hidden-state plumbing for MTP/DSpark drafters)
    delegates through ``language_model`` via the protocol defaults.
    """

    # The MoE router needs raw token ids to detect image sentinel tokens
    # (borrowed reserved ids, see common/mm_preprocess.py) and apply bias_vl.
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

        image_enabled = (
            config.vision_n_layers > 0
            and self.multimodal_config.get_limit_per_prompt("image") > 0
        )
        with self._mark_tower_model(vllm_config, {"image"}):
            self.vision: DeepseekV4ViT | None = None
            self.aligner: DeepseekV4Aligner | None = None
            self.image_start: nn.Parameter | None = None
            self.image_end: nn.Parameter | None = None
            self.image_newline: nn.Parameter | None = None
            self.image_pad: nn.Parameter | None = None
            if image_enabled:
                self.vision = DeepseekV4ViT(config)
                self.aligner = DeepseekV4Aligner(config)
                for name in (
                    "image_start",
                    "image_end",
                    "image_newline",
                    "image_pad",
                ):
                    setattr(
                        self,
                        name,
                        nn.Parameter(
                            torch.empty(config.hidden_size, dtype=torch.float32)
                        ),
                    )
                self.vision.to(dtype=model_config.dtype)
                self.aligner.to(dtype=model_config.dtype)

        with self._mark_language_model(vllm_config):
            # The arch convertor routes any config with a vision tower to
            # this wrapper class; mark the copy handed to the text backbone
            # so it resolves to DeepseekV4ForCausalLM instead of recursing
            # (with_hf_config deepcopies the config, the marker survives).
            config._dsv4_vl_inner = True  # type: ignore[attr-defined]
            try:
                self.language_model = init_vllm_registered_model(
                    vllm_config=vllm_config,
                    hf_config=config,
                    prefix=maybe_prefix(prefix, "language_model"),
                    architectures=["DeepseekV4ForCausalLM"],
                )
            finally:
                del config._dsv4_vl_inner  # type: ignore[attr-defined]
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
            expert_dtype, image_enabled
        )

    def _parse_and_validate_image_input(self, **kwargs: object) -> dict | None:
        patches = kwargs.pop("patches", None)
        if patches is None:
            return None
        vit_grid = kwargs.pop("vit_grid", None)
        llm_grid = kwargs.pop("llm_grid", None)
        perm = kwargs.pop("perm", None)
        assert vit_grid is not None and llm_grid is not None and perm is not None
        return {
            "patches": patches,
            "vit_grid": vit_grid,
            "llm_grid": llm_grid,
            "perm": perm,
        }

    def _encode_image(
        self,
        patches: torch.Tensor,
        n_vit_h: int,
        n_vit_w: int,
        perm: torch.Tensor,
    ) -> torch.Tensor:
        assert self.vision is not None and self.aligner is not None
        image_embeds = self.aligner(
            self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w
        )
        # Reorder into the N-layout block order used in the prompt.
        return image_embeds[perm.to(image_embeds.device)]

    def _process_image_input(
        self,
        patches: torch.Tensor,
        vit_grid: torch.Tensor,
        llm_grid: torch.Tensor,
        perm: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        assert self.vision is not None and self.aligner is not None
        patches = patches.to(self.aligner.w1.weight.dtype)

        embeds: list[torch.Tensor] = []
        vit_offset = 0
        llm_offset = 0
        for (n_vit_h, n_vit_w), (n_llm_h, n_llm_w) in zip(
            vit_grid.tolist(), llm_grid.tolist(), strict=True
        ):
            n_vit = n_vit_h * n_vit_w
            n_llm = n_llm_h * n_llm_w
            embeds.append(
                self._encode_image(
                    patches[vit_offset : vit_offset + n_vit],
                    n_vit_h,
                    n_vit_w,
                    perm[llm_offset : llm_offset + n_llm],
                )
            )
            vit_offset += n_vit
            llm_offset += n_llm
        return tuple(embeds)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None or self.vision is None:
            return []
        return self._process_image_input(
            image_input["patches"],
            image_input["vit_grid"],
            image_input["llm_grid"],
            image_input["perm"],
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.model_executor.models.utils import _merge_multimodal_embeddings

        # All ids are in-vocab here: image-block sentinels are borrowed
        # reserved tokens (their embedding rows are always overwritten below).
        inputs_embeds = self.language_model.embed_input_ids(input_ids)

        if self.image_start is not None:
            # Branch-free sentinel overwrite: safe inside compiled/captured
            # regions (no data-dependent control flow).
            sentinel_mask = image_sentinel_mask(input_ids)
            if is_multimodal is not None:
                # IMAGE positions get vision embeddings via the merge below.
                sentinel_mask = sentinel_mask & ~is_multimodal.to(input_ids.device)
            table = torch.stack(
                [
                    self.image_start,
                    self.image_pad,
                    self.image_pad,
                    self.image_newline,
                    self.image_end,
                ]
            ).to(inputs_embeds.dtype)
            idx = (input_ids - IMAGE_SENTINEL_BASE_ID).clamp(0, 4)
            inputs_embeds = torch.where(
                sentinel_mask.unsqueeze(-1), table[idx], inputs_embeds
            )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        assert is_multimodal is not None
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds
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
