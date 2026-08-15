# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Intel XPU Kimi-K3 text-only serving model."""

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import (
	HasInnerState,
	IsHybrid,
	SupportsEagle3,
	SupportsMultiModal,
	SupportsPP,
	SupportsQuant,
)
from vllm.model_executor.models.utils import (
	AutoWeightsLoader,
	WeightsMapper,
	init_vllm_registered_model,
	maybe_prefix,
)
from vllm.multimodal.inputs import NestedTensors
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.kimi_k3 import KimiK3Config

from .linear import KimiLinearForCausalLM, KimiLinearModel


class KimiK3ForConditionalGeneration(
	nn.Module,
	SupportsMultiModal,
	SupportsPP,
	SupportsQuant,
	SupportsEagle3,
	HasInnerState,
	IsHybrid,
):
	"""Kimi-K3 language model serving on XPU without the vision tower."""

	packed_modules_mapping = KimiLinearModel.packed_modules_mapping
	hf_to_vllm_mapper = WeightsMapper(
		orig_to_new_prefix={
			"language_model.layers.": "language_model.model.layers.",
			"vision_tower.": None,
			"mm_projector.": None,
		}
	)

	def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
		super().__init__()
		self.model_config = vllm_config.model_config
		config: KimiK3Config = self.model_config.hf_config
		self.config = config
		self.quant_config = vllm_config.quant_config

		with self._mark_language_model(vllm_config):
			self.language_model = init_vllm_registered_model(
				vllm_config=vllm_config,
				hf_config=config.text_config,
				prefix=maybe_prefix(prefix, "language_model"),
				architectures=["KimiLinearForCausalLM"],
			)
		self.make_empty_intermediate_tensors = (
			self.language_model.make_empty_intermediate_tensors
		)

	@classmethod
	def get_placeholder_str(cls, modality: str, i: int) -> str | None:
		del i
		if modality == "image":
			return "<|kimi_image_placeholder|>"
		raise ValueError(f"Unsupported modality: {modality}")

	def embed_multimodal(self, **kwargs: object) -> NestedTensors | None:
		if kwargs.get("pixel_values") is None:
			return None
		raise NotImplementedError(
			"Kimi-K3 image inputs are not yet supported on XPU; "
			"start the server with --limit-mm-per-prompt '{\"image\": 0}'"
		)

	def forward(
		self,
		input_ids: torch.Tensor,
		positions: torch.Tensor,
		intermediate_tensors: IntermediateTensors | None = None,
		inputs_embeds: torch.Tensor | None = None,
		**kwargs: object,
	) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
		del kwargs
		if intermediate_tensors is not None:
			inputs_embeds = None
		return self.language_model(
			input_ids=input_ids,
			positions=positions,
			intermediate_tensors=intermediate_tensors,
			inputs_embeds=inputs_embeds,
		)

	def compute_logits(
		self,
		hidden_states: torch.Tensor,
		**kwargs: object,
	) -> torch.Tensor | None:
		del kwargs
		return self.language_model.compute_logits(hidden_states)

	def copy_inputs_before_cuda_graphs(
		self,
		input_buffers: dict[str, torch.Tensor],
		**kwargs: object,
	) -> None:
		self.language_model.mamba_cache.copy_inputs_before_cuda_graphs(
			input_buffers, **kwargs
		)

	def get_seqlen_agnostic_capture_inputs(
		self,
		batch_size: int,
	) -> dict[str, torch.Tensor]:
		return self.language_model.mamba_cache.get_seqlen_agnostic_capture_inputs(
			batch_size
		)

	@classmethod
	def get_mamba_state_dtype_from_config(
		cls,
		vllm_config: VllmConfig,
	) -> tuple[torch.dtype, torch.dtype]:
		text_config = vllm_config.model_config.hf_config.text_config
		return KimiLinearForCausalLM.get_mamba_state_dtype_from_config(
			vllm_config.with_hf_config(text_config)
		)

	@classmethod
	def get_mamba_state_shape_from_config(
		cls,
		vllm_config: VllmConfig,
	) -> tuple[tuple[int, int], tuple[int, int, int]]:
		text_config = vllm_config.model_config.hf_config.text_config
		return KimiLinearForCausalLM.get_mamba_state_shape_from_config(
			vllm_config.with_hf_config(text_config)
		)

	@classmethod
	def get_mamba_state_copy_func(cls):
		return KimiLinearForCausalLM.get_mamba_state_copy_func()

	def load_weights(
		self,
		weights: Iterable[tuple[str, torch.Tensor]],
	) -> set[str]:
		loader = AutoWeightsLoader(self)
		return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

__all__ = ["KimiK3ForConditionalGeneration"]